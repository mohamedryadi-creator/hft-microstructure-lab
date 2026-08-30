"""Streaming decoder for the Nasdaq TotalView-ITCH 5.0 BinaryFILE format.

The design constraint is that a single trading day is ~400 million messages and
~12 GB uncompressed, and that we want a handful of symbols out of 8 713.  So the
decoder

* consumes the gzip stream in chunks and never materialises the day,
* keeps a ``set`` of the ``stock_locate`` handles we care about and skips every
  other message with two byte reads and a set lookup,
* writes the messages it keeps into flat typed columns (``array.array``), which
  the pipeline flushes to parquet.

Storage layout, one row per kept message
----------------------------------------
=========  =======  ====================================================
column     dtype    meaning
=========  =======  ====================================================
``ts``     int64    nanoseconds since midnight (US/Eastern)
``etype``  uint8    ``ord`` of the ITCH message type
``ref``    uint64   order reference (the *original* one for ``U``)
``aux``    uint64   ``U``: new order reference.  ``E``/``C``/``P``/``Q``:
                    match number.  Otherwise 0.
``side``   int8     +1 buy, -1 sell, 0 when the message does not carry one
``shares`` int32    order or execution size
``price``  int32    price in 1/10 000 dollar, 0 when not carried
``flag``   uint8    ``C``: printable.  ``Q``: cross type.  ``H``: trading
                    state.  Otherwise 0.
=========  =======  ====================================================

``side`` is 0 for ``E``, ``C``, ``X``, ``D`` and ``U`` because the wire format
does not repeat it; it belongs to the resting order and is recovered by the book
builder, which is the only component that knows the order map.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field

from .spec import MSG_LEN

_LEN_BY_CODE = [0] * 256
for _t, _n in MSG_LEN.items():
    _LEN_BY_CODE[ord(_t)] = _n

# ord() of the message types, hoisted out of the hot loop.
_A, _F, _E, _C, _X, _D, _U, _P, _Q, _H, _R, _S = (
    ord(c) for c in "AFECXDUPQHRS"
)
_BUY = ord("B")


@dataclass
class EventBuffer:
    """Column store for the messages kept for one symbol."""

    ts: array = field(default_factory=lambda: array("q"))
    etype: array = field(default_factory=lambda: array("B"))
    ref: array = field(default_factory=lambda: array("Q"))
    aux: array = field(default_factory=lambda: array("Q"))
    side: array = field(default_factory=lambda: array("b"))
    shares: array = field(default_factory=lambda: array("i"))
    price: array = field(default_factory=lambda: array("i"))
    flag: array = field(default_factory=lambda: array("B"))

    COLUMNS = ("ts", "etype", "ref", "aux", "side", "shares", "price", "flag")

    def __len__(self) -> int:
        return len(self.ts)

    def push(self, ts, etype, ref, aux, side, shares, price, flag) -> None:
        self.ts.append(ts)
        self.etype.append(etype)
        self.ref.append(ref)
        self.aux.append(aux)
        self.side.append(side)
        self.shares.append(shares)
        self.price.append(price)
        self.flag.append(flag)

    def clear(self) -> None:
        for name in self.COLUMNS:
            del getattr(self, name)[:]

    def as_dict(self) -> dict[str, array]:
        return {name: getattr(self, name) for name in self.COLUMNS}


class ItchExtractor:
    """Decode an ITCH 5.0 byte stream, keeping only ``symbols``.

    Feed it consecutive chunks of the *decompressed* stream with :meth:`feed`;
    partial messages spanning a chunk boundary are carried over internally.

    Parameters
    ----------
    symbols
        Tickers to keep.  Matched against the ``R`` stock-directory messages at
        the head of the file, so a symbol that never appears there simply yields
        an empty buffer -- which :meth:`missing` reports.
    """

    def __init__(self, symbols):
        self.symbols = [s.upper() for s in symbols]
        self._wanted = set(self.symbols)
        self.buffers: dict[str, EventBuffer] = {s: EventBuffer() for s in self.symbols}
        self.locate_to_symbol: dict[int, str] = {}
        self._targets: dict[int, EventBuffer] = {}
        self.system_events: list[tuple[int, str]] = []
        self.n_messages = 0
        self.n_kept = 0
        self._residual = b""

    # -- introspection -----------------------------------------------------
    def missing(self) -> list[str]:
        """Requested symbols the stock directory never mentioned."""
        seen = set(self.locate_to_symbol.values())
        return sorted(self._wanted - seen)

    # -- the hot loop ------------------------------------------------------
    def feed(self, chunk: bytes) -> None:
        buf = self._residual + chunk if self._residual else chunk
        n = len(buf)
        i = 0
        targets = self._targets
        while True:
            if i + 2 > n:
                break
            ln = (buf[i] << 8) | buf[i + 1]
            j = i + 2
            end = j + ln
            if end > n:
                break
            code = buf[j]
            # Checked on *every* message, not just the kept ones: a single wrong
            # length silently shifts the rest of the day by a few bytes and every
            # number downstream becomes garbage that still looks like data.
            if _LEN_BY_CODE[code] != ln:
                raise ValueError(
                    f"ITCH message {chr(code)!r} at byte {i} has length {ln}, "
                    f"expected {_LEN_BY_CODE[code]}: the stream is misaligned"
                )
            hit = targets.get((buf[j + 1] << 8) | buf[j + 2])
            if hit is not None:
                self._decode(hit, code, buf, j)
            elif code == _R:
                self._directory(buf, j)
            elif code == _S:
                self.system_events.append(
                    (int.from_bytes(buf[j + 5 : j + 11], "big"), chr(buf[j + 11]))
                )
            self.n_messages += 1
            i = end
        self._residual = buf[i:] if i < n else b""

    def _directory(self, buf, j) -> None:
        symbol = buf[j + 11 : j + 19].decode("ascii").strip()
        locate = (buf[j + 1] << 8) | buf[j + 2]
        self.locate_to_symbol[locate] = symbol
        if symbol in self._wanted:
            self._targets[locate] = self.buffers[symbol]

    def _decode(self, out: EventBuffer, code: int, buf, j: int) -> None:
        ts = int.from_bytes(buf[j + 5 : j + 11], "big")
        if code == _A or code == _F:
            out.push(
                ts, code,
                int.from_bytes(buf[j + 11 : j + 19], "big"), 0,
                1 if buf[j + 19] == _BUY else -1,
                int.from_bytes(buf[j + 20 : j + 24], "big"),
                int.from_bytes(buf[j + 32 : j + 36], "big"), 0,
            )
        elif code == _E:
            out.push(
                ts, code,
                int.from_bytes(buf[j + 11 : j + 19], "big"),
                int.from_bytes(buf[j + 23 : j + 31], "big"), 0,
                int.from_bytes(buf[j + 19 : j + 23], "big"), 0, 0,
            )
        elif code == _X:
            out.push(
                ts, code,
                int.from_bytes(buf[j + 11 : j + 19], "big"), 0, 0,
                int.from_bytes(buf[j + 19 : j + 23], "big"), 0, 0,
            )
        elif code == _D:
            out.push(ts, code, int.from_bytes(buf[j + 11 : j + 19], "big"), 0, 0, 0, 0, 0)
        elif code == _U:
            out.push(
                ts, code,
                int.from_bytes(buf[j + 11 : j + 19], "big"),
                int.from_bytes(buf[j + 19 : j + 27], "big"), 0,
                int.from_bytes(buf[j + 27 : j + 31], "big"),
                int.from_bytes(buf[j + 31 : j + 35], "big"), 0,
            )
        elif code == _C:
            out.push(
                ts, code,
                int.from_bytes(buf[j + 11 : j + 19], "big"),
                int.from_bytes(buf[j + 23 : j + 31], "big"), 0,
                int.from_bytes(buf[j + 19 : j + 23], "big"),
                int.from_bytes(buf[j + 32 : j + 36], "big"),
                buf[j + 31],
            )
        elif code == _P:
            out.push(
                ts, code,
                int.from_bytes(buf[j + 11 : j + 19], "big"),
                int.from_bytes(buf[j + 36 : j + 44], "big"),
                1 if buf[j + 19] == _BUY else -1,
                int.from_bytes(buf[j + 20 : j + 24], "big"),
                int.from_bytes(buf[j + 32 : j + 36], "big"), 0,
            )
        elif code == _Q:
            shares = int.from_bytes(buf[j + 11 : j + 19], "big")
            out.push(
                ts, code, 0,
                int.from_bytes(buf[j + 31 : j + 39], "big"), 0,
                min(shares, 2_147_483_647),
                int.from_bytes(buf[j + 27 : j + 31], "big"),
                buf[j + 39],
            )
        elif code == _H:
            out.push(ts, code, 0, 0, 0, 0, 0, buf[j + 19])
        else:
            return
        self.n_kept += 1
