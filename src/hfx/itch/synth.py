"""Build a synthetic ITCH 5.0 byte stream.

Every claim this repository makes about the real feed rests on the decoder
reading the binary correctly, and the real feed cannot be committed or reached
from CI.  So the decoder is tested against a stream produced *here*, from the
published field offsets, by an encoder written independently of the decoder: if
the two disagree about where the price sits, the test says so.

It also lets notebook 01 show the format on a book small enough to check by eye.
"""

from __future__ import annotations

import struct


def _head(msg_type: str, locate: int, ts: int) -> bytes:
    return (
        msg_type.encode("ascii")
        + struct.pack(">HH", locate, 0)
        + ts.to_bytes(6, "big")
    )


def system_event(ts: int, code: str = "Q") -> bytes:
    return _head("S", 0, ts) + code.encode("ascii")


def stock_directory(locate: int, symbol: str, ts: int = 0) -> bytes:
    return (
        _head("R", locate, ts)
        + symbol.ljust(8).encode("ascii")
        + b"Q"          # market category
        + b" "          # financial status
        + struct.pack(">I", 100)
        + b"N"          # round lots only
        + b"C"          # issue classification
        + b"  "         # issue subtype
        + b"P"          # authenticity
        + b"N"          # short sale threshold
        + b"N"          # IPO flag
        + b"1"          # LULD reference price tier
        + b"N"          # ETP flag
        + struct.pack(">I", 0)
        + b"N"          # inverse indicator
    )


def add_order(locate, ts, ref, side, shares, price, mpid=None) -> bytes:
    body = (
        struct.pack(">Q", ref)
        + (b"B" if side > 0 else b"S")
        + struct.pack(">I", shares)
        + b"        "                      # stock, unused by the decoder
        + struct.pack(">I", price)
    )
    if mpid is None:
        return _head("A", locate, ts) + body
    return _head("F", locate, ts) + body + mpid.ljust(4).encode("ascii")


def execute(locate, ts, ref, shares, match) -> bytes:
    return _head("E", locate, ts) + struct.pack(">QIQ", ref, shares, match)


def execute_with_price(locate, ts, ref, shares, match, price, printable=True) -> bytes:
    return (
        _head("C", locate, ts)
        + struct.pack(">Q", ref)
        + struct.pack(">IQ", shares, match)
        + (b"Y" if printable else b"N")
        + struct.pack(">I", price)
    )


def cancel(locate, ts, ref, shares) -> bytes:
    return _head("X", locate, ts) + struct.pack(">QI", ref, shares)


def delete(locate, ts, ref) -> bytes:
    return _head("D", locate, ts) + struct.pack(">Q", ref)


def replace(locate, ts, old_ref, new_ref, shares, price) -> bytes:
    return _head("U", locate, ts) + struct.pack(">QQII", old_ref, new_ref, shares, price)


def hidden_trade(locate, ts, ref, side, shares, price, match) -> bytes:
    return (
        _head("P", locate, ts)
        + struct.pack(">Q", ref)
        + (b"B" if side > 0 else b"S")
        + struct.pack(">I", shares)
        + b"        "
        + struct.pack(">I", price)
        + struct.pack(">Q", match)
    )


def cross_trade(locate, ts, shares, price, match, cross_type="O") -> bytes:
    return (
        _head("Q", locate, ts)
        + struct.pack(">Q", shares)
        + b"        "
        + struct.pack(">I", price)
        + struct.pack(">Q", match)
        + cross_type.encode("ascii")
    )


def trading_action(locate, ts, state="T") -> bytes:
    return (
        _head("H", locate, ts)
        + b"        "
        + state.encode("ascii")
        + b" "
        + b"    "
    )


def frame(messages) -> bytes:
    """Prefix each message with its two-byte big-endian length."""
    return b"".join(struct.pack(">H", len(m)) + m for m in messages)
