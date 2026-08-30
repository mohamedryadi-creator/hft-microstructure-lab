"""Nasdaq TotalView-ITCH 5.0 wire format.

The exchange publishes the feed as a *BinaryFILE*: a concatenation of records,
each one a two-byte big-endian length followed by that many bytes of message.
Byte 0 of a message is its type; bytes 1-2 are the ``stock_locate`` (an integer
handle for the ticker, assigned by the ``R`` messages at the head of the file);
bytes 3-4 are a tracking number; bytes 5-10 are a six-byte big-endian timestamp
in **nanoseconds since midnight** US/Eastern.

``MSG_LEN`` below was checked against the real feed rather than copied: every
length in the 2019-01-30 file agrees with this table, and :func:`hfx.itch.reader`
raises if it ever meets a length it does not expect.  A silent disagreement here
would misalign the whole stream, so it is worth failing loudly on.
"""

from __future__ import annotations

# Message body length in bytes, *excluding* the two-byte length prefix.
MSG_LEN: dict[str, int] = {
    "S": 12,   # system event
    "R": 39,   # stock directory
    "H": 25,   # stock trading action
    "Y": 20,   # Reg SHO short sale price test restriction
    "L": 26,   # market participant position
    "V": 35,   # MWCB decline level
    "W": 12,   # MWCB status
    "K": 28,   # IPO quoting period update
    "J": 35,   # LULD auction collar
    "h": 21,   # operational halt
    "A": 36,   # add order, no attribution
    "F": 40,   # add order with attribution (MPID)
    "E": 31,   # order executed
    "C": 36,   # order executed with price
    "X": 23,   # order cancel (partial)
    "D": 19,   # order delete (full)
    "U": 35,   # order replace
    "P": 44,   # trade, non-cross (a hidden or non-displayed execution)
    "Q": 40,   # cross trade (open, close, halt cross)
    "B": 19,   # broken trade
    "I": 50,   # net order imbalance indicator
    "N": 20,   # retail price improvement indicator
}

#: Prices on the wire are unsigned 4-byte integers in units of 1/10 000 dollar.
PRICE_SCALE = 10_000

#: Nanoseconds since midnight for the regular US equity session.
OPEN_NS = int(9.5 * 3600 * 1e9)
CLOSE_NS = int(16.0 * 3600 * 1e9)

#: Event codes we keep per symbol.  ``ord`` of the ITCH message type.
KEPT_TYPES = tuple("AFECXDUPQH")
