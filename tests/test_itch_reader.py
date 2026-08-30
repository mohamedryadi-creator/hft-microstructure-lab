"""The decoder against an independently written encoder.

Nothing downstream survives a misread of the binary, and the real feed is 4 GB
of it that CI cannot see.  So the encoder in ``hfx.itch.synth`` writes the field
offsets out from the published specification, the decoder reads them back, and
the two are compared field by field.
"""

import struct

import pytest

from hfx.itch import synth
from hfx.itch.reader import ItchExtractor
from hfx.itch.spec import MSG_LEN


def test_encoder_lengths_match_the_specification():
    built = {
        "S": synth.system_event(1),
        "R": synth.stock_directory(1, "AAPL"),
        "A": synth.add_order(1, 1, 1, 1, 100, 1_000_000),
        "F": synth.add_order(1, 1, 1, 1, 100, 1_000_000, mpid="XYZ"),
        "E": synth.execute(1, 1, 1, 100, 7),
        "C": synth.execute_with_price(1, 1, 1, 100, 7, 1_000_000),
        "X": synth.cancel(1, 1, 1, 50),
        "D": synth.delete(1, 1, 1),
        "U": synth.replace(1, 1, 1, 2, 100, 1_000_000),
        "P": synth.hidden_trade(1, 1, 1, 1, 100, 1_000_000, 7),
        "Q": synth.cross_trade(1, 1, 100, 1_000_000, 7),
        "H": synth.trading_action(1, 1),
    }
    for msg_type, message in built.items():
        assert len(message) == MSG_LEN[msg_type], msg_type


def sample_stream():
    """Two symbols, one of which we do not ask for, plus system messages."""
    messages = [
        synth.stock_directory(1, "AAPL"),
        synth.stock_directory(2, "SIRI"),
        synth.system_event(int(4 * 3600e9), "O"),
        synth.system_event(int(9.5 * 3600e9), "Q"),
        synth.trading_action(1, int(9.5 * 3600e9), "T"),
        synth.add_order(1, 1_000, 10, +1, 300, 1_500_000),
        synth.add_order(1, 2_000, 11, -1, 200, 1_500_100, mpid="NSDQ"),
        synth.add_order(2, 2_500, 12, +1, 5_000, 60_000),   # not requested
        synth.execute(1, 3_000, 10, 100, 555),
        synth.cancel(1, 4_000, 11, 50),
        synth.replace(1, 5_000, 11, 13, 150, 1_500_200),
        synth.delete(1, 6_000, 13),
        synth.execute_with_price(1, 7_000, 10, 200, 556, 1_499_900, printable=False),
        synth.hidden_trade(1, 8_000, 0, -1, 400, 1_500_050, 557),
        synth.cross_trade(1, 9_000, 1_000_000, 1_500_000, 558, "O"),
    ]
    return synth.frame(messages)


def decode(chunks):
    extractor = ItchExtractor(["AAPL"])
    for chunk in chunks:
        extractor.feed(chunk)
    return extractor


def test_fields_round_trip():
    ext = decode([sample_stream()])
    buf = ext.buffers["AAPL"]
    assert ext.missing() == []
    assert ext.locate_to_symbol == {1: "AAPL", 2: "SIRI"}
    assert [chr(c) for c in buf.etype] == list("HAFEXUDCPQ")
    assert list(buf.ts) == [
        int(9.5 * 3600e9), 1_000, 2_000, 3_000, 4_000,
        5_000, 6_000, 7_000, 8_000, 9_000,
    ]

    rows = dict(zip([chr(c) for c in buf.etype], range(len(buf.etype))))
    i = rows["A"]
    assert (buf.ref[i], buf.side[i], buf.shares[i], buf.price[i]) == (10, 1, 300, 1_500_000)
    i = rows["F"]
    assert (buf.ref[i], buf.side[i], buf.shares[i], buf.price[i]) == (11, -1, 200, 1_500_100)
    i = rows["E"]
    assert (buf.ref[i], buf.shares[i], buf.aux[i], buf.side[i]) == (10, 100, 555, 0)
    i = rows["X"]
    assert (buf.ref[i], buf.shares[i]) == (11, 50)
    i = rows["U"]
    assert (buf.ref[i], buf.aux[i], buf.shares[i], buf.price[i]) == (11, 13, 150, 1_500_200)
    i = rows["D"]
    assert buf.ref[i] == 13
    i = rows["C"]
    assert (buf.ref[i], buf.shares[i], buf.aux[i], buf.price[i]) == (10, 200, 556, 1_499_900)
    assert chr(buf.flag[i]) == "N"
    i = rows["P"]
    assert (buf.side[i], buf.shares[i], buf.price[i], buf.aux[i]) == (-1, 400, 1_500_050, 557)
    i = rows["Q"]
    assert (buf.shares[i], buf.price[i], buf.aux[i]) == (1_000_000, 1_500_000, 558)
    assert chr(buf.flag[i]) == "O"
    i = rows["H"]
    assert chr(buf.flag[i]) == "T"

    # System events carry no stock_locate and are collected separately.
    assert ext.system_events == [(int(4 * 3600e9), "O"), (int(9.5 * 3600e9), "Q")]


@pytest.mark.parametrize("size", [1, 2, 3, 7, 13, 64, 4096])
def test_chunk_boundaries_are_invisible(size):
    """A message split across two reads must decode exactly as an unsplit one."""
    stream = sample_stream()
    whole = decode([stream]).buffers["AAPL"]
    split = decode(
        [stream[i : i + size] for i in range(0, len(stream), size)]
    ).buffers["AAPL"]
    assert whole.as_dict().keys() == split.as_dict().keys()
    for name in whole.COLUMNS:
        assert list(getattr(whole, name)) == list(getattr(split, name)), name


def test_untracked_symbols_are_skipped_but_counted():
    ext = decode([sample_stream()])
    assert len(ext.buffers["AAPL"]) == 10
    assert ext.n_kept == 10
    assert ext.n_messages == 15


def test_missing_symbol_is_reported():
    ext = ItchExtractor(["AAPL", "NOPE"])
    ext.feed(sample_stream())
    assert ext.missing() == ["NOPE"]
    assert len(ext.buffers["NOPE"]) == 0


def test_a_wrong_length_prefix_raises_rather_than_drifting():
    stream = bytearray(sample_stream())
    # The first record is a 39-byte stock directory; claim it is 38.
    stream[0:2] = struct.pack(">H", 38)
    with pytest.raises(ValueError, match="misaligned"):
        ItchExtractor(["AAPL"]).feed(bytes(stream))
