"""Stream one ITCH day and write the panel's messages to parquet.

Disk, not time, is the binding constraint: the seven published days are 31 GB
gzipped and about 90 GB inflated.  Nothing here touches disk except the columns
we keep, and those are flushed in row-group-sized batches so peak memory stays
around a hundred megabytes whatever the day looks like.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

from ..itch.reader import EventBuffer, ItchExtractor
from ..itch.source import iter_inflated, itch_url, local_copy

FLUSH_ROWS = 1_000_000
PROGRESS_BYTES = 1 << 31  # report every 2 GB inflated

_ARROW_TYPE = {
    "ts": "int64",
    "etype": "uint8",
    "ref": "uint64",
    "aux": "uint64",
    "side": "int8",
    "shares": "int32",
    "price": "int32",
    "flag": "uint8",
}


def _schema():
    import pyarrow as pa

    return pa.schema([(name, getattr(pa, t)()) for name, t in _ARROW_TYPE.items()])


def _table(buf: EventBuffer):
    import numpy as np
    import pyarrow as pa

    cols = {}
    for name, arr in buf.as_dict().items():
        cols[name] = pa.array(np.frombuffer(arr, dtype=_ARROW_TYPE[name]))
    return pa.table(cols, schema=_schema())


def events_path(outdir: str, symbol: str, date: str) -> str:
    return os.path.join(outdir, f"{symbol}_{date}_events.parquet")


def build_day(
    date: str,
    symbols: Sequence[str],
    outdir: str,
    raw_dir: str | None = None,
    progress: bool = True,
) -> dict:
    """Extract ``symbols`` from the ITCH file for ``date`` into ``outdir``.

    Returns a summary dict with message counts and timings.  Re-running is a
    no-op for symbols whose parquet file already exists -- the whole set is
    skipped only if *every* symbol is already there, since the stream has to be
    read from the start either way.
    """
    import pyarrow.parquet as pq

    os.makedirs(outdir, exist_ok=True)
    todo = [s for s in symbols if not os.path.exists(events_path(outdir, s, date))]
    if not todo:
        return {"date": date, "skipped": True, "symbols": list(symbols)}

    source = (raw_dir and local_copy(date, raw_dir)) or itch_url(date)
    extractor = ItchExtractor(todo)
    writers = {}
    schema = _schema()
    t0 = time.time()
    n_bytes = 0
    reported = 0

    def flush(symbol: str, force: bool = False) -> None:
        buf = extractor.buffers[symbol]
        if len(buf) == 0 or (len(buf) < FLUSH_ROWS and not force):
            return
        if symbol not in writers:
            writers[symbol] = pq.ParquetWriter(
                events_path(outdir, symbol, date) + ".part",
                schema,
                compression="zstd",
            )
        writers[symbol].write_table(_table(buf))
        buf.clear()

    try:
        for chunk in iter_inflated(source):
            n_bytes += len(chunk)
            extractor.feed(chunk)
            for symbol in todo:
                flush(symbol)
            if progress and n_bytes // PROGRESS_BYTES > reported:
                reported = n_bytes // PROGRESS_BYTES
                el = time.time() - t0
                print(
                    f"  {date}  {n_bytes / 1e9:6.2f} GB  "
                    f"{extractor.n_messages / 1e6:7.1f}M msgs  "
                    f"{extractor.n_kept / 1e6:6.2f}M kept  {el / 60:5.1f} min",
                    flush=True,
                )
        for symbol in todo:
            flush(symbol, force=True)
    finally:
        for w in writers.values():
            w.close()
    missing = extractor.missing()
    for symbol in todo:
        part = events_path(outdir, symbol, date) + ".part"
        if os.path.exists(part):
            os.replace(part, events_path(outdir, symbol, date))

    return {
        "date": date,
        "skipped": False,
        "symbols": list(todo),
        "missing": missing,
        "n_messages": extractor.n_messages,
        "n_kept": extractor.n_kept,
        "inflated_gb": n_bytes / 1e9,
        "minutes": (time.time() - t0) / 60,
        "system_events": extractor.system_events,
    }
