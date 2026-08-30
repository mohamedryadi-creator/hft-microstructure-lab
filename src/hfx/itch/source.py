"""Where the ITCH files live, and how to read one without storing it.

Nasdaq publishes a handful of complete TotalView-ITCH 5.0 days for free at
``https://emi.nasdaq.com/ITCH/Nasdaq ITCH/``.  They are 3.5-5.6 GB gzipped and
11-14 GB inflated, so the reader here is a generator of inflated chunks: the
pipeline consumes it, keeps the twelve symbols it wants, and the rest is never
written anywhere.
"""

from __future__ import annotations

import os
import time
import urllib.request
import zlib
from collections.abc import Iterator

BASE = "https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/"

#: The seven complete days published, ISO date -> file stem (MMDDYYYY).
DAYS: dict[str, str] = {
    "2019-01-30": "01302019",
    "2019-03-27": "03272019",
    "2019-07-30": "07302019",
    "2019-08-30": "08302019",
    "2019-10-30": "10302019",
    "2019-12-30": "12302019",
    "2020-01-30": "01302020",
}


def itch_url(date: str) -> str:
    """URL of the ITCH file for an ISO date such as ``"2019-01-30"``."""
    if date not in DAYS:
        raise KeyError(f"no published ITCH file for {date}; have {sorted(DAYS)}")
    return f"{BASE}{DAYS[date]}.NASDAQ_ITCH50.gz"


def iter_inflated(
    source: str, chunk_size: int = 1 << 23, retries: int = 4
) -> Iterator[bytes]:
    """Yield inflated chunks of a gzipped ITCH file, local path or URL.

    A dropped connection restarts the whole transfer rather than resuming: the
    gzip stream carries state across the whole file, so a byte-range resume would
    need the decompressor state too.  A day is ten minutes; a wrong resume would
    be silent corruption.
    """
    if not source.startswith(("http://", "https://")):
        with open(source, "rb") as fh:
            dec = zlib.decompressobj(zlib.MAX_WBITS | 16)
            while True:
                raw = fh.read(chunk_size)
                if not raw:
                    break
                out = dec.decompress(raw)
                if out:
                    yield out
        return

    last: Exception | None = None
    for attempt in range(retries):
        try:
            dec = zlib.decompressobj(zlib.MAX_WBITS | 16)
            req = urllib.request.Request(source, headers={"User-Agent": "hfx/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                while True:
                    raw = resp.read(chunk_size)
                    if not raw:
                        break
                    out = dec.decompress(raw)
                    if out:
                        yield out
            return
        except Exception as exc:  # network, timeout, truncated stream
            last = exc
            if attempt + 1 < retries:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not stream {source} after {retries} attempts") from last


def local_copy(date: str, directory: str) -> str | None:
    """Path of an already-downloaded ITCH file, if one is sitting there."""
    path = os.path.join(directory, f"{DAYS[date]}.NASDAQ_ITCH50.gz")
    return path if os.path.exists(path) else None
