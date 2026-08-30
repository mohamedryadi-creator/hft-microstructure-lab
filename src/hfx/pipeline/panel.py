"""The symbol panel, and why these twelve.

Most of the results in this repository are cross-sectional in one variable: the
ratio of the *spread* to the *tick*.  Nasdaq's tick is $0.01 for every stock
above $1, so that ratio is set by the price level: a $6 stock has a tick worth
17 basis points and its spread is pinned at one tick almost all day (a "large
tick" asset); a $1 700 stock has a tick worth 0.06 bp and its spread is tens of
ticks wide (a "small tick" asset).  The two regimes have visibly different
price formation -- queues matter in one, they barely exist in the other -- and a
panel that spans only one of them cannot show it.

The twelve below run from $6 to $1 700 on 2019-01-30, are all Nasdaq-listed and
liquid enough that a single day gives tens of thousands of trades.
"""

from __future__ import annotations

#: symbol -> approximate close on 2019-01-30, in dollars.  Indicative only; the
#: pipeline measures the real price and the real spread.
PANEL: dict[str, float] = {
    "SIRI": 6.0,      # tick = 167 bp: the most extreme large-tick name here
    "MU": 40.0,
    "INTC": 47.0,
    "CSCO": 48.0,
    "MSFT": 105.0,
    "AAPL": 165.0,
    "TSLA": 308.0,
    "NFLX": 340.0,
    "REGN": 400.0,
    "ISRG": 500.0,
    "GOOG": 1_100.0,
    "AMZN": 1_670.0,  # tick = 0.06 bp: the most extreme small-tick name here
}

SYMBOLS: tuple[str, ...] = tuple(PANEL)

#: A cheap subset, for a quick end-to-end run.
QUICK_SYMBOLS: tuple[str, ...] = ("SIRI", "INTC", "MSFT", "AMZN")
