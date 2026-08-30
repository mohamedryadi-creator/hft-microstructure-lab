"""A price-time priority book driven by order-by-order messages.

The exchange does not publish the book; it publishes the *changes* to it, one
order at a time.  Rebuilding the book is therefore the first thing that has to
be right, and the easiest thing to get subtly wrong -- a leaked order, a
mishandled replace, and the spread is wrong for the rest of the day in a way
that still looks like a plausible spread.

Design notes
------------
Levels are a ``dict`` from price (in 1/10 000 dollar) to resting size, and the
best price on each side comes from a heap with lazy deletion: a price is pushed
when its level is created and popped only once it is found empty at the top.
Stale entries buried in the heap would otherwise accumulate over a day of
churn, so the heap is rebuilt whenever it grows past a few times the number of
live levels.

Queue position is tracked per order: ``queue_ahead`` is the size resting at the
same price when the order arrived.  Because Nasdaq's priority at a price level
is strictly time-based for displayed orders, that number is exactly the volume
that must trade or cancel ahead of it.  It is what makes the fill-probability
measurement in chapter 05 possible at all.
"""

from __future__ import annotations

import heapq


class OrderBook:
    """Displayed limit order book for one symbol."""

    __slots__ = ("orders", "levels", "_heaps", "_rebuilt")

    def __init__(self) -> None:
        # ref -> [side, price, shares, queue_ahead, ts_added]
        self.orders: dict[int, list] = {}
        # side index 0 = bid, 1 = ask; price -> resting size
        self.levels: tuple[dict[int, int], dict[int, int]] = ({}, {})
        self._heaps: tuple[list[int], list[int]] = ([], [])
        self._rebuilt = 0

    # -- mutation ----------------------------------------------------------
    def add(self, ref: int, side: int, price: int, shares: int, ts: int = 0) -> int:
        """Insert a displayed order.  Returns the size resting ahead of it."""
        idx = 0 if side > 0 else 1
        levels = self.levels[idx]
        ahead = levels.get(price, 0)
        if ahead == 0:
            heapq.heappush(self._heaps[idx], -price if idx == 0 else price)
            self._maybe_rebuild(idx)
        levels[price] = ahead + shares
        self.orders[ref] = [side, price, shares, ahead, ts]
        return ahead

    def reduce(self, ref: int, shares: int):
        """Take ``shares`` off a resting order (an execution or a cancel).

        Returns ``(side, price, remaining)``, or ``None`` if the order is
        unknown -- which happens legitimately for orders that were already
        resting when the feed started, and for odd-lot or non-displayed
        interest the feed does not carry.
        """
        order = self.orders.get(ref)
        if order is None:
            return None
        side, price, resting, _ahead, _ts = order
        taken = shares if shares < resting else resting
        idx = 0 if side > 0 else 1
        levels = self.levels[idx]
        left = levels.get(price, 0) - taken
        if left > 0:
            levels[price] = left
        else:
            levels.pop(price, None)
        order[2] = resting - taken
        if order[2] <= 0:
            del self.orders[ref]
        return side, price, order[2]

    def delete(self, ref: int):
        """Remove a resting order entirely."""
        order = self.orders.get(ref)
        if order is None:
            return None
        return self.reduce(ref, order[2])

    def replace(self, old_ref: int, new_ref: int, shares: int, price: int, ts: int = 0):
        """Cancel-replace: priority is lost, so the new order goes to the back.

        Returns ``(side, queue_ahead)`` or ``None`` when the original order is
        unknown, in which case the side cannot be recovered and the new order
        cannot be placed either.
        """
        order = self.orders.get(old_ref)
        if order is None:
            return None
        side = order[0]
        self.delete(old_ref)
        ahead = self.add(new_ref, side, price, shares, ts)
        return side, ahead

    # -- queries -----------------------------------------------------------
    def best_bid(self) -> int:
        heap, levels = self._heaps[0], self.levels[0]
        while heap:
            price = -heap[0]
            if levels.get(price, 0) > 0:
                return price
            heapq.heappop(heap)
        return 0

    def best_ask(self) -> int:
        heap, levels = self._heaps[1], self.levels[1]
        while heap:
            price = heap[0]
            if levels.get(price, 0) > 0:
                return price
            heapq.heappop(heap)
        return 0

    def size_at(self, side: int, price: int) -> int:
        return self.levels[0 if side > 0 else 1].get(price, 0)

    def top(self) -> tuple[int, int, int, int]:
        """``(bid, bid_size, ask, ask_size)``; a missing side comes back as 0."""
        bid, ask = self.best_bid(), self.best_ask()
        return (
            bid,
            self.levels[0].get(bid, 0) if bid else 0,
            ask,
            self.levels[1].get(ask, 0) if ask else 0,
        )

    def depth(self, side: int, levels: int) -> list[tuple[int, int]]:
        """The first ``levels`` price levels on ``side``, best first."""
        book = self.levels[0 if side > 0 else 1]
        prices = sorted((p for p, s in book.items() if s > 0), reverse=side > 0)
        return [(p, book[p]) for p in prices[:levels]]

    # -- internals ---------------------------------------------------------
    def _maybe_rebuild(self, idx: int) -> None:
        heap, levels = self._heaps[idx], self.levels[idx]
        if len(heap) > 64 and len(heap) > 4 * (len(levels) + 1):
            live = [-p if idx == 0 else p for p, s in levels.items() if s > 0]
            heapq.heapify(live)
            self._heaps = (live, self._heaps[1]) if idx == 0 else (self._heaps[0], live)
            self._rebuilt += 1
