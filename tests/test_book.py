"""The book against a deliberately slow reference.

The fast book keeps its best price in a heap with lazy deletion, which is
exactly the kind of structure that works until the day it silently does not.
The reference here recomputes the best price by scanning every level, is
obviously correct, and is far too slow to use on 400 million messages -- so it
is used on a hundred thousand random ones instead.
"""

import random

from hfx.book.book import OrderBook


class ReferenceBook:
    """Correct by inspection: no caching, no heap, scan for the best price."""

    def __init__(self):
        self.orders = {}

    def add(self, ref, side, price, shares):
        ahead = sum(
            s for (sd, p, s) in self.orders.values() if sd == side and p == price
        )
        self.orders[ref] = (side, price, shares)
        return ahead

    def reduce(self, ref, shares):
        if ref not in self.orders:
            return None
        side, price, resting = self.orders[ref]
        left = max(resting - shares, 0)
        if left:
            self.orders[ref] = (side, price, left)
        else:
            del self.orders[ref]
        return side, price, left

    def delete(self, ref):
        if ref not in self.orders:
            return None
        return self.reduce(ref, self.orders[ref][2])

    def replace(self, old_ref, new_ref, shares, price):
        if old_ref not in self.orders:
            return None
        side = self.orders[old_ref][0]
        self.delete(old_ref)
        return side, self.add(new_ref, side, price, shares)

    def top(self):
        bids = [(p, s) for (sd, p, s) in self.orders.values() if sd > 0]
        asks = [(p, s) for (sd, p, s) in self.orders.values() if sd < 0]
        bid = max((p for p, _ in bids), default=0)
        ask = min((p for p, _ in asks), default=0)
        return (
            bid,
            sum(s for p, s in bids if p == bid),
            ask,
            sum(s for p, s in asks if p == ask),
        )


def test_add_reduce_delete_and_queue_position():
    book = OrderBook()
    assert book.add(1, +1, 1_000_000, 300) == 0
    assert book.add(2, +1, 1_000_000, 200) == 300      # rests behind the first
    assert book.add(3, +1, 999_900, 500) == 0          # a different price level
    assert book.add(4, -1, 1_000_100, 400) == 0
    assert book.top() == (1_000_000, 500, 1_000_100, 400)

    assert book.reduce(1, 100) == (+1, 1_000_000, 200)
    assert book.top() == (1_000_000, 400, 1_000_100, 400)

    book.delete(2)
    book.delete(1)
    # The whole best level is gone, so the book falls back one tick.
    assert book.top() == (999_900, 500, 1_000_100, 400)

    assert book.reduce(999, 10) is None                # unknown reference


def test_replace_loses_priority():
    book = OrderBook()
    book.add(1, +1, 1_000_000, 300)
    book.add(2, +1, 1_000_000, 200)
    side, ahead = book.replace(1, 3, 300, 1_000_000)
    assert side == +1
    # Order 1 gave up its place at the front; the replacement sits behind 2.
    assert ahead == 200
    assert book.top() == (1_000_000, 500, 0, 0)
    assert book.orders[3][3] == 200


def test_depth_is_sorted_outwards():
    book = OrderBook()
    for i, price in enumerate([999_800, 999_900, 1_000_000]):
        book.add(i, +1, price, 100 * (i + 1))
    for i, price in enumerate([1_000_100, 1_000_200]):
        book.add(10 + i, -1, price, 50)
    assert book.depth(+1, 3) == [(1_000_000, 300), (999_900, 200), (999_800, 100)]
    assert book.depth(-1, 2) == [(1_000_100, 50), (1_000_200, 50)]


def test_matches_the_reference_on_random_message_sequences():
    rng = random.Random(20240130)
    fast, slow = OrderBook(), ReferenceBook()
    live, next_ref = [], 1
    for step in range(60_000):
        action = rng.random()
        if action < 0.45 or not live:
            side = 1 if rng.random() < 0.5 else -1
            price = 1_000_000 + 100 * rng.randint(-5, 5)
            shares = 100 * rng.randint(1, 10)
            ref, next_ref = next_ref, next_ref + 1
            assert fast.add(ref, side, price, shares) == slow.add(ref, side, price, shares)
            live.append(ref)
        elif action < 0.70:
            ref = rng.choice(live)
            shares = 100 * rng.randint(1, 10)
            assert fast.reduce(ref, shares) == slow.reduce(ref, shares)
            if ref not in slow.orders:
                live.remove(ref)
        elif action < 0.90:
            ref = rng.choice(live)
            assert fast.delete(ref) == slow.delete(ref)
            live.remove(ref)
        else:
            ref = rng.choice(live)
            new_ref, next_ref = next_ref, next_ref + 1
            price = 1_000_000 + 100 * rng.randint(-5, 5)
            shares = 100 * rng.randint(1, 10)
            assert fast.replace(ref, new_ref, shares, price) == slow.replace(
                ref, new_ref, shares, price
            )
            live.remove(ref)
            live.append(new_ref)
        if step % 17 == 0:
            assert fast.top() == slow.top()
    assert fast.top() == slow.top()
    # The heap must not have grown without bound over 60 000 messages.
    assert len(fast._heaps[0]) + len(fast._heaps[1]) < 2_000
