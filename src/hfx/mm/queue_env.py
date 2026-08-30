r"""A market-making environment where the book reacts, and adverse selection is free.

Chapter 05's environment is the one the closed form describes: fills arrive as a
Poisson process whose intensity depends only on how far the quote sits from the
mid, and the mid diffuses on its own.  It has no book in it.  A maker there
never gets picked off, because nothing connects the arrival of a fill to what
the price does next.

This one is chapter 04's book with an agent standing in it.

* Both best queues follow the **estimated** intensities and the **measured**
  order-size distribution -- the same objects the queue-reactive chapter fitted.
* The agent may rest one unit at each best.  **Its position in the queue is
  tracked**: consumption eats the size in front of it first, and it trades only
  once that reaches zero.  This is the quantity the closed form has no way to
  express, and the reason a maker cares which queue it joins.
* When a queue empties the price moves a tick, both queues are redrawn from the
  measured regeneration law, and any resting order dies with the level.

Adverse selection then needs no modelling at all.  Being filled on the bid means
the bid queue was being consumed; a queue being consumed is a queue on its way
to empty; and a bid queue that empties takes the price down.  The maker is
bought into a falling price by the same mechanism that filled it, which is what
"picked off" means and what the exponential-intensity model leaves out.

Everything is vectorised across a batch of independent books, so a learner sees
millions of transitions in seconds of numpy.
"""

from __future__ import annotations

import numpy as np

#: Actions on one side: rest at the touch, or stay out.
OUT, AT_TOUCH = 0, 1

#: Most events of one kind applied to one queue in a single step.
MAX_EVENTS = 6


class QueueBookEnv:
    """A batch of two-queue books with a market maker in each.

    Parameters
    ----------
    lam : (3, NQ) intensities per second by kind (limit, cancel, market) and
        queue size in average event sizes, from
        :func:`hfx.queue.reactive.intensities`.
    samplers : three :class:`hfx.queue.reactive.SizeSampler`, one per event
        kind -- the same objects chapter 04 simulates with.  Per kind and on a
        continuous scale, both of which matter: market orders are two thirds the
        size of limit orders, and rounding every order up to one average size
        coarsens the queue enough to change where it spends its time.
    regen : (NQ,) the measured distribution of the best queue immediately after
        a price change.
    tick : the price increment, in the same units as the reward.
    rebate : paid to the maker on every fill, in the same units as the tick.
        At zero, passive quoting at the touch of a one-tick book is a losing
        game -- the adverse move that follows a fill costs more than the half
        spread the fill earned -- which is the reason exchanges pay makers at
        all, and the quantity chapter 06 solves for.
    dt : the time step.  Small enough that at most one event per queue is
        likely; the environment reports the share of steps where more than one
        would have fired.
    inventory_limit : the maker stops quoting the side that would take it past
        this.
    q_max : the queue is reflected here rather than allowed to grow.  Chapter 04
        reports that the intensities above roughly ten average sizes are counts
        over a few seconds of residence a day, and they imply a drift of
        +47 average sizes per second at bucket 30 -- an artifact of a thin
        denominator, not a property of the market.  Left alone, every book
        climbs into it and freezes.  ``QueueBookEnv.from_tables`` picks the cap
        from the observed residence time instead of guessing it.
    """

    def __init__(self, lam, samplers, regen, tick: float = 0.01,
                 dt: float = 0.05, inventory_limit: int = 6, batch: int = 512,
                 q_max: float | None = None, rebate: float = 0.0, rng=None):
        self.lam = np.nan_to_num(np.asarray(lam, dtype=float))
        self.nq = self.lam.shape[1]
        self.q_max = float(self.nq - 1 if q_max is None else min(q_max, self.nq - 1))
        self.samplers = list(samplers)
        regen = np.asarray(regen, dtype=float)[: self.nq]
        self.regen = regen / regen.sum() if regen.sum() > 0 else np.ones(self.nq) / self.nq
        self.regen_cdf = np.cumsum(self.regen)
        self.tick = float(tick)
        self.dt = float(dt)
        # A per-fill rebate, in the same units as the tick.  Chapter 06 derives
        # what an exchange should pay; this is where that payment is felt.
        self.rebate = float(rebate)
        self.limit = int(inventory_limit)
        self.batch = int(batch)
        self.rng = np.random.default_rng() if rng is None else rng
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self):
        b = self.batch
        self.q_bid = self._draw_queue(b)
        self.q_ask = self._draw_queue(b)
        self.inventory = np.zeros(b, dtype=np.int64)
        self.ahead_bid = np.full(b, -1.0)      # -1 means "not resting"
        self.ahead_ask = np.full(b, -1.0)
        self.n_steps = 0
        self.n_multi = 0
        self.n_truncated = 0
        return self.observation()

    def _draw_queue(self, n):
        """A fresh best queue, at the centre of a bucket drawn from the data.

        Bucket centres, not bucket floors: a third of INTC's regenerations leave
        under one average order at the touch, and a queue that starts at 0.5 is
        a queue about to disappear.  Rounding those up to 1 would quietly remove
        the fastest price moves in the market.
        """
        bucket = np.searchsorted(self.regen_cdf, self.rng.random(n))
        return np.minimum(bucket.astype(float) + 0.5, self.q_max)

    def _draw_size(self, kind: int, n: int):
        return self.samplers[kind].draw(self.rng.random(n))

    def observation(self):
        """``(inventory, imbalance, own bid position, own ask position)``.

        Queue positions are reported as the fraction of the queue in front of
        the order, which is the scale-free version and the one that transfers
        between a six-dollar stock and an eighteen-hundred-dollar one.
        """
        total = self.q_bid + self.q_ask
        imbalance = np.where(total > 0, (self.q_bid - self.q_ask) / np.where(total > 0, total, 1), 0.0)
        pos_bid = np.where(self.ahead_bid < 0, -1.0,
                           self.ahead_bid / np.maximum(self.q_bid, 1e-9))
        pos_ask = np.where(self.ahead_ask < 0, -1.0,
                           self.ahead_ask / np.maximum(self.q_ask, 1e-9))
        return np.column_stack([self.inventory.astype(float), imbalance, pos_bid, pos_ask])

    # -- dynamics ---------------------------------------------------------
    def _rates(self, queue):
        idx = np.clip(queue.astype(int), 0, self.nq - 1)
        return self.lam[0, idx], self.lam[1, idx], self.lam[2, idx]

    @classmethod
    def from_tables(cls, events, time_ns, size_hist, regen, occupancy: float = 0.99,
                    **kwargs):
        """Build the environment straight from a replay's queue-reactive tables.

        The cap is the queue size below which the real book spends
        ``occupancy`` of its day.  Anything beyond it is a state the market
        visits for seconds, where the estimated intensity is a count over a
        thin denominator -- and where, on this data, it implies a drift of tens
        of average sizes per second.  Reflecting there keeps the environment
        inside the region its own parameters were measured in.
        """
        from ..book.replay import size_bucket_centres
        from ..queue.reactive import SizeSampler, intensities

        lam, _valid = intensities(events, time_ns, min_seconds=2.0)
        seconds = np.asarray(time_ns, dtype=float) / 1e9
        share = np.cumsum(seconds) / max(seconds.sum(), 1e-9)
        q_max = float(np.searchsorted(share, occupancy) + 1)
        samplers = [SizeSampler(np.asarray(size_hist)[k], size_bucket_centres())
                    for k in range(3)]
        return cls(lam, samplers, regen, q_max=q_max, **kwargs)

    def step(self, action_bid, action_ask):
        r"""Advance every book one step.

        ``action_bid`` and ``action_ask`` are ``OUT`` or ``AT_TOUCH``.  Returns
        ``(observation, reward, info)``; there is no terminal state -- the
        objective is the average reward per unit time.
        """
        b = self.batch
        rng = self.rng
        cash = np.zeros(b)
        # Counted here rather than inferred from the inventory afterwards: a
        # step in which both sides fill leaves the inventory unchanged and would
        # otherwise vanish from the count.
        fills = np.zeros(b, dtype=np.int64)

        # Post or pull.  Joining costs nothing but puts the order at the back of
        # whatever queue is there, which is the whole point.
        want_bid = (np.asarray(action_bid) == AT_TOUCH) & (self.inventory < self.limit)
        want_ask = (np.asarray(action_ask) == AT_TOUCH) & (self.inventory > -self.limit)
        joining_bid = want_bid & (self.ahead_bid < 0)
        joining_ask = want_ask & (self.ahead_ask < 0)
        self.ahead_bid = np.where(joining_bid, self.q_bid, np.where(want_bid, self.ahead_bid, -1.0))
        self.ahead_ask = np.where(joining_ask, self.q_ask, np.where(want_ask, self.ahead_ask, -1.0))

        for side in (0, 1):
            queue = self.q_bid if side == 0 else self.q_ask
            ahead = self.ahead_bid if side == 0 else self.ahead_ask
            limit_rate, cancel_rate, market_rate = self._rates(queue)

            # Poisson counts, not "did at least one arrive".  The estimated
            # intensities reach two hundred per second in the sparsely visited
            # states at the top of the grid, where a Bernoulli draw would throw
            # away a third of the events and slow the whole book down.  Counts
            # above MAX_EVENTS are truncated and reported.
            n_add = rng.poisson(limit_rate * self.dt)
            n_cancel = rng.poisson(cancel_rate * self.dt)
            n_take = rng.poisson(market_rate * self.dt)
            self.n_multi += int(np.sum(np.maximum(n_add + n_cancel + n_take - 1, 0)))
            self.n_truncated += int(np.sum(np.maximum(n_add - MAX_EVENTS, 0))
                                    + np.sum(np.maximum(n_cancel - MAX_EVENTS, 0))
                                    + np.sum(np.maximum(n_take - MAX_EVENTS, 0)))

            resting = ahead >= 0
            for k in range(MAX_EVENTS):
                adds = n_add > k
                if adds.any():
                    queue = queue + adds * self._draw_size(0, b)

                # Cancellations are uniform over the queue, so a share of them
                # sits in front of the order and a share behind: an order deep
                # in the queue advances as the book ahead of it is pulled.
                cancels = n_cancel > k
                if cancels.any():
                    cancel_size = cancels * self._draw_size(1, b)
                    share_ahead = np.where(resting & (queue > 0),
                                           ahead / np.maximum(queue, 1e-9), 0.0)
                    ahead = np.where(resting,
                                     np.maximum(ahead - cancel_size * share_ahead, 0.0), ahead)
                    queue = np.maximum(queue - cancel_size, 0.0)

                # Market orders eat strictly from the front, one order at a time,
                # which is what makes queue position worth anything.
                takes = n_take > k
                if takes.any():
                    take_size = takes * self._draw_size(2, b)
                    hit = resting & (take_size > ahead) & (take_size > 0)
                    filled = hit if k == 0 else (filled | hit)
                    ahead = np.where(resting, np.maximum(ahead - take_size, 0.0), ahead)
                    queue = np.maximum(queue - take_size, 0.0)
                elif k == 0:
                    filled = np.zeros(b, dtype=bool)
            if MAX_EVENTS == 0:
                filled = np.zeros(b, dtype=bool)

            if side == 0:
                self.q_bid = queue
                self.ahead_bid = np.where(filled, -1.0, ahead)
                self.inventory = self.inventory + filled
                fills += filled
                cash += filled * (self.tick / 2.0 + self.rebate)   # bought below the mid
            else:
                self.q_ask = queue
                self.ahead_ask = np.where(filled, -1.0, ahead)
                self.inventory = self.inventory - filled
                fills += filled
                cash += filled * (self.tick / 2.0 + self.rebate)   # sold above it

        # A queue that emptied moves the price, and the maker marks its
        # inventory at the new one.  This is where being filled costs money.
        down = self.q_bid <= 0
        up = self.q_ask <= 0
        move = (up.astype(float) - down.astype(float)) * self.tick
        cash += self.inventory * move
        redraw = down | up
        if redraw.any():
            n = int(redraw.sum())
            self.q_bid = np.where(redraw, self._draw_queue(b), self.q_bid)
            self.q_ask = np.where(redraw, self._draw_queue(b), self.q_ask)
            self.ahead_bid = np.where(redraw, -1.0, self.ahead_bid)
            self.ahead_ask = np.where(redraw, -1.0, self.ahead_ask)
            self.n_steps += 0 * n
        # Only the top is clipped.  The bottom must stay open: a queue holding
        # less than one average order sits in bucket 0, where market orders
        # arrive an order of magnitude faster than anywhere else, and clipping
        # it to 1 removes exactly the states in which prices move.
        self.q_bid = np.minimum(self.q_bid, self.q_max)
        self.q_ask = np.minimum(self.q_ask, self.q_max)
        self.n_steps += 1
        info = {"price_move": move, "moved": redraw, "fills": fills}
        return self.observation(), cash, info

    # -- diagnostics ------------------------------------------------------
    @property
    def multi_event_share(self) -> float:
        """Extra events per side-step beyond the first: how coarse the clock is."""
        return self.n_multi / max(2 * self.batch * self.n_steps, 1)

    @property
    def truncated_share(self) -> float:
        """Events lost to the per-step cap.  Must be negligible to be honest."""
        return self.n_truncated / max(2 * self.batch * self.n_steps, 1)
