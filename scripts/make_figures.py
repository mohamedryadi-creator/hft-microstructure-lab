#!/usr/bin/env python
"""Figures for the README, from the committed measurements."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt  # noqa: E402

from hfx.pipeline.results import curve, curves, panel  # noqa: E402
from hfx.viz import colour_for, use_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "docs", "img")


def overview(P, C, day="2019-01-30"):
    use_style()
    fig, axes = plt.subplots(1, 5, figsize=(21, 3.8))

    g = P.groupby("symbol").agg(
        price=("price", "mean"), spread=("median_spread_ticks", "mean"),
        p1=("p_spread_one_tick", "mean"), vol=("vol_preavg_pct", "mean"),
        uz=("uz_vol_pct", "mean"), qr=("qr_vol_half_pct", "mean"),
    ).sort_values("price")

    def label(ax, x, y, sym):
        ax.annotate(sym, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax = axes[0]
    for sym, r in g.iterrows():
        ax.scatter(r.price, r.p1, color=colour_for(r.spread), s=42)
        label(ax, r.price, r.p1, sym)
    ax.set_xscale("log")
    ax.set_xlabel("price ($)")
    ax.set_ylabel("time with a one-tick spread")
    ax.set_title("The tick axis")

    ax = axes[1]
    for sym in ["SIRI", "INTC", "AAPL", "AMZN"]:
        try:
            steps = curve(C, sym, day, "signature_steps")
            rv = curve(C, sym, day, "signature_rv")
        except KeyError:
            continue
        iv = float(P[(P.symbol == sym) & (P.date == day)].preavg.iloc[0])
        ax.semilogx(steps / 10.0, rv / iv, "o-", ms=3, label=sym)
    ax.axhline(1.0, color="0.6", lw=1, ls="--")
    ax.set_xlabel("sampling step (s)")
    ax.set_ylabel("realized variance / pre-averaged IV")
    ax.set_title("Signature plots")
    ax.legend(fontsize=7)

    ax = axes[2]
    for sym, r in g.iterrows():
        ax.scatter(r.vol, r.uz, color=colour_for(r.spread), s=42)
        label(ax, r.vol, r.uz, sym)
    lim = [0.8, 2.4]
    ax.plot(lim, lim, color="0.6", lw=1, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("pre-averaged, from quotes (%/day)")
    ax.set_ylabel("uncertainty zones, from trades (%/day)")
    ax.set_title("Two estimators, one number")

    ax = axes[3]
    for sym, r in g.iterrows():
        ax.scatter(r.vol, r.qr, color=colour_for(r.spread), s=42)
        label(ax, r.vol, r.qr, sym)
    lim = [0.02, 4]
    ax.plot(lim, lim, color="0.6", lw=1, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("realized volatility (%/day)")
    ax.set_ylabel("queue-reactive volatility (%/day)")
    ax.set_title("Volatility out of queues alone")

    ax = axes[4]
    try:
        agents = __import__("pandas").read_csv(os.path.join(ROOT, "results", "agents.csv"))
        agents = agents.set_index("symbol")
        for sym, r in g.iterrows():
            if sym not in agents.index:
                continue
            blind = agents.loc[sym, "blind_reward"] * 1e4
            sighted = agents.loc[sym, "sighted_reward"] * 1e4
            ax.plot([0, 1], [blind, sighted], color=colour_for(r.spread), lw=1.2,
                    marker="o", ms=4)
            label(ax, 1, sighted, sym)
        ax.axhline(0.0, color="0.4", lw=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["cannot see\nthe book", "sees\nthe book"])
        ax.set_xlim(-0.25, 1.45)
        ax.set_ylabel(r"market maker's reward, $10^{-4}$/s")
        ax.set_title("What the book is worth to a maker")
    except FileNotFoundError:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(IMG, exist_ok=True)
    path = os.path.join(IMG, "overview.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    print("wrote", os.path.relpath(overview(panel(), curves()), ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
