"""Shared plotting style, so the figures in the report look like one document."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Large tick assets warm, small tick assets cool: the panel's one axis.
TICK_COLOURS = {"large": "#b5482f", "medium": "#c98a2e", "small": "#2f6fb5"}
SERIES = ["#2f6fb5", "#b5482f", "#3d8a52", "#7a5ba8", "#c98a2e", "#4b4b4b"]


def use_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.6,
            "font.size": 10,
        }
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=SERIES)


def tick_regime(spread_ticks: float) -> str:
    """Classify an asset by how many ticks wide its spread usually is."""
    if spread_ticks <= 1.2:
        return "large"
    if spread_ticks <= 4.0:
        return "medium"
    return "small"


def colour_for(spread_ticks: float) -> str:
    return TICK_COLOURS[tick_regime(spread_ticks)]


def finish(ax, title=None, xlabel=None, ylabel=None, legend=False):
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    return ax


def subplots(*args, **kwargs):
    use_style()
    return plt.subplots(*args, **kwargs)
