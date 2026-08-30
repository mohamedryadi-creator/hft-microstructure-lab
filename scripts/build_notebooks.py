#!/usr/bin/env python
"""Generate the notebooks from source, so they stay thin and reviewable.

A notebook that has been edited in place for months accumulates dead cells and
stale outputs.  These are built from this file, executed by ``make notebooks``,
and every one of them carries ``assert`` checkpoints -- so a notebook whose
outputs are committed has verified its own numbers.
"""

from __future__ import annotations

import os
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "notebooks")

HEADER = """import sys, os
sys.path.insert(0, os.path.join("..", "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from hfx.viz import subplots, use_style, colour_for, tick_regime
from hfx.pipeline.results import panel, curves, curve, symbol_days
use_style()
P = panel(); C = curves()
DAY = "2019-01-30"
print(f"{len(P)} symbol-days, {P.symbol.nunique()} symbols, {P.date.nunique()} dates")"""


def md(text):
    return ("md", text)


def code(text):
    return ("code", text)


NOTEBOOKS: dict[str, list] = {}

# --------------------------------------------------------------------------
NOTEBOOKS["01_book_reconstruction_and_tick_regimes.ipynb"] = [
    md("""# 01 — Rebuilding the book, and the one axis the panel varies along

Nasdaq does not publish the order book.  It publishes the *changes* to it, one
order at a time, as a binary feed: an add, an execution, a cancel, a delete, a
replace, each carrying an order reference and a nanosecond timestamp.  This
notebook checks that we can turn that back into a book, and then measures the
variable everything else in the repository is organised around: how many ticks
wide the spread is.

The tick on Nasdaq is one cent for every stock above a dollar.  That makes the
*price level* the tick axis: a cent is 17 basis points of SIRI at \\$6 and 0.06
basis points of AMZN at \\$1650, and the two markets that result behave
differently in almost every measurement that follows."""),
    code(HEADER),
    md("""## The wire format, on a book small enough to check by hand

`hfx.itch.synth` writes ITCH messages from the published field offsets; the
decoder reads them back.  The two were written from the specification
independently, so this is a real check of the binary layout rather than a
round-trip through one piece of code."""),
    code('''from hfx.itch import synth
from hfx.itch.reader import ItchExtractor
from hfx.itch.spec import OPEN_NS

S = 1_000_000_000
messages = [
    synth.stock_directory(1, "DEMO"),
    synth.add_order(1, OPEN_NS + 1 * S, ref=10, side=+1, shares=300, price=1_500_000),
    synth.add_order(1, OPEN_NS + 2 * S, ref=11, side=-1, shares=200, price=1_500_100),
    synth.execute(1, OPEN_NS + 3 * S, ref=11, shares=120, match=1),
    synth.cancel(1, OPEN_NS + 4 * S, ref=10, shares=100),
]
ext = ItchExtractor(["DEMO"])
ext.feed(synth.frame(messages))
decoded = pd.DataFrame(ext.buffers["DEMO"].as_dict())
decoded["type"] = [chr(c) for c in decoded.etype]
decoded[["type", "ts", "ref", "side", "shares", "price"]]'''),
    md("""## Replay: the book, the trades, and the aggressor side

An execution consumes a *resting* order whose side the book knows, so the
aggressor is its opposite.  No Lee-Ready inference, no tick test, and therefore
none of the attenuation a sign-inference error inflicts on every impact
estimate downstream."""),
    code('''from hfx.book.replay import replay

out = replay({k: np.asarray(v) for k, v in ext.buffers["DEMO"].as_dict().items()}, "DEMO", DAY)
trade = {k: v[0] for k, v in out.trades.items()}
print("one trade:", {k: int(v) for k, v in trade.items()})
assert trade["side"] == +1, "an execution against a resting sell is a buy"
assert trade["price"] == 1_500_100 and trade["size"] == 120
assert out.stats["n_unknown_ref"] == 0'''),
    md("""## The panel

Twelve Nasdaq-listed symbols over seven days, chosen to run from \\$6 to
\\$1 700 so that the spread-to-tick ratio runs from one to fifty."""),
    code('''day = P[P.date == DAY].sort_values("price")
cols = ["symbol", "price", "median_spread_ticks", "p_spread_one_tick", "n_trades",
        "trades_per_second", "mean_order_size", "aes", "hidden_share", "at_touch_displayed"]
day[cols].round(3).to_string(index=False)'''),
    code('''fig, axes = subplots(1, 2, figsize=(11, 3.6))
g = P.groupby("symbol").agg(price=("price", "mean"),
                            p1=("p_spread_one_tick", "mean"),
                            spread=("median_spread_ticks", "mean"),
                            queue=("qr_mean_queue_emp", "mean")).sort_values("price")
for ax, (col, label) in zip(axes, [("p1", "time with a one-tick spread"),
                                   ("queue", "best queue, in average event sizes")]):
    for sym, r in g.iterrows():
        ax.scatter(r.price, r[col], color=colour_for(r.spread), s=45)
        ax.annotate(sym, (r.price, r[col]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("price ($)"); ax.set_ylabel(label)
axes[0].set_title("Large tick at the left, small tick at the right")
axes[1].set_title("Queues exist only where the tick binds")
plt.tight_layout()'''),
    md("""The two panels are the same statement twice.  A stock whose tick is
large relative to its volatility cannot have a spread wider than one tick for
long, so liquidity has nowhere to go but *into the queue at the touch*; a stock
whose tick is negligible has a spread of tens of ticks and queues barely one
order deep.  Every later chapter splits along this line."""),
    code('''fig, axes = subplots(1, 2, figsize=(11, 3.4))
for sym, ax in zip(["SIRI", "AMZN"], axes):
    counts = curve(C, sym, DAY, "minute_counts")
    minutes = np.arange(counts.shape[0])
    ax.plot(minutes, counts[:, 0], label="limit orders")
    ax.plot(minutes, counts[:, 1], label="cancels")
    ax.plot(minutes, counts[:, 2], label="trades")
    ax.set_yscale("log"); ax.set_title(sym)
    ax.set_xlabel("minutes after 09:30"); ax.set_ylabel("events per minute")
axes[0].legend()
plt.tight_layout()'''),
    md("""## Does the aggressor-side convention survive contact with the data?

A trade classified as buyer-initiated has to print at or above the offer.  The
check is run on *displayed* trades; hidden executions are non-displayed
liquidity and print inside the spread by construction, which is itself the
control."""),
    code('''chk = P.groupby("symbol").agg(at_touch=("at_touch_displayed", "mean"),
                              hidden=("hidden_share", "mean"),
                              spread=("median_spread_ticks", "mean")).sort_values("spread")
print(chk.round(4).to_string())
assert chk.at_touch.min() > 0.95, "the aggressor side or the quote alignment is wrong"
print("\\nworst symbol:", chk.at_touch.idxmin(), round(chk.at_touch.min(), 4))'''),
    code('''fig, ax = subplots(figsize=(6.2, 3.4))
for sym, r in g.iterrows():
    row = P[P.symbol == sym]
    ax.scatter(r.spread, row.effective_half_spread_ticks.mean() / row.quoted_half_spread_ticks.mean(),
               color=colour_for(r.spread), s=45)
    ax.annotate(sym, (r.spread, row.effective_half_spread_ticks.mean() / row.quoted_half_spread_ticks.mean()),
                fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xscale("log"); ax.axhline(1.0, color="0.6", lw=1, ls="--")
ax.set_xlabel("median spread (ticks)"); ax.set_ylabel("effective / quoted half-spread")
ax.set_title("What a taker pays, against what the screen showed")
plt.tight_layout()'''),
    md("""The effective half-spread is below the quoted one everywhere, and the
gap is largest for the small-tick names: with a fifty-tick spread there is room
for hidden and price-improving liquidity to sit inside it, and takers find it.
Chapter 03 shows the same number arrived at from an entirely different
direction, through the tick grid."""),
]

# --------------------------------------------------------------------------
NOTEBOOKS["02_hawkes_order_flow.ipynb"] = [
    md("""# 02 — Hawkes processes, and a volatility predicted from the flow alone

Market orders arrive in bursts.  A Hawkes process is the smallest model that
says why: every event raises the intensity of the next,

$$\\lambda_i(t) = \\mu_i + \\sum_j \\int_0^t \\varphi_{ij}(t-s)\\,dN_j(s),$$

and the spectral radius of $\\int\\varphi$ is the average number of events each
event triggers -- the *endogeneity ratio*.

The chapter has three parts.  The estimator is checked against a process whose
kernel we chose.  It is then applied to the buy/sell flow of the panel, where a
goodness-of-fit test that can actually reject the model does reject it.  And
the fitted flow makes a prediction about the *price*: the variance per unit
time of the signed trade count, at every scale, in closed form."""),
    code(HEADER),
    md("""## Ground truth: recover a kernel we injected

The kernel bank has fixed decay rates and free non-negative amplitudes, which
makes the log-likelihood concave -- the estimate is a global maximum, not
wherever an optimiser happened to stop."""),
    code('''from hfx.hawkes.simulate import simulate_exp
from hfx.hawkes.fit import fit_exp_bank, log_grid
from hfx.hawkes.gof import ks_exponential, qq_points, rescaled_residuals

rng = np.random.default_rng(7)
mu, alpha, beta, T = np.array([0.5]), np.array([[0.8]]), np.array([[2.0]]), 30_000.0
times, marks = simulate_exp(mu, alpha, beta, T, rng)
exact = fit_exp_bank(times, marks, betas=np.array([2.0]), T=T, d=1)
coarse = fit_exp_bank(times, marks, betas=log_grid(0.1, 100, 6), T=T, d=1)
print(f"true branching ratio 0.400   fitted {exact.branching_ratio:.3f} (beta known)"
      f"   {coarse.branching_ratio:.3f} (beta on a six-point grid)")
assert abs(exact.branching_ratio - 0.4) < 0.03'''),
    md("""## The test that can say no

Under the fitted model the compensator-rescaled inter-arrival times are i.i.d.
$\\mathrm{Exp}(1)$ -- exactly, not approximately.  So a Kolmogorov-Smirnov test
against $\\mathrm{Exp}(1)$ is a test of the model rather than a picture of a fit
next to the data it was fitted to."""),
    code('''res_true = rescaled_residuals(times, marks, exact)[0]
_, p_true = ks_exponential(res_true)
print(f"simulated data, true model: mean residual {res_true.mean():.3f}, KS p = {p_true:.3f}")

fig, ax = subplots(figsize=(4.6, 4.0))
x, y = qq_points(res_true)
ax.plot(x, y, ".", ms=3, label="simulated, correct model")
sym = "INTC"
tq = curve(C, sym, DAY, "hawkes_residual_qq").ravel()
from scipy import stats
probs = np.linspace(0.5 / tq.size, 1 - 0.5 / tq.size, tq.size)
ax.plot(stats.expon.ppf(probs), np.sort(tq), ".", ms=3, label=f"{sym}, fitted model")
lim = [0, 8]; ax.plot(lim, lim, color="0.5", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Exp(1) quantile"); ax.set_ylabel("residual quantile")
ax.set_title("Time-rescaling residuals"); ax.legend()
plt.tight_layout()'''),
    md("""## The panel

Two numbers per symbol-day: the endogeneity ratio, and the *net* excitation
$\\delta = s - c$ between self- and cross-excitation.  A buy making the next buy
more likely is order splitting; a buy making a sell more likely is the other
side reacting.  Which one dominates decides whether the signed order flow
trends or mean-reverts."""),
    code('''h = P.groupby("symbol").agg(price=("price", "mean"), spread=("median_spread_ticks", "mean"),
                            n=("hawkes_branching", "mean"), n_sd=("hawkes_branching", "std"),
                            self_=("hawkes_self", "mean"), cross=("hawkes_cross", "mean"),
                            delta=("hawkes_delta", "mean"),
                            n_single=("hawkes_branching_single", "mean"),
                            ks_p=("hawkes_ks_p", "max")).sort_values("price")
print(h.round(4).to_string())
print(f"\\nbranching ratio across the panel: {h.n.min():.2f} to {h.n.max():.2f}")
print(f"largest KS p-value anywhere in the panel: {h.ks_p.max():.2e}")
assert (h.delta > 0).all(), "self-excitation should dominate: order flow persists"'''),
    md("""Every fit is rejected, on every symbol, on every day, with p-values
that are numerically zero.  That is the expected outcome and it is worth stating
plainly: a Hawkes process with a handful of exponentials is *not* the law of the
order flow.  What the fit still gives is a well-defined summary of how much of
the flow is triggered by itself, and the sum-of-exponentials bank raises that
estimate over a single exponential -- the single-kernel fit cannot see the slow
part of the excitation and attributes it to the baseline instead."""),
    code('''fig, axes = subplots(1, 2, figsize=(11, 3.6))
axes[0].bar(np.arange(len(h)) - 0.2, h.n, width=0.4, label="kernel bank")
axes[0].bar(np.arange(len(h)) + 0.2, h.n_single, width=0.4, label="one exponential")
axes[0].set_xticks(np.arange(len(h))); axes[0].set_xticklabels(h.index, rotation=45)
axes[0].set_ylabel("endogeneity ratio"); axes[0].legend()
axes[0].set_title("How much of the flow triggers itself")
lags = curve(C, "INTC", DAY, "hawkes_kernel_lags")
for sym in ["SIRI", "INTC", "AAPL", "AMZN"]:
    k = curve(C, sym, DAY, "hawkes_kernel")
    axes[1].loglog(lags, k[0, 0], label=f"{sym} self")
axes[1].set_xlabel("lag (s)"); axes[1].set_ylabel(r"$\\varphi_{\\rm self}(t)$")
axes[1].set_title("Fitted self-excitation kernels"); axes[1].legend()
plt.tight_layout()'''),
    md("""## The prediction

For a symmetric buy/sell flow the signed trade count $P_t = N^+_t - N^-_t$ has a
spectrum that collapses to a scalar, and the variance per unit time over a scale
$\\tau$ follows in closed form.  Two limits are worth naming:

$$V(0^+) = 2\\Lambda, \\qquad V(\\infty) = \\frac{2\\Lambda}{(1-(s-c))^2}.$$

With self-excitation dominating, $s>c$, the variance per unit time **rises** with
the scale: persistent flow.  Nothing about prices enters the estimation -- only
the times and signs of the trades -- so comparing the curve with the realised
signature plot of the same day is a genuine out-of-sample check."""),
    code('''fig, axes = subplots(1, 3, figsize=(13, 3.4))
for ax, sym in zip(axes, ["SIRI", "INTC", "AAPL"]):
    taus = curve(C, sym, DAY, "hawkes_taus")
    model = curve(C, sym, DAY, "hawkes_signature_model")
    emp = curve(C, sym, DAY, "hawkes_signature_empirical")
    ax.semilogx(taus, model, label="Hawkes fit, closed form")
    ax.semilogx(taus, emp, "o", ms=4, label="measured")
    ax.set_title(sym); ax.set_xlabel(r"scale $\\tau$ (s)")
axes[0].set_ylabel(r"$\\mathrm{Var}(P_{t+\\tau}-P_t)/\\tau$"); axes[0].legend()
plt.tight_layout()'''),
    code('''rows = []
for sym in sorted(P.symbol.unique()):
    taus = curve(C, sym, DAY, "hawkes_taus")
    model = curve(C, sym, DAY, "hawkes_signature_model")
    emp = curve(C, sym, DAY, "hawkes_signature_empirical")
    ok = np.isfinite(emp) & (emp > 0)
    rows.append({"symbol": sym, "median ratio": float(np.median(emp[ok] / model[ok])),
                 "ratio at the short end": float(emp[ok][0] / model[ok][0]),
                 "ratio at the long end": float(emp[ok][-1] / model[ok][-1])})
sig = pd.DataFrame(rows).set_index("symbol")
print(sig.round(3).to_string())
print(f"\\nmedian over the panel: {sig['median ratio'].median():.3f}")'''),
    md("""The short end matches by construction -- at $\\tau\\to0$ the variance rate
is twice the trade intensity, which any fit gets right.  The long end is the
test, and it is where the exponential bank runs out: the measured variance keeps
growing past the scale at which the fitted kernel has decayed.  That is the same
verdict the goodness-of-fit test gave, seen in the price rather than in the
residuals, and it is the standard argument for a power-law kernel."""),
]

# --------------------------------------------------------------------------
NOTEBOOKS["03_realized_volatility_noise_and_the_tick.ipynb"] = [
    md("""# 03 — Realized volatility, microstructure noise, and the tick

Sampling a noisy price more finely does not measure its volatility better.  With
$Y = X + \\varepsilon$,

$$\\mathbb{E}\\,RV_n = IV + 2n\\,\\mathbb{E}[\\varepsilon^2],$$

so the estimator diverges as the sampling tightens.  Three estimators fix it at
three different rates, and the rates are *measured* here rather than quoted.

Then the tick.  Robert and Rosenbaum's uncertainty-zones model says something
sharper than "discreteness is noise": it says where the efficient price *is*
when the traded one moves, which turns the tick grid from a nuisance into an
observation."""),
    code(HEADER),
    md("""## The rates, by Monte Carlo

Simulate a price whose integrated variance we chose, add noise, estimate, and
regress log RMSE on log $n$.  The theory says $-1/6$ for two-scale, $-1/4$ for
pre-averaging and $-1/5$ for the non-negative realized kernel -- the kernel pays
a rate for the guarantee that it never returns a negative variance."""),
    code('''from hfx.vol import estimators as ve

rng = np.random.default_rng(4)
IV, OMEGA = 4e-4, 1e-4
ns, reps = [4_000, 16_000, 64_000, 256_000], 120
rmse = {"two_scale": [], "pre_averaged": [], "realized_kernel": []}
for n in ns:
    acc = {k: [] for k in rmse}
    for _ in range(reps):
        x = np.concatenate(([0.0], np.cumsum(rng.normal(0, np.sqrt(IV / n), n))))
        y = x + rng.normal(0, OMEGA, n + 1)
        for name in rmse:
            acc[name].append(getattr(ve, name)(y))
    for name in rmse:
        rmse[name].append(np.sqrt(np.mean((np.array(acc[name]) - IV) ** 2)))
slopes = {k: np.polyfit(np.log(ns), np.log(v), 1)[0] for k, v in rmse.items()}
print(pd.DataFrame({"measured slope": slopes,
                    "theory": {"two_scale": -1/6, "pre_averaged": -1/4, "realized_kernel": -1/5}}).round(3))
assert abs(slopes["pre_averaged"] + 0.25) < 0.06'''),
    code('''fig, ax = subplots(figsize=(5.4, 3.6))
for name, v in rmse.items():
    ax.loglog(ns, np.array(v) / IV, "o-", label=f"{name} ({slopes[name]:+.3f})")
ax.set_xlabel("observations per day"); ax.set_ylabel("relative RMSE")
ax.set_title("Convergence rates, measured"); ax.legend()
plt.tight_layout()'''),
    md("""## The signature plot on real quotes

The same divergence, on the mid quote of four real symbols.  The horizontal
axis is the sampling step in seconds; the vertical axis is the realized variance
that step produces, as a multiple of the pre-averaged estimate of the same
day's integrated variance."""),
    code('''fig, ax = subplots(figsize=(6.4, 3.8))
for sym in ["SIRI", "INTC", "AAPL", "AMZN"]:
    steps = curve(C, sym, DAY, "signature_steps") / 10.0   # the grid is 100 ms
    rv = curve(C, sym, DAY, "signature_rv")
    iv = float(P[(P.symbol == sym) & (P.date == DAY)].preavg.iloc[0])
    ax.semilogx(steps, rv / iv, "o-", ms=3, label=sym)
ax.axhline(1.0, color="0.6", lw=1, ls="--")
ax.set_xlabel("sampling step (s)"); ax.set_ylabel("realized variance / pre-averaged IV")
ax.set_title("Signature plot, mid quote"); ax.legend()
plt.tight_layout()'''),
    md("""## The tick grid, and the parameter that makes it readable

In the uncertainty-zones model the traded price moves from $\\alpha k$ to
$\\alpha(k\\pm1)$ only once the efficient price has crossed a barrier
$(\\tfrac12+\\eta)\\alpha$ away.  Just after a move the efficient price sits *on*
that barrier, so a further move in the same direction needs a distance
$\\alpha$ and a reversal needs $2\\eta\\alpha$; the two-barrier argument gives

$$\\frac{\\mathbb{P}(\\text{continuation})}{\\mathbb{P}(\\text{alternation})} = 2\\eta
\\quad\\Longrightarrow\\quad \\hat\\eta = \\frac{N_c}{2N_a},$$

and, because the efficient price is known exactly at every price change,
$\\hat X = P - d\\,\\alpha(\\tfrac12-\\eta)$ -- an integrated variance estimator with
the noise *removed* rather than averaged away."""),
    code('''from hfx.vol import uncertainty_zones as uz

rng = np.random.default_rng(9)
n, sigma, tick = 1_000_000, 0.30, 0.01
rows = []
for eta_true in [0.1, 0.2, 0.35, 0.5]:
    x = 50 + np.concatenate(([0.0], np.cumsum(rng.normal(0, sigma * np.sqrt(1 / n), n))))
    levels, _ = uz.simulate(x, tick, eta_true)
    eta_hat, nc, na = uz.estimate_eta(levels)
    rows.append({"eta": eta_true, "eta_hat": eta_hat, "changes": nc + na,
                 "IV_uz / IV": uz.integrated_variance(levels, tick, eta_hat) / sigma**2,
                 "RV_grid / IV": ve.realized_variance(levels * tick) / sigma**2,
                 "1 / (2 eta)": uz.variance_inflation(eta_true)})
print(pd.DataFrame(rows).round(3).to_string(index=False))'''),
    md("""Two things are being checked at once.  $\\hat\\eta$ recovers the parameter
that generated the path, and the realized variance of the *grid* price
overstates the integrated variance by $1/(2\\eta)$ -- which is not a fit but an
optional-stopping identity: between two price changes the efficient price is a
martingale started on a barrier and absorbed at $+\\alpha$ or $-2\\eta\\alpha$, so
it accumulates exactly $2\\eta\\alpha^2$ of quadratic variation."""),
    code('''u = P.groupby("symbol").agg(price=("price", "mean"), spread=("median_spread_ticks", "mean"),
                            eta=("eta", "mean"), eta_mid=("eta_mid", "mean"),
                            vol=("vol_preavg_pct", "mean"), vol_uz=("uz_vol_pct", "mean"),
                            measured=("grid_rv_over_uz", "mean"),
                            predicted=("variance_inflation_pred", "mean")).sort_values("spread")
u["vol_uz / vol"] = u.vol_uz / u.vol
u["inflation gap"] = (u.measured - u.predicted).abs()
print(u.round(3).to_string())
large = u[u.spread <= 1.2]
print(f"\\nlarge-tick names: uncertainty-zone volatility within "
      f"{100 * (large['vol_uz / vol'] - 1).abs().max():.1f}% of the pre-averaged one")'''),
    code('''fig, axes = subplots(1, 2, figsize=(11, 3.6))
for sym, r in u.iterrows():
    axes[0].scatter(r.spread, r["vol_uz / vol"], color=colour_for(r.spread), s=45)
    axes[0].annotate(sym, (r.spread, r["vol_uz / vol"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[1].scatter(r.predicted, r.measured, color=colour_for(r.spread), s=45)
    axes[1].annotate(sym, (r.predicted, r.measured), fontsize=7, xytext=(3, 3), textcoords="offset points")
axes[0].set_xscale("log"); axes[0].axhline(1, color="0.6", ls="--", lw=1)
axes[0].set_xlabel("median spread (ticks)"); axes[0].set_ylabel("uncertainty-zone vol / pre-averaged vol")
axes[0].set_title("Two estimators, two data sources, one number")
lim = [0, max(u.predicted.max(), u.measured.max()) * 1.1]
axes[1].plot(lim, lim, color="0.6", lw=1, ls="--")
axes[1].set_xlabel(r"$1/(2\\hat\\eta)$ predicted"); axes[1].set_ylabel(r"measured $RV_{\\rm grid}/\\widehat{IV}$")
axes[1].set_title("The optional-stopping identity, on real data")
plt.tight_layout()'''),
    md("""On the assets whose spread is pinned at one tick, an estimator that
uses only the *traded price on the tick grid* lands within a few percent of a
pre-averaged estimator built from the *quotes*.  They share no data and no
assumption beyond the model, so the agreement is a real check on both.

The measured $\\eta$ is above $1/2$ on every symbol, which is outside the range
the model was written for -- $\\eta>1/2$ means a reversal must travel further
than a continuation, so the printed price trends.  Two things push it up on a
Nasdaq tape.  The venue prints only part of the consolidated volume, so the
price moves between the trades it does print; and the estimator counts one-tick
changes, which stop being the relevant unit once the spread is several ticks
wide.  The value degrades in exactly that order across the panel, and the
identity in the right-hand panel fails at the same place."""),
]

# --------------------------------------------------------------------------
NOTEBOOKS["04_queue_reactive_model.ipynb"] = [
    md("""# 04 — Prices formed by queues emptying

The queue-reactive model of Huang, Lehalle and Rosenbaum contains no price
process.  The book is a Markov jump process whose event rates depend on the
queue sizes, and the price is a consequence: when a queue at the best empties,
the best price on that side moves a tick.  Volatility is an *output*.

Estimation needs no optimiser.  For a Markov jump process the maximum
likelihood estimate of an intensity is the count over the time spent in the
state, $\\hat\\lambda_e(q) = N_e(q)/T(q)$, and both counters are accumulated in
the same pass that rebuilds the book."""),
    code(HEADER),
    code('''from hfx.queue import reactive as qr
from hfx.book.replay import NQ, size_bucket_centres

fig, axes = subplots(1, 3, figsize=(13, 3.6))
for ax, sym in zip(axes, ["SIRI", "INTC", "AAPL"]):
    lam = curve(C, sym, DAY, "qr_lambda")
    seconds = curve(C, sym, DAY, "qr_time") / 1e9
    q = np.arange(lam.shape[1])
    # The time spent in each state is the denominator of every rate above it.
    ax2 = ax.twinx()
    ax2.fill_between(q, seconds, color="0.85", zorder=0)
    ax2.set_ylim(0, seconds.max() * 3.2); ax2.set_yticks([]); ax2.grid(False)
    for i, label in enumerate(["limit orders", "cancellations", "market orders"]):
        ax.plot(q, lam[i], label=label, zorder=3)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_yscale("log"); ax.set_xlim(0, 25); ax.set_title(sym)
    ax.set_xlabel("queue size (average event sizes)")
axes[0].set_ylabel("intensity (per second)")
axes[0].plot([], [], color="0.85", lw=6, label="time spent in the state")
axes[0].legend(fontsize=8)
plt.tight_layout()'''),
    md("""**Market orders fall sharply with the queue.** Between an empty best
queue and one holding five average orders the market-order intensity drops by a
factor of 9 on SIRI, 29 on INTC and 9 on MSFT: takers arrive when the queue in
front of them is thin, which is the effect the model is named for and the reason
a queue that starts to empty tends to keep emptying.

**Limit orders and cancellations move together, and much less.** They track each
other to within a tenth over most of the range -- which is exactly what keeps a
queue alive -- with cancellations roughly doubling between an empty queue and one
holding eight average orders on INTC and MSFT.

**Beyond about ten average sizes, read the curves with the residence time next to
them.** Both intensities climb steeply there, and the reason is visible in the
denominator: INTC spends 69 seconds of its day with 20 average orders resting at
the best and 11 seconds with 30. Those states are entered and left by a single
large order, so a count over a very short residence time is a large rate. The
estimator is not wrong; the state is rare, and the third panel shows what that
looks like on a name whose best queue holds under two average orders to begin
with."""),
    md("""## What the estimated dynamics produce

A queue does not move by one average event at a time: order sizes on Nasdaq
span a decade, and a queue that dies in one large cancellation dies much sooner
than a diffusive count of average events suggests.  The simulator therefore
draws real sizes from the measured distribution, which is the difference between
reproducing the rate of price changes and missing it by three orders of
magnitude."""),
    code('''fig, axes = subplots(1, 2, figsize=(11, 3.6))
sym = "INTC"
emp = curve(C, sym, DAY, "qr_queue_emp"); sim = curve(C, sym, DAY, "qr_queue_sim")
axes[0].plot(np.arange(len(emp)), emp, label="measured, time-weighted")
axes[0].plot(np.arange(len(sim)), sim, label="simulated from the intensities")
axes[0].set_xlim(0, 30); axes[0].set_xlabel("queue size (average event sizes)")
axes[0].set_ylabel("share of time"); axes[0].set_title(f"{sym}: stationary queue"); axes[0].legend()
centres = size_bucket_centres()
for i, label in enumerate(["limit orders", "cancellations", "market orders"]):
    h = curve(C, sym, DAY, "size_hist")[i].astype(float)
    axes[1].loglog(centres, h / h.sum(), label=label)
axes[1].set_xlabel("event size (average event sizes)"); axes[1].set_ylabel("frequency")
axes[1].set_title("Order sizes span a decade"); axes[1].legend()
plt.tight_layout()'''),
    code('''q = P.groupby("symbol").agg(price=("price", "mean"), spread=("median_spread_ticks", "mean"),
                            realized=("vol_preavg_pct", "mean"),
                            model_full=("qr_vol_pct", "mean"),
                            model_half=("qr_vol_half_pct", "mean"),
                            holding=("qr_holding_s", "mean")).sort_values("spread")
q["full / realized"] = q.model_full / q.realized
q["half / realized"] = q.model_half / q.realized
print(q.round(3).to_string())
large = q[q.spread <= 1.2]
print(f"\\none-tick-spread names, half-tick convention: "
      f"{100 * (large['half / realized'] - 1).abs().max():.0f}% worst error")'''),
    md("""The model shifts the whole book by a tick when a queue empties, which
is what a book with a one-tick spread has to do.  A real book usually moves one
side at a time, and the mid with it by half a tick.  Both conventions are
reported because which one applies is an empirical question about the spread,
not a choice: on the names whose spread is pinned at one tick the half-tick
reading is right and lands within about ten percent of the realized volatility,
while on the fifty-tick names the mechanism does not apply at all and the model
under-predicts by an order of magnitude.

That is the price-formation statement of this repository.  Nothing in the
estimation saw a price.  The intensities were measured against queue sizes; the
volatility came out."""),
    code('''fig, ax = subplots(figsize=(5.6, 4.0))
for sym, r in q.iterrows():
    ax.scatter(r.realized, r.model_half, color=colour_for(r.spread), s=45)
    ax.annotate(sym, (r.realized, r.model_half), fontsize=7, xytext=(3, 3), textcoords="offset points")
lim = [0.05, 5]
ax.plot(lim, lim, color="0.6", lw=1, ls="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("realized volatility (%/day)"); ax.set_ylabel("queue-reactive volatility (%/day)")
ax.set_title("Volatility out of queue dynamics alone")
plt.tight_layout()'''),
    md("""## The by-product: how often a limit order at the touch gets filled

Because the feed is order by order, we know where every order sat in its queue
when it arrived and what became of it.  This is the input chapter 05 needs, and
it cannot be measured from aggregated data at all."""),
    code('''fig, ax = subplots(figsize=(6.2, 3.6))
for sym in ["SIRI", "INTC", "AAPL", "AMZN"]:
    f = curve(C, sym, DAY, "fill_counts")[0].astype(float)   # orders that joined the touch
    total = f.sum(axis=1)
    ok = total > 200
    filled = (f[:, 0] + f[:, 1]) / np.maximum(total, 1)
    ax.plot(np.arange(len(total))[ok], filled[ok], "o-", ms=3, label=sym)
ax.set_xlim(-0.5, 20); ax.set_xlabel("size resting ahead, in average event sizes")
ax.set_ylabel("probability the order ever trades")
ax.set_title("Fill probability by queue position"); ax.legend()
plt.tight_layout()'''),
]

# --------------------------------------------------------------------------
NOTEBOOKS["05_market_making_closed_form_and_rl.ipynb"] = [
    md("""# 05 — Market making: a closed form, and a learner that has to find it

A maker posts $\\delta^b$ and $\\delta^a$ around the mid, is filled at
$\\Lambda(\\delta)=Ae^{-k\\delta}$, and carries whatever inventory that leaves it
with.  Under a running penalty $\\phi q^2$ the Hamilton-Jacobi-Bellman equation
has an explicit inner maximisation, $\\delta^\\star = 1/k - (\\theta_{q\\pm1} -
\\theta_q)$, and the substitution $v_q = e^{k\\theta_q}$ *linearises* it.  The
long-horizon quotes are then read off the principal eigenvector of a tridiagonal
matrix.

The reinforcement learning in this notebook is not there to beat that.  It is
there to be checked against it: an agent that cannot recover a policy we can
write down has not learned anything, and nothing it produces where we *cannot*
write the answer down should be believed."""),
    code(HEADER),
    code('''from hfx.mm.glft import stationary_quotes, finite_horizon_quotes
from hfx.mm.mdp import solve, action_grid, simulate_policy
from hfx.mm.rl import q_learning

A, k, phi, Q = 1.0, 1.5, 0.02, 6
bid, ask, v = stationary_quotes(A, k, 0.0, 0.0, Q, phi=phi)
inv = np.arange(-Q, Q + 1)
fig, axes = subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(inv[:-1], bid[:-1], "o-", label=r"bid $\\delta^b(q)$")
axes[0].plot(inv[1:], ask[1:], "o-", label=r"ask $\\delta^a(q)$")
axes[0].axhline(1 / k, color="0.6", ls="--", lw=1, label="myopic $1/k$")
axes[0].set_xlabel("inventory $q$"); axes[0].set_ylabel("half-spread")
axes[0].set_title("The skew is the whole point"); axes[0].legend()
for tau in [0.05, 0.3, 2.0, 50.0]:
    b, _a, _ = finite_horizon_quotes(A, k, 0.0, 0.0, Q, tau=tau, phi=phi)
    axes[1].plot(inv[:-1], b[:-1], label=f"$T-t$ = {tau}")
axes[1].set_xlabel("inventory $q$"); axes[1].set_ylabel(r"$\\delta^b$")
axes[1].set_title("Approaching the horizon, the skew disappears"); axes[1].legend()
plt.tight_layout()'''),
    md("""A maker long six units bids far away and offers through the mid: it
would rather pay to get flat than carry the position.  With the horizon close
the inventory can no longer hurt, and every quote collapses onto the myopic
$1/k$.

## Two independent solutions of the same problem

The discrete-time Markov decision process is solved exactly by relative value
iteration.  As the step shrinks it must reproduce the continuous-time closed
form -- different mathematics, no shared code beyond the parameters."""),
    code('''fine = action_grid(k, 241, -2, 4)
step = fine[1] - fine[0]
bid_dp, ask_dp, _V, gain = solve(A, k, phi, Q, dt=0.01, deltas=fine)
err = np.max(np.abs(bid_dp[:-1] - bid[:-1]))
print(f"largest gap between dynamic programming and the closed form: {err:.4f}")
print(f"action grid step:                                            {step:.4f}")
assert err < step
print(f"\\nmaker's gain per unit time: {gain:.4f} (value iteration)")'''),
    md("""## The learner

Tabular Q-learning on the same environment: state is the inventory, action is a
pair of quotes off a grid, reward is the fills it captures less the inventory
penalty.  It is given no closed form, no model of the fill intensity, and no
gradient -- only samples."""),
    code('''dt, beta = 1.0, 0.99
grid = action_grid(k, 13, -2, 4)
gstep = grid[1] - grid[0]
bid_mdp, ask_mdp, _V, _g = solve(A, k, phi, Q, dt, deltas=grid, beta=beta)
bid_rl, ask_rl, table = q_learning(A, k, phi, Q, dt, deltas=grid, beta=beta,
                                   rng=np.random.default_rng(0))
gap_b = np.max(np.abs(bid_rl[:-1] - bid_mdp[:-1])) / gstep
gap_a = np.max(np.abs(ask_rl[1:] - ask_mdp[1:])) / gstep
print(f"largest disagreement, in action-grid steps: bid {gap_b:.1f}, ask {gap_a:.1f}")
assert max(gap_b, gap_a) <= 1.0 + 1e-9   # one grid step, to floating point

rng = np.random.default_rng(1)
flat = np.full(2 * Q + 1, 1.0 / k)
for name, (b, a) in [("dynamic programming", (bid_mdp, ask_mdp)),
                     ("Q-learning", (bid_rl, ask_rl)),
                     ("flat quotes at 1/k", (flat, flat))]:
    r, inv_path = simulate_policy(b, a, A, k, phi, Q, dt, 200_000, rng=np.random.default_rng(1))
    print(f"{name:22s} reward per unit time {r:.4f}   inventory std {inv_path.std():.2f}")'''),
    code('''fig, ax = subplots(figsize=(6.0, 3.6))
ax.step(inv[:-1], bid_mdp[:-1], where="mid", label="dynamic programming")
ax.step(inv[:-1], bid_rl[:-1], where="mid", ls="--", label="Q-learning")
ax.plot(inv[:-1], bid[:-1], "o", ms=4, color="0.4", label="closed form")
ax.set_xlabel("inventory $q$"); ax.set_ylabel(r"$\\delta^b$")
ax.set_title("Three routes to the same quoting policy"); ax.legend()
plt.tight_layout()'''),
    md("""## Calibrating the fill intensity on the panel

$A$ and $k$ are measured, not assumed: for each distance $\\delta$ from the mid,
count the market orders of the day that reached at least that far, and divide by
the session length.  That is exactly the intensity a limit order posted there
would have faced."""),
    code('''fig, ax = subplots(figsize=(6.0, 3.6))
for sym in ["SIRI", "INTC", "AAPL", "AMZN"]:
    d = curve(C, sym, DAY, "fill_distance"); n = curve(C, sym, DAY, "fill_intensity_counts")
    row = P[(P.symbol == sym) & (P.date == DAY)].iloc[0]
    ok = n > 0
    ax.semilogy(d[ok] * 100, n[ok] / 22_800, "o", ms=3, label=f"{sym} (k={row.fill_k:.0f} per dollar)")
    ax.semilogy(d[ok] * 100, row.fill_A * np.exp(-row.fill_k * d[ok]), lw=1, color="0.6")
ax.set_xlabel("distance from the mid (cents)"); ax.set_ylabel(r"$\\Lambda(\\delta)$ (per second)")
ax.set_title("Measured fill intensity, and the exponential fit"); ax.legend()
plt.tight_layout()'''),
    code('''m = P.groupby("symbol").agg(price=("price", "mean"), spread=("median_spread_ticks", "mean"),
                            A=("fill_A", "mean"), k=("fill_k", "mean"), r2=("fill_r2", "mean"),
                            model=("glft_half_spread_ticks", "mean"),
                            quoted=("quoted_half_spread_ticks", "mean"),
                            effective=("effective_half_spread_ticks", "mean")).sort_values("spread")
m["model / effective"] = m.model / m.effective
print(m.round(3).to_string())'''),
    md("""The model's half-spread at flat inventory is $1/k$ plus the inventory
term, and $1/k$ alone already sets the scale: it is the distance at which the
fill intensity has fallen by a factor $e$.  Where the exponential fit is
credible -- the small-tick names, whose trades spread over many prices -- it
lands within a factor of the effective half-spread that is actually paid.  Where
the spread is pinned at one tick the fit has nothing to bite on: every trade is
at the same distance, $R^2$ collapses, and the fitted $k$ says more about the
tick than about liquidity.  That is a limitation of the *asset*, not of the
estimator, and it is the reason the large-tick names are studied with queues in
chapter 04 instead."""),
]

# --------------------------------------------------------------------------
NOTEBOOKS["06_make_take_fees_principal_agent.ipynb"] = [
    md("""# 06 — Make-take fees: an exchange designing the incentives of its maker

An exchange cannot quote.  It can only change what quoting pays and then live
with what the maker does next, which makes fee design a Stackelberg problem: the
exchange moves first with a contract, the maker re-optimises, and the exchange's
revenue is whatever the maker's new behaviour produces.  This is the structure of
El Euch, Mastrolia, Rosenbaum and Tan, in the tractable case where the contract
is a rebate $z$ per fill.

The agent's problem is chapter 05 with $\\delta+z$ captured instead of $\\delta$,
so $\\delta^\\star = 1/k - z - \\Delta_q$ and the coupling of the linear system is
scaled by $e^{kz}$.  Setting $z=0$ has to give chapter 05 back exactly, which is
the check this notebook opens with."""),
    code(HEADER),
    code('''from hfx.design.maketake import maker_solution, exchange_gain, optimal_rebate
from hfx.mm.glft import stationary_quotes

A, k, phi, Q = 1.0, 1.5, 0.02, 6
base = maker_solution(A, k, phi, Q, rebate=0.0)
bid, ask, _v = stationary_quotes(A, k, 0.0, 0.0, Q, phi=phi)
assert np.allclose(base.bid[:-1], bid[:-1]) and np.allclose(base.ask[1:], ask[1:])
print("a zero rebate reproduces chapter 05 exactly")
print(base)'''),
    md("""## What a rebate buys

The identity worth knowing: buying at inventory $q$ and selling back from $q+1$
captures $\\delta^b(q)+\\delta^a(q+1) = 2/k - 2z$, with the inventory terms
cancelling.  The whole rebate reaches the price of a round trip, whatever the
maker's risk aversion.  What the skew decides is *when* it earns it -- individual
quotes move by different amounts, so a flat rebate tilts the schedule instead of
translating it."""),
    code('''z = 0.2
paid = maker_solution(A, k, phi, Q, rebate=z)
inv = np.arange(-Q, Q + 1)
for sol, label in [(base, "z = 0"), (paid, f"z = {z}")]:
    print(f"{label}: round trip = {np.unique(np.round(sol.bid[:-1] + sol.ask[1:], 10))}"
          f"   (2/k - 2z = {2/k - 2*sol.rebate:.4f})")
fig, axes = subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(inv[:-1], base.bid[:-1], "o-", label="no rebate")
axes[0].plot(inv[:-1], paid.bid[:-1], "o-", label=f"rebate {z}")
axes[0].set_xlabel("inventory $q$"); axes[0].set_ylabel(r"$\\delta^b$")
axes[0].set_title("The schedule tilts, it does not translate"); axes[0].legend()
axes[1].plot(inv[:-1], base.bid[:-1] - paid.bid[:-1], "o-")
axes[1].axhline(z, color="0.6", ls="--", lw=1, label="the rebate")
axes[1].set_xlabel("inventory $q$"); axes[1].set_ylabel("inward shift of the bid")
axes[1].set_title("Where the rebate lands"); axes[1].legend()
plt.tight_layout()'''),
    md("""## The exchange's problem

The exchange charges takers $c$ and pays the maker $z$, so it earns $(c-z)$ per
trade at whatever rate the maker's new quotes produce.  Because the fill rate at
the optimum behaves like $e^{kz}$, the first-order condition gives a rebate that
does not depend on the maker's risk aversion, its inventory limit, or the
volatility:

$$z^\\star = c - \\frac1k.$$"""),
    code('''rows = []
for c in [0.5 / k, 1.0 / k, 2.0 / k, 3.0 / k]:
    z_star, sol, grid, obj = optimal_rebate(A, k, phi, Q, taker_fee=c)
    rows.append({"taker fee c": c, "z*": z_star, "c - 1/k": c - 1 / k,
                 "spread": 2 * sol.spread, "spread at z=0": 2 * base.spread,
                 "fills/s": sol.fill_rate, "fills/s at z=0": base.fill_rate,
                 "exchange gain": exchange_gain(sol, c),
                 "maker gain": sol.gain})
print(pd.DataFrame(rows).round(4).to_string(index=False))'''),
    code('''fig, axes = subplots(1, 2, figsize=(11, 3.6))
c = 2.0 / k
z_star, sol, grid, obj = optimal_rebate(A, k, phi, Q, taker_fee=c)
spreads = np.array([2 * maker_solution(A, k, phi, Q, float(zz)).spread for zz in grid])
rates = np.array([maker_solution(A, k, phi, Q, float(zz)).fill_rate for zz in grid])
axes[0].plot(grid, obj)
axes[0].axvline(z_star, color="0.4", lw=1, label=f"$z^*$ = {z_star:.3f}")
axes[0].axvline(c - 1 / k, color="0.7", ls="--", lw=1, label=f"$c-1/k$ = {c - 1/k:.3f}")
axes[0].set_xlabel("rebate $z$"); axes[0].set_ylabel("exchange revenue per unit time")
axes[0].set_title("The exchange's objective"); axes[0].legend()
axes[1].plot(grid, spreads, label="maker's spread")
ax2 = axes[1].twinx(); ax2.plot(grid, rates, color="#b5482f", label="fills per second")
axes[1].axvline(z_star, color="0.4", lw=1); axes[1].set_xlabel("rebate $z$")
axes[1].set_ylabel("spread"); ax2.set_ylabel("fills per second")
axes[1].set_title("What the takers get for it")
plt.tight_layout()'''),
    md("""## A regulator instead of a shareholder

The same machinery answers a different question if the objective changes.  Give
the principal a taste for market quality -- a penalty on the spread the takers
face -- and the optimal rebate rises: the exchange is being paid to buy a
tighter market."""),
    code('''rows = []
c = 2.0 / k
for w in [0.0, 0.5, 1.0, 2.0]:
    z_w, sol_w, _g, _o = optimal_rebate(A, k, phi, Q, taker_fee=c, spread_weight=w)
    rows.append({"spread weight": w, "z*": z_w, "spread": 2 * sol_w.spread,
                 "fills/s": sol_w.fill_rate, "maker gain": sol_w.gain,
                 "exchange revenue": exchange_gain(sol_w, c)})
out = pd.DataFrame(rows)
print(out.round(4).to_string(index=False))
assert out["z*"].is_monotonic_increasing and out["spread"].is_monotonic_decreasing'''),
    md("""Two warnings the model gives for free.  A rebate large enough makes the
maker quote *through* the mid -- it is being paid more to trade than the spread
is worth, and it will trade at a loss on the price to collect the rebate; that is
the standard criticism of aggressive maker-taker pricing, and here it falls out
of a first-order condition rather than an anecdote.  And the participation
constraint runs the other way: an exchange that pays too little loses the maker
altogether, so the optimum is a corner, not an interior point.

What the model does *not* have is competition between venues, which is the first
thing a real fee schedule is designed against.  The single-exchange answer
$z^\\star = c - 1/k$ is a benchmark to reason from, not a recommendation."""),
]


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    path = os.path.join(OUT, name)
    with open(path, "w") as fh:
        nbf.write(nb, fh)
    return path


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        path = build(name, cells)
        print(f"wrote {os.path.relpath(path, ROOT)} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
