r"""One symbol-day, from reconstructed messages to every number the study needs.

Everything here reads the parquet written by :mod:`hfx.pipeline.build` and
writes small derived measurements, so the notebooks -- and any reader -- work
from ``results/`` without downloading a byte.  The raw feed is never
redistributed; the fitted exponents, the intensity tables and the curves are.
"""

from __future__ import annotations

import numpy as np

from ..book.replay import NQ, TICK, replay, size_bucket_centres
from ..hawkes.fit import fit_exp_bank, log_grid
from ..hawkes.gof import ks_exponential, rescaled_residuals
from ..hawkes.spectrum import empirical_signature_plot, signature_plot_spectral
from ..itch.spec import CLOSE_NS, OPEN_NS, PRICE_SCALE
from ..mm.glft import fill_intensity_curve, stationary_quotes
from ..queue import reactive as qr
from ..vol import estimators as ve
from ..vol import uncertainty_zones as uz

#: Statistics use 09:35-15:55.  The first and last minutes of the session are a
#: different market -- the book is still forming after the opening cross and
#: unwinding into the closing one -- and they contribute spreads of seventy
#: ticks that are real but not what any of these models is about.
STAT_OPEN = OPEN_NS + 5 * 60_000_000_000
STAT_CLOSE = CLOSE_NS - 5 * 60_000_000_000
SECONDS = (STAT_CLOSE - STAT_OPEN) / 1e9

#: Sampling scales for the signature plot, in seconds.
TAUS = np.geomspace(0.05, 1800.0, 24)
#: Lags for the trade-sign autocorrelation.
MAX_HAWKES_EVENTS = 60_000


def _mid_on_grid(quotes, step_seconds: float):
    """Last mid quote on a regular grid, over the statistics window."""
    ts, bid, ask = quotes["ts"], quotes["bid"], quotes["ask"]
    keep = (ts >= STAT_OPEN) & (ts <= STAT_CLOSE)
    ts, bid, ask = ts[keep], bid[keep], ask[keep]
    if ts.size < 10:
        return np.empty(0)
    mid = (bid + ask) / 2.0 / PRICE_SCALE
    grid = np.arange(STAT_OPEN, STAT_CLOSE, int(step_seconds * 1e9))
    idx = np.searchsorted(ts, grid, side="right") - 1
    ok = idx >= 0
    return mid[idx[ok]]


def study_symbol_day(events, symbol: str, date: str, rng=None) -> tuple[dict, dict]:
    """Return ``(row, curves)`` for one symbol-day."""
    rng = np.random.default_rng(0) if rng is None else rng
    out = replay(events, symbol, date)
    trades, quotes = out.trades, out.quotes
    aes = out.stats["aes"]
    row: dict = {"symbol": symbol, "date": date, "aes": aes}
    curves: dict = {}

    # ---------- chapter 01: the book, the tick, the flow -----------------
    spread_time = out.spread_time.astype(float)
    total = spread_time.sum()
    row["n_events"] = out.stats["n_events"]
    row["n_unknown_ref"] = out.stats["n_unknown_ref"]
    row["p_spread_one_tick"] = float(spread_time[1] / total) if total else np.nan
    row["mean_spread_ticks"] = (
        float(np.average(np.arange(spread_time.size), weights=spread_time)) if total else np.nan
    )
    cum = np.cumsum(spread_time) / total if total else np.zeros_like(spread_time)
    row["median_spread_ticks"] = float(np.searchsorted(cum, 0.5)) if total else np.nan
    row["p90_spread_ticks"] = float(np.searchsorted(cum, 0.9)) if total else np.nan
    curves["spread_time"] = spread_time
    curves["minute_counts"] = out.minute_counts
    curves["queue_time"] = out.queue_time
    curves["imbalance_time"] = out.imbalance_time

    keep = (
        (trades["ts"] >= STAT_OPEN)
        & (trades["ts"] <= STAT_CLOSE)
        & (trades["bid"] > 0)
        & (trades["ask"] > 0)
        # A spread cap, relative rather than absolute: five times the median is
        # wide for INTC and narrow for AMZN, and a fixed number of ticks would
        # keep everything for one and throw away everything for the other.  What
        # it removes is the book still forming, not normal trading.
        & (
            (trades["ask"] - trades["bid"])
            <= max(5.0, 5.0 * row["median_spread_ticks"]) * TICK
        )
    )
    tr = {k: v[keep] for k, v in trades.items()}
    n_tr = tr["ts"].size
    row["n_trades"] = int(n_tr)
    if n_tr < 500:
        return row, curves

    mid = (tr["bid"] + tr["ask"]) / 2.0
    displayed = tr["hidden_size"] == 0
    at_touch = np.where(tr["side"] > 0, tr["price"] >= tr["ask"], tr["price"] <= tr["bid"])
    row["price"] = float(np.mean(tr["price"]) / PRICE_SCALE)
    row["volume"] = int(tr["size"].sum())
    row["hidden_share"] = float(tr["hidden_size"].sum() / tr["size"].sum())
    row["at_touch_displayed"] = float(at_touch[displayed].mean())
    row["mean_order_size"] = float(tr["size"].mean())
    row["trades_per_second"] = float(n_tr / SECONDS)
    # Effective half-spread actually paid, in ticks, on displayed trades.
    row["effective_half_spread_ticks"] = float(
        np.mean(tr["side"][displayed] * (tr["price"][displayed] - mid[displayed])) / TICK
    )
    row["quoted_half_spread_ticks"] = float(np.mean(tr["ask"] - tr["bid"]) / (2 * TICK))

    # ---------- adverse selection: what the maker does not keep ----------
    # The effective half-spread is what the taker pays at the instant of the
    # trade.  The maker only keeps it if the mid stays put, and it does not: a
    # buy is followed by an up-move on average, so the maker who sold it is
    # holding a position that has already moved against them.  The difference is
    # the adverse-selection cost, and it needs no new data -- the pre-trade mid
    # is already on every trade and the mid path is already in the quotes.
    # The decomposition is an identity: what the taker pays at the instant of
    # the trade is what the maker keeps plus what the mid takes back.
    #     s_effective = s_realized(h) + adverse_selection(h)
    # A model with no informed flow in it -- Avellaneda-Stoikov, and the closed
    # form of chapter 05 -- describes the maker's *net* capture, so it is
    # s_realized it should be compared against, not s_effective.  Horizons run
    # down to ten milliseconds because the correction is already large there.
    horizons = np.array([0.01, 0.05, 0.1, 1.0, 10.0, 60.0])
    q_ts = quotes["ts"]
    q_mid = (quotes["bid"] + quotes["ask"]) / 2.0
    drift = np.full(horizons.size, np.nan)
    if q_ts.size > 100:
        for h, horizon in enumerate(horizons):
            later = np.searchsorted(q_ts, tr["ts"] + int(horizon * 1e9), side="right") - 1
            ok = (later >= 0) & (tr["ts"] + horizon * 1e9 <= q_ts[-1])
            if ok.sum() > 100:
                signed = tr["side"][ok] * (q_mid[later[ok]] - mid[ok])
                drift[h] = float(np.mean(signed) / TICK)
        for h, horizon in enumerate(horizons):
            row[f"adverse_selection_{horizon:g}s_ticks"] = drift[h]
            row[f"realized_half_spread_{horizon:g}s_ticks"] = (
                row["effective_half_spread_ticks"] - drift[h]
            )
    curves["impact_horizons"] = horizons
    curves["impact_ticks"] = drift

    # ---------- chapter 05 input: the fill intensity curve ----------------
    distance = (tr["side"] * (tr["price"] - mid)).astype(float) / PRICE_SCALE
    grid = np.linspace(0.0, np.percentile(distance, 99.5), 24)
    reached = np.array([(distance >= d).sum() for d in grid], dtype=float)
    A_hat, k_hat, r2 = fill_intensity_curve(grid, reached, SECONDS)
    row["fill_A"] = A_hat
    row["fill_k"] = k_hat
    row["fill_r2"] = r2
    curves["fill_distance"] = grid
    # Not "fill_counts": chapter 04 uses that name for the fate of orders by
    # queue position, and the later assignment silently overwrote this one.
    curves["fill_intensity_counts"] = reached

    # ---------- chapter 02: Hawkes on the market-order flow ---------------
    t_sec = (tr["ts"] - STAT_OPEN) / 1e9
    marks = (tr["side"] < 0).astype(np.int64)          # 0 = buy, 1 = sell
    if n_tr > MAX_HAWKES_EVENTS:
        t_sec, marks = t_sec[:MAX_HAWKES_EVENTS], marks[:MAX_HAWKES_EVENTS]
    span = float(t_sec[-1] - t_sec[0])
    betas = log_grid(0.05, 500.0, 7)
    fit = fit_exp_bank(t_sec, marks, betas=betas, T=span, d=2, symmetric=True)
    norms = fit.branching_matrix
    row["hawkes_self"] = float(norms[0, 0])
    row["hawkes_cross"] = float(norms[0, 1])
    row["hawkes_branching"] = float(fit.branching_ratio)
    row["hawkes_delta"] = float(norms[0, 0] - norms[0, 1])
    row["hawkes_lambda"] = float(fit.mean_intensity()[0])
    residuals = rescaled_residuals(t_sec, marks, fit)[0]
    stat, pvalue = ks_exponential(residuals)
    row["hawkes_ks"] = stat
    row["hawkes_ks_p"] = pvalue
    single = fit_exp_bank(t_sec, marks, betas=np.array([1.0]), T=span, d=2, symmetric=True)
    row["hawkes_branching_single"] = float(single.branching_ratio)
    row["hawkes_ks_p_single"] = ks_exponential(rescaled_residuals(t_sec, marks, single)[0])[1]

    taus_flow = np.geomspace(0.05, min(600.0, span / 20), 16)
    curves["hawkes_taus"] = taus_flow
    curves["hawkes_signature_model"] = signature_plot_spectral(
        fit.alpha[0, 0], fit.alpha[0, 1], fit.betas, row["hawkes_lambda"], taus_flow
    )
    curves["hawkes_signature_empirical"] = empirical_signature_plot(
        t_sec, marks, taus_flow, T=span
    )
    curves["hawkes_kernel_lags"] = np.geomspace(1e-3, 100.0, 60)
    curves["hawkes_kernel"] = fit.kernel(curves["hawkes_kernel_lags"])
    curves["hawkes_residual_qq"] = np.column_stack(
        [np.sort(residuals)[:: max(1, residuals.size // 500)]]
    )

    # ---------- chapter 03: realized volatility and the tick --------------
    # A tenth of a second, not a second.  The bid-ask bounce lives below the
    # second; sampled at one second the mid of these names is already past the
    # minimum of its signature plot, and the noise variance comes back zero --
    # true, and useless.
    log_mid = np.log(_mid_on_grid(quotes, 0.1))
    if log_mid.size > 1000:
        row["rv_1s"] = ve.realized_variance(log_mid)
        row["preavg"] = ve.pre_averaged(log_mid)
        row["tsrv"] = ve.two_scale(log_mid)
        row["kernel"] = ve.realized_kernel(log_mid)
        row["noise_var"] = ve.noise_variance(log_mid, iv=row["preavg"])
        row["vol_preavg_pct"] = 100 * float(np.sqrt(max(row["preavg"], 0.0)))
        row["optimal_sampling"] = ve.optimal_sampling(row["preavg"], row["noise_var"])
        row["rv_10s"] = ve.realized_variance(log_mid[::100])
        steps = np.unique(np.round(np.geomspace(1, 9000, 22)).astype(int))
        curves["signature_steps"] = steps
        curves["signature_rv"] = ve.signature_plot(log_mid, steps)

    levels = (tr["price"][displayed] // TICK).astype(np.int64)
    eta, n_cont, n_alt = uz.estimate_eta(levels)
    # The same estimator on the mid, which for a venue that prints only part of
    # the consolidated tape sees the price move when this venue does not trade.
    q_keep = (quotes["ts"] >= STAT_OPEN) & (quotes["ts"] <= STAT_CLOSE)
    mid_levels = ((quotes["bid"][q_keep] + quotes["ask"][q_keep]) // 2) // (TICK // 2)
    row["eta_mid"] = uz.estimate_eta(mid_levels)[0]
    row["eta"] = eta
    row["eta_continuations"] = n_cont
    row["eta_alternations"] = n_alt
    row["eta_se"] = (
        float(eta * np.sqrt(1 / max(n_cont, 1) + 1 / max(n_alt, 1))) if np.isfinite(eta) else np.nan
    )
    tick = TICK / PRICE_SCALE
    row["implicit_spread_ticks"] = (
        float(uz.implicit_spread(tick, eta) / tick) if np.isfinite(eta) else np.nan
    )
    if np.isfinite(eta):
        iv_uz = uz.integrated_variance(levels, tick)
        row["uz_variance"] = float(iv_uz / row.get("price", np.nan) ** 2)
        row["uz_vol_pct"] = 100 * float(np.sqrt(max(row["uz_variance"], 0.0)))
        rv_grid = ve.realized_variance(levels * tick) / row["price"] ** 2
        row["grid_rv_over_uz"] = float(rv_grid / row["uz_variance"]) if row["uz_variance"] > 0 else np.nan
        row["variance_inflation_pred"] = float(uz.variance_inflation(eta))

    # ---------- chapter 04: the queue-reactive model ----------------------
    ev_counts = out.qr_events[0, 0] + out.qr_events[0, 1]
    ev_time = out.qr_time[0, 0] + out.qr_time[0, 1]
    lam, valid = qr.intensities(ev_counts, ev_time, min_seconds=2.0)
    curves["qr_lambda"] = lam
    curves["qr_time"] = ev_time
    # The joint two-queue tables: the same estimates conditioned on both sides
    # at once, which is what chapter 07 solves the first-passage problem on.
    curves["qr_events2"] = out.qr_events2
    curves["qr_time2"] = out.qr_time2
    curves["qr_regen"] = out.regen_hist
    curves["size_hist"] = out.size_hist
    curves["fill_counts"] = out.fill_counts
    centres = size_bucket_centres()
    samplers = [qr.SizeSampler(out.size_hist[k], centres) for k in range(3)]
    regen = (out.regen_hist[0] + out.regen_hist[1]).astype(float)
    emp_queue = (out.queue_time[0] + out.queue_time[1]).astype(float)
    if emp_queue.sum() > 0:
        row["qr_mean_queue_emp"] = float(np.average(np.arange(NQ), weights=emp_queue))
    if regen.sum() > 0 and np.isfinite(lam).any():
        _dirs, holding, visited = qr.simulate(lam, samplers, regen, 2000, rng=rng)
        row["qr_moves_simulated"] = int(holding.size)
    else:
        holding, visited = np.empty(0), np.empty(0)
    if holding.size >= 200:
        row["qr_holding_s"] = float(np.mean(holding))
        row["qr_vol_pct"] = 100 * qr.implied_volatility(holding, tick, row["price"])
        # The model shifts the whole book a tick when a queue empties.  A real
        # book usually moves one side, and the mid with it by half a tick, so
        # the same depletion rate implies half the volatility.  Both are
        # reported: which one is right is an empirical question about the
        # spread, not something to settle by choosing a convention.
        row["qr_vol_half_pct"] = row["qr_vol_pct"] / 2.0
        row["qr_mean_queue_sim"] = float(np.mean(np.clip(visited, 0, NQ - 1)))
        sim_hist, _ = np.histogram(np.clip(visited, 0, NQ - 1), bins=np.arange(NQ + 1))
        curves["qr_queue_sim"] = sim_hist.astype(float) / max(sim_hist.sum(), 1)
        curves["qr_queue_emp"] = emp_queue / emp_queue.sum()

    # ---------- chapter 05: the quotes those parameters imply -------------
    if np.isfinite(A_hat) and np.isfinite(k_hat) and k_hat > 0 and "preavg" in row:
        sigma = float(np.sqrt(row["preavg"] / SECONDS)) * row["price"]
        phi = 1e-6
        try:
            bid, ask, _v = stationary_quotes(A_hat, k_hat, 0.0, sigma, 8, phi=phi)
            curves["glft_bid"] = bid
            curves["glft_ask"] = ask
            row["glft_half_spread_flat"] = float(bid[8])
            row["glft_half_spread_ticks"] = float(bid[8] / tick)
        except Exception:
            pass
    return row, curves
