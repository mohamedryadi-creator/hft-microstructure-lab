# Measurements

Everything here is produced by [`scripts/run_study.py`](../scripts/run_study.py)
from the Nasdaq ITCH messages extracted by
[`scripts/build_dataset.py`](../scripts/build_dataset.py), and committed so that
the notebooks — and any reader — work from real measurements without downloading
a byte.

Regenerate with `make data && make study`: about 31 GB is transferred, streamed
one file at a time and never written to disk, and roughly 15 minutes of
computation follows.

## Files

| File | Contents |
|---|---|
| `panel.csv` | One row per (symbol, day): fitted exponents, spreads, volatilities by four estimators, queue statistics, and the quality flags that say whether each fit can be trusted |
| `curves.npz` | Per (symbol, day), keyed `SYMBOL\|DATE\|name`: signature plots, Hawkes kernels and their implied variance curves, queue-reactive intensity tables (marginal and joint), event-size and fill-probability histograms |
| `learning.csv` | Chapter 07: one row per (symbol, model, scope) with AUC, Brier, log loss and accuracy on the held-out days |
| `learning.npz` | Chapter 07: the analytic, fitted and empirical probability surfaces on the joint queue grid, calibration curves, and the logistic coefficients |
| `agents.csv` | Chapter 08: the best blind and sighted quoting policies per symbol, their standard errors, and the rebate each family needs to break even |
| `agents.npz` | Chapter 08: the full policy-search grid, the rebate frontiers, and the deep agent's training loss |

## Reading the quality flags before the numbers

Several columns are estimates that do not always succeed, and the file says so
rather than hiding it.

- `fill_r2` — quality of the exponential fit to the fill-intensity curve. It
  falls to 0.15–0.43 on the large-tick names, where every trade prints at the
  same distance from the mid and the exponential has nothing to bite on. Below
  about 0.9, `fill_A` and `fill_k` should not be quoted.
- `eta` — the uncertainty-zone parameter. Values above 1/2 are outside the range
  the model was written for; they are reported as measured, and `eta_mid` gives
  the same estimator applied to the mid quote instead of the traded price.
- `grid_rv_over_uz` against `variance_inflation_pred` — a self-consistency check
  of the uncertainty-zone model that holds to 0.02–0.06 on large-tick names and
  fails by 0.2–0.5 on the others. It marks the regime boundary.
- `hawkes_ks_p` — the goodness-of-fit p-value. It is essentially zero
  everywhere: the fitted branching ratio is a summary of the flow, not a
  validated model of it.
- `at_touch_displayed` — the share of displayed trades printing at or through the
  touch. This is the check that the aggressor side and the quote alignment are
  both right; it should be above 0.95, and it is.
- `qr_moves_simulated` — how many price moves the queue-reactive simulation
  completed before its step budget ran out. Below about 200 the derived
  volatility is not reported at all.
- `fallback_share` in `learning.csv` — the share of states where the joint
  two-queue intensities had too little residence time to be estimated and the
  own-queue rates stood in. It runs from 0.16 on INTC to 0.95 on ISRG, and it is
  the same tick axis as everything else: the joint tables are only usable where
  the book actually visits the grid.
- `blind_se` and `sighted_se` in `agents.csv` — standard errors across simulation
  seeds. The environment's randomness does not depend on the agent, so policies
  are compared on identical price paths and these are paired errors, which is why
  they are two orders of magnitude below the effect they are measuring.

## Data provenance

See [`../NOTICE`](../NOTICE). No raw Nasdaq data is redistributed here.
