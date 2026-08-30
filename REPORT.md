# High-frequency price formation, market design, and reinforcement learning

**A study of Nasdaq TotalView-ITCH message data across seven trading days and twelve symbols.**

This report is the written account of the repository: what was modelled, what was
measured, what the measurements say, and where each model stops working. The
mathematics is derived in full in the companion French document
(`theorie.pdf`, kept outside the repository); this is the English record of the
implementation and the results.

---

## 1. What the project is about, and why it is organised this way

Three questions run through the six chapters.

1. **How is a price formed?** Not "what process does it follow" but what
   mechanically has to happen for a printed price to change.
2. **What does a market maker do, and what should an exchange pay it to do?**
3. **When a model has a closed form, does an estimator — or a learning agent —
   recover it?**

They are held together by one variable: the **ratio of the spread to the tick**.
Nasdaq's tick is one cent for every stock above a dollar, so the price level sets
that ratio. A cent is 17 basis points of SIRI at \$6 and 0.06 basis points of
AMZN at \$1 800, and the two markets that result do not form prices the same way.
Every chapter below is read twice, once on each side of that line, and several of
them stop working on one side. Saying exactly where is most of the value.

## 2. Method: two legs, and neither works alone

The repository never validates an estimator on the data it is supposed to
measure.

- **A simulator proves the estimator is correct.** Inject a kernel, a
  volatility, an $\eta$, a set of intensities that you chose, and check that the
  estimator hands it back. That tests the algebra and its transcription — nothing
  else.
- **Real data says what the market does.** An estimator validated only on
  synthetic data says nothing about the world; a measurement whose estimator was
  never validated says nothing at all.

Every chapter has both legs, and the repository's test suite is the first one:
63 tests, none touching the network, several of them Monte Carlo studies whose
tolerances are derived rather than tuned.

## 3. The data

Nasdaq publishes seven complete TotalView-ITCH 5.0 days for free at
`https://emi.nasdaq.com/ITCH/Nasdaq ITCH/`. Each file is 3.5–5.6 GB gzipped, 11–14 GB
inflated, and carries **every** add, execute, cancel, delete and replace message
for **every** Nasdaq-listed security on that day, with nanosecond timestamps and
order reference numbers.

| | |
|---|---|
| Days | 2019-01-30, 2019-03-27, 2019-07-30, 2019-08-30, 2019-10-30, 2019-12-30, 2020-01-30 |
| Symbols | SIRI, MU, INTC, CSCO, MSFT, AAPL, TSLA, NFLX, REGN, ISRG, GOOG, AMZN |
| Messages read | ≈ 2.7 billion |
| Messages kept | 66 592 917 |
| Market orders reconstructed | 1 347 006 |

Three properties of this feed are what make the study possible at all.

- **It is order by order.** Every message carries an order reference, so the book
  can be rebuilt exactly and a specific order can be followed from arrival to
  fill or cancellation. The fill-probability curve of chapter 04 cannot be
  measured from any aggregated feed.
- **The aggressor side is known, not inferred.** An execution consumes a resting
  order whose side the book knows, so the aggressor is its opposite. There is no
  Lee-Ready step and therefore none of the attenuation that a sign-inference
  error inflicts on every impact estimate downstream.
- **There is a real tick.** The whole tick-size axis exists only because the
  price grid does.

The extraction streams each file from the socket, inflates it in memory, keeps
the panel's messages and discards the rest; the raw file is never written to
disk. About 580 MB of parquet lands, and only the derived measurements —
roughly 1.5 MB — are committed, so every notebook runs offline.

### 3.1 Reading the binary correctly

Everything rests on the decoder. It is checked two ways. The message-length table
is verified against the real feed (every length in the 2019-01-30 file agrees),
and a wrong length raises immediately rather than shifting the stream silently.
And an **independently written encoder** (`hfx.itch.synth`) produces messages from
the published field offsets which the decoder reads back field by field, so a
mistake in the transcription fails the test suite rather than the analysis.

The reconstruction is then checked against the exchange's own prints: across all
84 symbol-days there were **zero** executions, cancels or deletes referencing an
order the book did not have.

---

## 4. Chapter 01 — The book, and the one axis the panel varies along

The book is rebuilt from the message flow and the panel is described by the
statistics that matter for everything downstream.

| symbol | price ($) | median spread (ticks) | P(1 tick) | market orders | per second | mean size | AES | best queue (AES) | hidden share | at/through touch |
|---|---|---|---|---|---|---|---|---|---|---|
| **CSCO** | 49.21 | 1.00 | 0.980 | 61,900 | 0.39 | 288 | 235 | 8.3 | 0.138 | 0.9559 |
| **INTC** | 54.48 | 1.00 | 0.981 | 83,550 | 0.52 | 292 | 208 | 7.8 | 0.116 | 0.9688 |
| **MU** | 46.43 | 1.00 | 0.947 | 104,508 | 0.65 | 271 | 234 | 4.8 | 0.114 | 0.9600 |
| **SIRI** | 6.38 | 1.00 | 1.000 | 16,007 | 0.10 | 1,955 | 2,876 | 23.2 | 0.190 | 0.9561 |
| **MSFT** | 139.12 | 1.14 | 0.737 | 204,001 | 1.28 | 164 | 119 | 3.9 | 0.143 | 0.9786 |
| **AAPL** | 231.73 | 2.29 | 0.231 | 287,771 | 1.80 | 125 | 123 | 1.8 | 0.152 | 0.9895 |
| **NFLX** | 324.25 | 12.14 | 0.001 | 116,633 | 0.73 | 71 | 81 | 1.2 | 0.253 | 0.9937 |
| **TSLA** | 344.89 | 15.57 | 0.001 | 229,587 | 1.44 | 85 | 87 | 1.5 | 0.264 | 0.9916 |
| **REGN** | 347.86 | 27.00 | 0.000 | 23,202 | 0.15 | 41 | 78 | 0.7 | 0.292 | 0.9948 |
| **ISRG** | 545.34 | 39.43 | 0.000 | 25,066 | 0.16 | 35 | 64 | 0.8 | 0.328 | 0.9964 |
| **AMZN** | 1,797.43 | 49.86 | 0.000 | 136,089 | 0.85 | 37 | 58 | 0.9 | 0.328 | 0.9977 |
| **GOOG** | 1,244.22 | 53.71 | 0.000 | 58,692 | 0.37 | 36 | 46 | 1.1 | 0.323 | 0.9970 |

Read the table from the top down and the market changes character. SIRI's spread
is one tick 100% of the time and its best queue holds eighteen average orders;
AMZN's spread is fifty ticks and its best queue holds barely one. Liquidity in a
large-tick name has nowhere to go but into the queue at the touch, because there
is no room between the quotes; in a small-tick name the queue is thin and the
competition happens on price instead.

**The aggressor-side convention survives contact with the data.** A trade
classified as buyer-initiated has to print at or above the offer, and across the
panel between 95.6% and 99.8% of displayed trades do, with the worst single
symbol-day at 92.1%. The control is the hidden trades: those are non-displayed
liquidity and print *inside* the spread, which is precisely what they do —
95–99% of them, depending on the symbol.

**Takers pay less than the screen shows.** The effective half-spread is below the
quoted one everywhere, and the gap widens with the tick ratio: with a fifty-tick
spread there is room for hidden and price-improving liquidity to sit inside it,
and takers find it. Chapter 03 reaches the same conclusion from a completely
different direction, through the tick grid.

## 5. Chapter 02 — Hawkes processes, and a volatility predicted from the flow

A Hawkes process is the smallest model that says why market orders arrive in
bursts: every event raises the intensity of the next. The spectral radius of
$\int\varphi$ is the average number of events each event triggers — the
**endogeneity ratio**.

Two implementation choices are the substance of the chapter. The decay rates are
**fixed** on a geometric grid and only the non-negative amplitudes are estimated,
which makes the log-likelihood concave: the estimate is a global maximum reached
by a convex solver, not the point at which an optimiser happened to stop. And the
model is tested by **time rescaling** — under the fitted intensity the
compensator-rescaled inter-arrival times are exactly i.i.d. $\mathrm{Exp}(1)$, so
a Kolmogorov-Smirnov test is a test of the model rather than a plot of a fit next
to the data it was fitted to.

| symbol | endogeneity n | day-to-day sd | n, single exponential | delta = self - cross | largest KS p |
|---|---|---|---|---|---|
| **CSCO** | 0.798 | 0.046 | 0.676 | 0.583 | 0.00000000 |
| **INTC** | 0.818 | 0.045 | 0.717 | 0.556 | 0.00000000 |
| **MU** | 0.845 | 0.045 | 0.726 | 0.545 | 0.00000000 |
| **SIRI** | 0.825 | 0.057 | 0.553 | 0.757 | 0.00004024 |
| **MSFT** | 0.857 | 0.075 | 0.745 | 0.570 | 0.00000000 |
| **AAPL** | 0.857 | 0.021 | 0.780 | 0.641 | 0.00000000 |
| **NFLX** | 0.861 | 0.021 | 0.771 | 0.660 | 0.00000000 |
| **TSLA** | 0.899 | 0.061 | 0.822 | 0.697 | 0.00000000 |
| **REGN** | 0.725 | 0.040 | 0.698 | 0.571 | 0.00000000 |
| **ISRG** | 0.744 | 0.088 | 0.696 | 0.590 | 0.00000000 |
| **AMZN** | 0.887 | 0.023 | 0.781 | 0.708 | 0.00000000 |
| **GOOG** | 0.818 | 0.034 | 0.760 | 0.610 | 0.00000000 |

Three things to take from this table.

**The flow is strongly endogenous and remarkably stable.** The endogeneity ratio
sits between 0.564 and 1.000 across the 84 symbol-days, with a median of 0.842
and a per-symbol average from 0.725 to 0.899; its day-to-day standard deviation
within a symbol has a median of 0.045 — a number that moves far less
across a year than the volatility does.

**The kernel bank matters.** A single exponential gives a systematically lower
ratio, because it cannot represent the slow part of the excitation and attributes
it to the baseline instead. The gap is 0.1 to 0.3 depending on the symbol, which
is the difference between "most of the flow is exogenous" and "most of it is
not".

**Every fit is rejected.** The largest KS p-value anywhere in 84 symbol-days is
$4.0\times10^{-5}$, on SIRI, the least active name in the panel; every other
symbol's best day is numerically zero. This is the expected outcome and
it is worth stating plainly: a Hawkes process with a handful of exponentials is
not the law of the order flow. What survives rejection is the summary — how much
of the flow triggers itself, and in which direction.

**Self-excitation dominates in every single symbol-day.** $\delta = s - c$ is
positive in **84 out of 84** symbol-days, with per-symbol averages from 0.545 to
0.757. A buy makes the next buy more
likely than it makes a sell: order splitting beats the reaction of the other
side. That has a consequence for the price, and the model states it in closed
form.

### 5.1 The signature plot the flow implies

For a symmetric buy/sell flow the Bartlett spectrum of the signed trade count
$P_t = N^+_t - N^-_t$ collapses to a scalar, and the variance per unit time over
a scale $\tau$ follows. Two limits name the whole curve:

$$V(0^+) = 2\Lambda, \qquad V(\infty) = \frac{2\Lambda}{(1-\delta)^2},$$

so $V(\infty)/V(0^+) = (1-\delta)^{-2}$: the entire microstructure-noise
structure of the signed flow, predicted from the *times and signs of trades* with
no price entering the estimation. With $\delta > 0$ on every symbol-day, the
prediction is that the variance per unit time **rises** with the scale — the flow
is persistent, not alternating, so the textbook decreasing signature plot is not
what the trade signs do.

Comparing that curve with the one measured on the same day is a genuine
out-of-sample check, and it passes at the short end and fails at the long end:
the measured variance keeps growing past the scale at which the fitted kernel has
decayed. That is the verdict of the goodness-of-fit test seen in the price
instead of the residuals, and it is the standard argument for a power-law kernel.

## 6. Chapter 03 — Realized volatility, microstructure noise, and the tick

Sampling a noisy price more finely does not measure its volatility better. With
$Y = X + \varepsilon$,

$$\mathbb{E}\,RV_n = IV + 2n\,\mathbb{E}[\varepsilon^2],$$

so the estimator diverges as the sampling tightens. Three estimators fix it at
three different rates, and the rates are **measured** here, not quoted: simulate
a price whose integrated variance we chose, add noise, estimate, and regress
$\log$ RMSE on $\log n$.

| estimator | theoretical rate | measured slope |
|---|---|---|
| two-scale | $n^{-1/6} = -0.167$ | **−0.166** |
| pre-averaged | $n^{-1/4} = -0.250$ | **−0.249** |
| realized kernel (Parzen, non-negative) | $n^{-1/5} = -0.200$ | **−0.191** |

The kernel's slower rate is the price of a guarantee: the non-flat-top Parzen
version is positive semi-definite and cannot return a negative variance.

**One implementation detail worth more than it looks.** The textbook two-scale
estimator averages realized variances over $K$ subsampled grids. If subgrid
$\ell$ is taken as $\{Y_{t_\ell}, Y_{t_{\ell+K}}, \dots\}$ it covers only
$[\ell\Delta, T - (K-1-\ell)\Delta]$, so its realized variance estimates a
*fraction* of the day. The resulting bias is $-(K-1)/n$, which at the recommended
$K = n^{2/3}$ is **−3.6%** on a day of one-second returns — larger than the
estimator's own error, and entirely avoidable by anchoring each subgrid at the
first and last observation. Measured with no noise at all: −3.6% before, within
one standard error of zero after.

### 6.1 The tick grid as an observation, not a nuisance

Robert and Rosenbaum's uncertainty-zones model says something sharper than
"discreteness is noise". The traded price moves from $\alpha k$ to
$\alpha(k\pm1)$ only once the efficient price has crossed a barrier
$(\tfrac12+\eta)\alpha$ away, so **just after a price change the efficient price
is known exactly** — it is sitting on that barrier. Two consequences, both
derived from a two-barrier optional-stopping argument rather than assumed:

$$\hat\eta = \frac{N_c}{2N_a}, \qquad
\hat X = P - d\,\alpha\Big(\tfrac12-\eta\Big), \qquad
\frac{RV_{\text{grid}}}{IV} \simeq \frac{1}{2\eta}.$$

The middle identity gives an integrated-variance estimator with the noise
**removed** rather than averaged away, built from the traded price on the tick
grid alone. The third is a free consistency check, because both sides are
separately measurable.

| symbol | pre-averaged, from quotes (%/day) | uncertainty zones, from trades (%/day) | ratio | eta | measured RV_grid / IV | predicted 1 / (2 eta) |
|---|---|---|---|---|---|---|
| **CSCO** | 0.945 | 0.999 | 1.056 | 0.706 | 0.730 | 0.713 |
| **INTC** | 1.073 | 1.102 | 1.027 | 0.709 | 0.733 | 0.711 |
| **MU** | 1.750 | 1.757 | 1.004 | 0.750 | 0.710 | 0.671 |
| **SIRI** | 1.214 | 1.178 | 0.971 | 0.519 | 1.138 | 1.105 |
| **MSFT** | 0.971 | 0.971 | 1.000 | 0.716 | 0.762 | 0.701 |
| **AAPL** | 1.176 | 1.212 | 1.031 | 0.870 | 0.736 | 0.588 |
| **NFLX** | 1.561 | 1.666 | 1.067 | 1.761 | 0.758 | 0.347 |
| **TSLA** | 2.036 | 2.138 | 1.050 | 0.981 | 0.876 | 0.536 |
| **REGN** | 1.335 | 1.630 | 1.221 | 4.395 | 0.691 | 0.264 |
| **ISRG** | 1.146 | 1.299 | 1.134 | 3.025 | 0.815 | 0.276 |
| **AMZN** | 1.068 | 1.043 | 0.976 | 1.275 | 0.953 | 0.513 |
| **GOOG** | 0.983 | 0.956 | 0.973 | 0.672 | 0.990 | 0.764 |

**The headline of the chapter.** An estimator that uses only the traded price on
the tick grid, and one built from the quotes by pre-averaging, share no data and
no assumption beyond the model — and they agree. On the five names whose spread
is pinned at one tick the agreement is within **5.6%**; across all twelve the
ratio stays between 0.97 and 1.22.

**The optional-stopping identity holds where the model applies and fails where it
does not.** The gap between the measured $RV_{\text{grid}}/\widehat{IV}$ and the
predicted $1/(2\hat\eta)$ is 0.017–0.061 on the five large-tick names and
0.23–0.54 on the six wide-spread ones. The regime boundary is visible in the data before any of the
volatility numbers are compared.

**A result that contradicts the model's own range.** The estimated $\eta$ is
above $1/2$ on every symbol — outside the interval the model was written for,
where a reversal must travel further than a continuation, so the printed price
trends. Two things push it up on a Nasdaq tape and both are real. The venue
prints only part of the consolidated volume, so the price moves between the
trades it does print. And the estimator counts one-tick changes, which stop being
the relevant unit once the spread is several ticks wide: $\eta$ degrades across
the panel in exactly that order, and the identity above fails at the same place.
It is reported as measured rather than truncated into range.

## 7. Chapter 04 — Prices formed by queues emptying

The queue-reactive model of Huang, Lehalle and Rosenbaum contains **no price
process**. The book is a Markov jump process whose event rates depend on the
queue sizes; the price is a consequence — when a queue at the best empties, the
best price on that side moves a tick. Volatility is an output.

Estimation needs no optimiser: for a Markov jump process the maximum-likelihood
intensity is $\hat\lambda_e(q) = N_e(q)/T(q)$, the count over the time spent in
the state, and both counters are accumulated in the pass that rebuilds the book.

**The estimated shapes are the ones the model predicts.** Cancellations rise
roughly in proportion to the queue — more resting size, more of it to pull.
Market orders **fall sharply** with the queue: takers arrive when the queue in
front of them is thin, which is the effect the model is named for. Limit orders
are close to flat with a lift at the very short queues.

### 7.1 A finding that contradicted the plan

The textbook version simulates the queue as a birth-death chain that moves by one
average event at a time, whose invariant law is the product
$\pi(q) \propto \prod_{j \le q}\lambda^\uparrow(j-1)/\lambda^\downarrow(j)$. On
Nasdaq equities that fails, and not by a little. Order sizes span a decade, so an
event does not move the queue by one unit, the $\pm1$ assumption is false, and
the product formula puts its mass where the data has none. For INTC the predicted
mean time between price changes came out at **3 617 seconds** against a measured
1.6.

Drawing event sizes from the **measured** size distribution instead fixes it: the
same intensities then give **1.50 seconds** against 1.6 measured. The correction
is not a tuning parameter — it is one fewer assumption.

| symbol | realized (%/day) | queue-reactive, half tick (%/day) | ratio | queue-reactive, full tick | mean holding (s) | best queue (AES) |
|---|---|---|---|---|---|---|
| **CSCO** | 0.945 | 0.759 | 0.803 | 1.518 | 4.62 | 8.3 |
| **INTC** | 1.073 | 0.950 | 0.886 | 1.900 | 3.14 | 7.8 |
| **MU** | 1.750 | 1.311 | 0.749 | 2.621 | 1.66 | 4.8 |
| **SIRI** | 1.214 | 1.638 | 1.350 | 3.276 | 71.12 | 23.2 |
| **MSFT** | 0.971 | 0.507 | 0.523 | 1.015 | 1.31 | 3.9 |
| **AAPL** | 1.176 | 0.467 | 0.397 | 0.933 | 0.65 | 1.8 |
| **NFLX** | 1.561 | 0.181 | 0.116 | 0.362 | 1.90 | 1.2 |
| **TSLA** | 2.036 | 0.199 | 0.098 | 0.399 | 1.57 | 1.5 |
| **REGN** | 1.335 | 0.086 | 0.064 | 0.171 | 7.29 | 0.7 |
| **ISRG** | 1.146 | 0.061 | 0.054 | 0.123 | 5.66 | 0.8 |
| **AMZN** | 1.068 | 0.035 | 0.033 | 0.070 | 1.56 | 0.9 |
| **GOOG** | 0.983 | 0.049 | 0.050 | 0.098 | 1.64 | 1.1 |

**The price-formation statement.** Nothing in the estimation saw a price. The
intensities were measured against queue sizes, the sizes against the order flow,
and the volatility came out. On the names whose spread is pinned at one tick —
the regime the model is for — it gives between 0.52 and 1.35 times the
realized volatility once the mid is credited with the half-tick it actually moves
when one side depletes --- the right order of magnitude from mechanics alone,
and not more than that. On the wide-spread names the mechanism does not apply
at all and the model under-predicts by a factor of 9 to 30, which is the correct
answer: a fifty-tick spread does not move because a queue emptied.

Both conventions are reported because which one applies is an empirical question
about the spread, not something to settle by choosing. The model shifts the whole
book by a tick; a real book usually moves one side and the mid with it by half.

### 7.2 The by-product: fill probability by queue position

Because the feed is order by order, every order that joined the touch can be
followed: how much size was resting ahead of it when it arrived, and whether it
ever traded. The resulting curve — the probability of a fill as a function of
queue position — is the input chapter 05 needs, and it cannot be measured from
any aggregated feed.

## 8. Chapter 05 — Market making: a closed form, and a learner that has to find it

A maker posts $\delta^b$ and $\delta^a$ around the mid, is filled at
$\Lambda(\delta) = Ae^{-k\delta}$, and carries whatever inventory that leaves it
with. Under a running penalty $\phi q^2$ the HJB equation has an explicit inner
maximisation, $\delta^\star = 1/k - (\theta_{q\pm1}-\theta_q)$, and the
substitution $v_q = e^{k\theta_q}$ **linearises** it. The long-horizon quotes are
read off the principal eigenvector of a tridiagonal matrix, and the maker's
ergodic gain is $\lambda_{\max}/k$.

Three independent routes to the same policy, which is the point:

| route | what it checks |
|---|---|
| principal eigenvector (continuous time) | the algebra |
| relative value iteration on the discretised MDP | that the continuous-time limit is right — agreement to within one action-grid step at $dt = 0.01$ |
| tabular Q-learning from simulated experience | that a learner given no model recovers the optimum — agreement to within **one grid step at every inventory level**, and 98.6% of the optimal reward against 37% for a flat quoter |

The reinforcement learning here is not trying to beat the closed form. It is
being checked against it, because an agent that cannot recover a policy we can
write down has not learned anything, and nothing it produces where we *cannot*
write the answer down should be believed.

### 8.1 Calibration on the panel

$A$ and $k$ are measured, not assumed: for each distance $\delta$ from the mid,
count the market orders of the day that reached at least that far and divide by
the session length — exactly the intensity a limit order posted there would have
faced.

| symbol | A (fills/s) | k (per $) | fit R2 | GLFT half-spread (ticks) | effective (ticks) | quoted (ticks) |
|---|---|---|---|---|---|---|
| **CSCO** | 0.358 | 133.24 | 0.361 | 6.215 | 0.470 | 0.529 |
| **INTC** | 0.638 | 216.19 | 0.432 | 7.550 | 0.481 | 0.533 |
| **MU** | 1.390 | 600.76 | 0.775 | 0.175 | 0.478 | 0.536 |
| **SIRI** | 0.047 | 43.63 | 0.153 | 2.570 | 0.460 | 0.510 |
| **MSFT** | 1.799 | 387.92 | 0.877 | 0.299 | 0.551 | 0.604 |
| **AAPL** | 2.693 | 207.38 | 0.957 | 0.549 | 0.917 | 1.001 |
| **NFLX** | 0.954 | 36.56 | 0.990 | 2.894 | 4.262 | 4.889 |
| **TSLA** | 1.790 | 34.02 | 0.991 | 3.711 | 5.212 | 6.220 |
| **REGN** | 0.174 | 15.68 | 0.982 | 6.628 | 8.652 | 9.861 |
| **ISRG** | 0.208 | 11.65 | 0.980 | 9.453 | 13.008 | 14.837 |
| **AMZN** | 1.275 | 10.62 | 0.977 | 9.991 | 16.783 | 19.626 |
| **GOOG** | 0.551 | 9.94 | 0.969 | 10.368 | 17.487 | 20.103 |

The fit is only credible where the trades spread over many prices. On the
seven names where $R^2$ exceeds 0.9 the model's half-spread comes out at 0.59 to
0.77 of the effective half-spread actually paid, median **0.68**: real makers quote
wider than the risk-neutral optimum, which is what a model without adverse
selection should under-predict. On the large-tick names $R^2$ falls to 0.15–0.43 — every
trade is at the same distance, so the exponential has nothing to bite on, and the
fitted $k$ says more about the tick than about liquidity. That is a limitation of
the *asset*, not of the estimator, and it is why the large-tick names are studied
with queues in chapter 04 instead.

## 9. Chapter 06 — Make-take fees as a principal-agent problem

An exchange cannot quote. It can only change what quoting pays and then live with
what the maker does next, which makes fee design a Stackelberg problem. With a
rebate $z$ per fill the agent captures $\delta + z$, so $\delta^\star = 1/k - z -
\Delta_q$ and the linear system's coupling is scaled by $e^{kz}$. Setting $z=0$
reproduces chapter 05 exactly — the check the chapter opens with — and the
ergodic gain from the eigenvalue agrees with value iteration to 0.1%.

**What a rebate buys, exactly.** Buying at inventory $q$ and selling back from
$q+1$ captures

$$\delta^b(q) + \delta^a(q+1) = \frac2k - 2z \quad \text{for every } q,$$

with the inventory terms cancelling. The whole rebate reaches the price of a
round trip whatever the maker's risk aversion. What the skew decides is *when* it
earns it: individual quotes move by amounts that vary across the inventory range,
so a flat rebate **tilts** the schedule instead of translating it. With no
inventory penalty the tilt vanishes and the schedule simply shifts.

**The optimal rebate.** The exchange charges takers $c$ and pays the maker $z$,
earning $(c-z)$ per trade at whatever rate the maker's new quotes produce. Since
the fill rate behaves like $e^{kz}$, the first-order condition gives

$$z^\star = c - \frac1k,$$

independent of the maker's risk aversion, its inventory limit, and the
volatility. The numerical optimum — which does carry the inventory law — sits
within 0.03 of that on the grid tested.

Two warnings fall out of the same first-order condition rather than out of
anecdote. A rebate large enough makes the maker quote **through** the mid: it is
paid more to trade than the spread is worth, and will lose on the price to
collect the rebate. And the participation constraint runs the other way — an
exchange that pays too little loses its maker, and the optimum becomes a corner.
Give the principal a regulator's taste for market quality instead of a
shareholder's and the optimal rebate rises monotonically: the exchange is being
paid to buy a tighter market.

---

## 10. Findings that contradicted the plan

Documented rather than smoothed over, because being wrong reproducibly is most of
what a repository like this is for.

**Match numbers do not group a sweep.** One aggressive order shows up as several
prints, and the obvious way to reassemble it is the match number. Nasdaq gives
every print its own: 89 735 executions on AAPL for 2019-01-30, 89 735 distinct
match numbers. What does group them is the timestamp — every execution an
incoming order causes carries the same nanosecond. Grouping by (timestamp,
aggressor side) turns 89 735 prints into 65 315 market orders with a mean of 1.55
prints each and a maximum of 679.

**The queue-reactive model needs real order sizes.** Covered in §7.1: the
textbook $\pm1$ birth-death chain misses the rate of price changes by three
orders of magnitude on INTC. Drawing sizes from the measured distribution brings
it to within 8%.

**The two-scale estimator's subgrids must span the day.** A −3.6% bias at the
recommended $K$, larger than the estimator's own error, from subgrids that cover
$[\ell\Delta, T-(K-1-\ell)\Delta]$ instead of $[0,T]$. Visible with the noise
switched off entirely, which is how it was found.

**The noise-variance estimator needs the signal removed.** $RV_n/(2n)$ carries a
bias of exactly $IV/(2n)$, and on a day of one-second returns with a
one-basis-point noise that bias nearly doubles the estimate: 1.85e-8 against a
true 1.0e-8 in the Monte Carlo. Subtracting a first-pass $IV$ gives 1.002e-8.

**Sampling the mid at one second is already too coarse.** The bid-ask bounce
lives below the second on these names, and above it the *persistence* the Hawkes
chapter measures takes over. Sampled at one second the mid of every symbol was
already past the minimum of its signature plot, so the estimated noise variance
came back zero on all 84 symbol-days and the optimal sampling frequency came back
infinite — all true, and all useless. At 100 ms the noise estimate is positive on
54 of 84, and the implied optimal sampling interval has a median of 2.2 seconds.
The two regimes are visible directly in the panel: the large-tick names have more
realized variance at 0.1 s than at 10 s, and the small-tick names have less.

**The uncertainty-zone parameter comes out above its admissible range.** Covered
in §6.1. Reported as measured, with the two mechanisms that produce it.

**A rebate does not translate the quoting schedule.** The obvious guess — every
quote moves inward by $z$ — is wrong, and the test that asserted it failed. What
is exactly true is the round-trip identity of §9, which is a better statement
anyway.

## 11. Limits

- **One venue.** Nasdaq's own book is a fraction of the consolidated tape for
  these names. That is what pushes $\eta$ above its range, and it caps how much
  can be said about any quantity defined on the national market.
- **Seven days.** Enough to see that the endogeneity ratio is stable to about
  0.045 within a symbol; not enough to say anything about regimes.
- **No adverse selection anywhere.** The market-making and fee models have
  inventory risk but no informed flow, which is the first-order reason a real
  maker quotes wider than $1/k$ — and is visible in the 2/3 ratio of §8.1.
- **No competition between venues** in the fee model, which is the first thing a
  real fee schedule is designed against. The single-exchange answer
  $z^\star = c - 1/k$ is a benchmark to reason from, not a recommendation.
- **The Hawkes kernel is a bank of exponentials**, rejected by its own
  goodness-of-fit test on every symbol-day. A power-law kernel is the standard
  answer and is not implemented here.
- **The queue-reactive model assumes a one-tick spread**, so half the panel is
  outside it by construction. The chapter says where, rather than reporting the
  numbers as if they meant something everywhere.

## 12. Reproducing it

```bash
make setup        # install the package and the test dependencies
make test         # 63 tests, no network
make data         # stream the seven ITCH days   (~90 min, ~31 GB transferred)
make study        # rebuild results/ from the extracted messages   (~15 min)
make build        # regenerate the notebooks from source
make notebooks    # execute them all from a clean kernel
```

Only `make data` touches the network. Everything else — including every notebook
— runs from the committed measurements in `results/`.
