r"""Two models of the same probability, and the metrics that separate them.

The logistic regression is written out here rather than imported.  It is the
interpretable half of the comparison -- a single coefficient on queue imbalance
is the number the whole chapter is about -- and a reader should be able to see
that nothing is happening inside it beyond a concave likelihood and its
gradient.  The non-parametric half is gradient boosting from scikit-learn, where
the point is precisely to let something flexible find whatever the simple model
missed.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class Logistic:
    r"""Ridge-penalised logistic regression, fitted by L-BFGS on the exact gradient.

    .. math::
        \ell(\beta) = \sum_i \big[y_i x_i^\top\beta - \log(1+e^{x_i^\top\beta})\big]
        - \tfrac{\lambda}{2}\|\beta\|^2 ,

    concave in :math:`\beta`, so the fit is a global maximum.  Features are
    standardised internally, which is what makes the coefficients comparable to
    each other and the penalty meaningful.
    """

    def __init__(self, penalty: float = 1.0):
        self.penalty = float(penalty)
        self.mean_ = None
        self.scale_ = None
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Z = (X - self.mean_) / self.scale_
        n, d = Z.shape

        def objective(theta):
            beta, bias = theta[:d], theta[d]
            eta = Z @ beta + bias
            # log(1 + e^eta), stable for large |eta|
            softplus = np.logaddexp(0.0, eta)
            loss = float(np.sum(softplus - y * eta)) + 0.5 * self.penalty * float(beta @ beta)
            p = 1.0 / (1.0 + np.exp(-eta))
            grad_beta = Z.T @ (p - y) + self.penalty * beta
            grad_bias = float(np.sum(p - y))
            return loss / n, np.concatenate([grad_beta, [grad_bias]]) / n

        result = minimize(objective, np.zeros(d + 1), jac=True, method="L-BFGS-B")
        self.coef_ = result.x[:d]
        self.intercept_ = float(result.x[d])
        self.converged_ = bool(result.success)
        return self

    def decision(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return Z @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-self.decision(X)))

    def coefficients(self, names):
        """Standardised coefficients, largest first -- the readable output."""
        order = np.argsort(-np.abs(self.coef_))
        return [(names[i], float(self.coef_[i])) for i in order]


def fit_boosted(X, y, max_iter: int = 200, seed: int = 0, **kwargs):
    """Gradient-boosted trees, the flexible half of the comparison."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed, **kwargs,
    )
    model.fit(np.asarray(X), np.asarray(y))
    return model


def evaluate(y_true, p, n_bins: int = 12) -> dict:
    """AUC, Brier, log loss, accuracy, and a calibration curve.

    Brier and log loss are the ones that matter here: a score can rank states
    correctly and still be badly wrong about the probability, and it is the
    probability the market maker of chapter 08 needs.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    ok = np.isfinite(p)
    y_true, p = y_true[ok], p[ok]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p) - 1, 0, n_bins - 1)
    predicted = np.full(n_bins, np.nan)
    observed = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins)
    for k in range(n_bins):
        m = idx == k
        counts[k] = m.sum()
        if m.sum() > 30:
            predicted[k] = p[m].mean()
            observed[k] = y_true[m].mean()
    return {
        "n": int(y_true.size),
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": float(np.mean((p - y_true) ** 2)),
        "log_loss": float(log_loss(y_true, p)),
        "accuracy": float(np.mean((p > 0.5) == (y_true > 0.5))),
        "base_rate": float(y_true.mean()),
        "calibration_predicted": predicted,
        "calibration_observed": observed,
        "calibration_counts": counts,
    }


def surface_from_scores(bucket_bid, bucket_ask, p, grid: int, min_count: int = 30):
    """Average a fitted probability onto the joint queue grid.

    This is what makes the fitted answer and the analytic one comparable: both
    end up as a ``(grid, grid)`` array indexed by the same two queue buckets.
    """
    total = np.zeros((grid, grid))
    count = np.zeros((grid, grid))
    np.add.at(total, (bucket_bid, bucket_ask), p)
    np.add.at(count, (bucket_bid, bucket_ask), 1.0)
    out = np.full((grid, grid), np.nan)
    enough = count >= min_count
    out[enough] = total[enough] / count[enough]
    return out, count
