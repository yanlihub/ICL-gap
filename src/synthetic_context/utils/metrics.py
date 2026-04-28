"""
Script: metrics.py
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy.stats import ks_2samp


# ─── Column-type detection ────────────────────────────────────────────────────

def identify_num_cat_columns(
    X: np.ndarray,
    cat_unique_threshold: int = 20,
) -> tuple[list[int], list[int]]:
    """Split feature indices into numerical (high-cardinality) and
    categorical-like (low-cardinality, i.e. ordinal-encoded discrete columns).

    Parameters
    ----------
    cat_unique_threshold:
        Columns with n_unique <= threshold are treated as categorical.
        Default 20 works well after TableVectorizer ordinal encoding.
    """
    num_cols: list[int] = []
    cat_cols: list[int] = []
    for j in range(X.shape[1]):
        n_unique = int(np.unique(X[:, j]).size)
        if n_unique <= cat_unique_threshold:
            cat_cols.append(j)
        else:
            num_cols.append(j)
    return num_cols, cat_cols


# ─── Marginal metrics ─────────────────────────────────────────────────────────

def compute_ks(
    X_real: np.ndarray,
    X_syn: np.ndarray,
    cat_unique_threshold: int = 20,
) -> float:
    """Mean two-sample Kolmogorov-Smirnov statistic across ALL columns.

    Measures marginal distribution fidelity.  KS is valid for both continuous
    and ordinal-encoded features (it compares empirical CDFs, which are
    well-defined for any numerical data).

    After ordinal encoding by TableVectorizer, all features are integers or
    floats — KS is applicable to all of them.  The cat_unique_threshold
    parameter is accepted for API compatibility but no longer used for
    column filtering.

    Implementation: scipy.stats.ks_2samp(..).statistic (Hodges 1958).

    Range     : [0, 1]   (0 = identical CDFs, 1 = maximally separated)
    Direction : lower is better
    Features  : ALL columns (numerical + ordinal-encoded categorical)
    Returns NaN if X has no columns.
    """
    p = X_real.shape[1]
    if p == 0:
        return float("nan")
    stats = [
        ks_2samp(X_real[:, j], X_syn[:, j]).statistic
        for j in range(p)
    ]
    return float(np.mean(stats))


def compute_jsd(
    X_real: np.ndarray,
    X_syn: np.ndarray,
    cat_unique_threshold: int = 20,
) -> float:
    """Mean Jensen-Shannon Distance (JSD) across categorical columns.

    Measures marginal distribution fidelity for discrete / low-cardinality
    features.  Numerical columns are excluded — use compute_ks() for those.

    Note: scipy.spatial.distance.jensenshannon returns the **JS distance**
    (= sqrt of JS divergence), not the divergence itself.  Both are [0, 1]
    when computed with log base 2.  Laplace smoothing (eps=1e-10) is applied
    to avoid log(0) on unseen synthetic categories.

    Implementation: scipy.spatial.distance.jensenshannon (Endres & Schindelin 2003).

    Range     : [0, 1]   (0 = identical frequency distributions, 1 = disjoint)
    Direction : lower is better
    Features  : categorical columns only (low-cardinality, n_unique ≤ threshold)
    Returns NaN if no categorical columns are detected.
    """
    from scipy.spatial.distance import jensenshannon

    _, cat_cols = identify_num_cat_columns(X_real, cat_unique_threshold)
    if not cat_cols:
        return float("nan")

    jsd_vals: list[float] = []
    for j in cat_cols:
        r_col = X_real[:, j]
        s_col = X_syn[:, j]
        all_vals = np.union1d(np.unique(r_col), np.unique(s_col))
        r_freq = np.array([(r_col == v).sum() for v in all_vals], dtype=float)
        s_freq = np.array([(s_col == v).sum() for v in all_vals], dtype=float)
        # Normalize to probability, add Laplace smoothing
        eps = 1e-10
        r_prob = (r_freq + eps) / (r_freq.sum() + eps * len(all_vals))
        s_prob = (s_freq + eps) / (s_freq.sum() + eps * len(all_vals))
        val = float(jensenshannon(r_prob, s_prob))
        if np.isfinite(val):
            jsd_vals.append(val)

    return float(np.mean(jsd_vals)) if jsd_vals else float("nan")


def compute_wd_1d(
    X_real: np.ndarray,
    X_syn: np.ndarray,
) -> float:
    """Mean per-column 1-D Wasserstein (Earth Mover's) distance.

    Each feature column is evaluated independently after standardization
    (z-score using real-data mean/std), so scale differences do not bias
    the average.  All features — including ordinal-encoded categorical
    columns — are included.

    Implementation: scipy.stats.wasserstein_distance per column (Villani 2008).

    Range     : [0, ∞)   (0 = identical marginals per column, after std'izing)
    Direction : lower is better
    Features  : ALL columns (numerical + ordinal-encoded categorical)

    Note: this is a *marginal* metric — it does not capture inter-feature
    correlations.  Use compute_wd_sliced() for joint structure.
    """
    from scipy.stats import wasserstein_distance

    std = X_real.std(axis=0)
    std[std == 0] = 1.0          # avoid division by zero for constant columns
    mean = X_real.mean(axis=0)
    X_r = (X_real - mean) / std
    X_s = (X_syn  - mean) / std

    wd_vals = [
        wasserstein_distance(X_r[:, j], X_s[:, j])
        for j in range(X_r.shape[1])
    ]
    return float(np.mean(wd_vals))


# ─── Joint-distribution metrics ───────────────────────────────────────────────

def compute_wd_sliced(
    X_real: np.ndarray,
    X_syn: np.ndarray,
    n_projections: int = 50,
    seed: int = 42,
    max_samples: int = 5000,
) -> float:
    """Sliced Wasserstein Distance (SWD) — joint distribution approximation.

    Approximates the p-dimensional Wasserstein distance by averaging 1-D WD
    over `n_projections` random unit directions.  Captures *joint* distribution
    differences (unlike WD-1D which is purely marginal).

    Both datasets are subsampled to `max_samples` and z-score standardized
    using real-data statistics before comparison.  All features — including
    ordinal-encoded categorical columns — are included.

    Implementation: ot.sliced_wasserstein_distance (Bonneel et al. 2015).
    Requires: pip install POT

    Range     : [0, ∞)   (0 = identical joint distributions)
    Direction : lower is better
    Features  : ALL columns (numerical + ordinal-encoded categorical)
    Returns NaN if POT is not installed.
    """
    try:
        import ot  # type: ignore[import]
    except ImportError:
        warnings.warn("POT not installed; wd_sliced will be NaN. Run: pip install POT")
        return float("nan")

    rng = np.random.default_rng(seed)
    n = min(max_samples, len(X_real), len(X_syn))
    idx_r = rng.choice(len(X_real), n, replace=False)
    idx_s = rng.choice(len(X_syn),  n, replace=False)

    std = X_real.std(axis=0)
    std[std == 0] = 1.0
    mean = X_real.mean(axis=0)
    X_r = ((X_real[idx_r] - mean) / std).astype(np.float64)
    X_s = ((X_syn[idx_s]  - mean) / std).astype(np.float64)

    try:
        swd = ot.sliced_wasserstein_distance(
            X_r, X_s, n_projections=n_projections, seed=int(seed)
        )
        return float(swd)
    except Exception as exc:
        warnings.warn(f"wd_sliced computation failed: {exc}")
        return float("nan")


def compute_mmd(
    X_real: np.ndarray,
    X_syn: np.ndarray,
    seed: int = 42,
    max_samples: int = 2000,
    gamma: float = 1.0,
) -> float:
    """Maximum Mean Discrepancy with RBF kernel.

    Measures joint distribution discrepancy in a Reproducing Kernel Hilbert
    Space (RKHS) via the RBF kernel K(x,y) = exp(-gamma * ||x-y||²).

    Both datasets are subsampled to `max_samples` for tractability, then
    z-score standardized using real-data statistics.  All features — including
    ordinal-encoded categorical columns — are included.

    Estimator: biased (includes diagonal terms in K_XX and K_YY), which is
    consistent and computationally simpler than the unbiased variant.
    Fixed gamma=1.0 after standardization (equivalent to median-heuristic
    when features are unit-variance).

    Implementation: closed-form RBF MMD (Gretton et al. 2012, JMLR).

    Range     : [0, ∞)   (0 = identical distributions in RKHS; clipped at 0
                           for numerical precision)
    Direction : lower is better
    Features  : ALL columns (numerical + ordinal-encoded categorical)
    """
    rng = np.random.default_rng(seed)
    n = min(max_samples, len(X_real), len(X_syn))
    idx_r = rng.choice(len(X_real), n, replace=False)
    idx_s = rng.choice(len(X_syn),  n, replace=False)

    # Standardize
    std = X_real.std(axis=0)
    std[std == 0] = 1.0
    mean = X_real.mean(axis=0)
    X_r = (X_real[idx_r] - mean) / std
    X_s = (X_syn[idx_s]  - mean) / std

    # RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)
    XX = np.dot(X_r, X_r.T)
    YY = np.dot(X_s, X_s.T)
    XY = np.dot(X_r, X_s.T)
    X_sq = np.diag(XX)
    Y_sq = np.diag(YY)
    K_XX = np.exp(-gamma * (X_sq[:, None] + X_sq[None, :] - 2 * XX))
    K_YY = np.exp(-gamma * (Y_sq[:, None] + Y_sq[None, :] - 2 * YY))
    K_XY = np.exp(-gamma * (X_sq[:, None] + Y_sq[None, :] - 2 * XY))

    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return float(max(0.0, mmd))   # numerical precision guard


# ─── Correlation-structure metric ─────────────────────────────────────────────

def compute_nfn(
    X_real: np.ndarray,
    X_syn: np.ndarray,
) -> float:
    """Normalized Frobenius Norm of Spearman correlation matrix difference (NFN).

    Measures how well the synthetic data preserves pairwise feature correlations.
    Uses Spearman (rank) correlation so that ordinal-encoded categorical features
    are treated consistently with numerical features.

    Formula:
        NFN = ||lower_tri(C_real − C_syn)||_F  /  sqrt(p*(p-1)/2)

    where C is the p×p Spearman correlation matrix, lower triangle (diagonal
    excluded), and the denominator normalizes by the number of unique feature
    pairs so NFN is comparable across datasets with different dimensionality.

    Range     : [0, 2]
                  0  = identical correlation structure
                  2  = maximally anti-correlated (each pairwise Spearman
                       coeff differs by 2, its maximum possible difference)
    Direction : lower is better
    Features  : ALL columns (Spearman rank transform makes it scale-invariant)
    Returns NaN for p < 2 (no pairs to compare).
    """
    p = X_real.shape[1]
    if p < 2:
        return float("nan")

    # Rank-transform each column, then use np.corrcoef (equivalent to Spearman)
    from scipy.stats import rankdata

    def _spearman_matrix(X: np.ndarray) -> np.ndarray:
        ranks = np.apply_along_axis(rankdata, 0, X)
        # corrcoef wants shape (p, n)
        return np.corrcoef(ranks.T)

    C_real = _spearman_matrix(X_real)
    C_syn  = _spearman_matrix(X_syn)

    diff = C_real - C_syn
    tril_mask = np.tril(np.ones((p, p), dtype=bool), k=-1)
    lower_vals = diff[tril_mask]

    n_pairs = p * (p - 1) / 2.0
    nfn = np.linalg.norm(lower_vals) / np.sqrt(n_pairs)
    return float(nfn)


# ─── Utility metric ───────────────────────────────────────────────────────────

def compute_tstr_auc(
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    classifier: str = "xgb",
) -> float:
    """Train on Synthetic, Test on Real (TSTR) macro ROC-AUC.

    Trains a downstream classifier *exclusively* on synthetic data, then
    evaluates it on the held-out real test set.  High score means the
    synthetic data encodes useful decision boundaries for real tasks.

    Supported classifiers:
      "xgb" — XGBoostClassifier (fallback: sklearn GradientBoosting if xgb absent)
      "mlp" — MLPClassifier(100, 100), max_iter=300
      "rf"  — RandomForestClassifier(100 trees)

    Implementation: standard sklearn roc_auc_score, multi-class OvR macro.

    Range     : [0, 1]   (chance ≈ 0.5 for balanced binary)
    Direction : higher is better
    Features  : ALL columns (same preprocessing as ICL evaluation)

    Parameters
    ----------
    classifier : str
        One of "xgb" (XGBoost, default), "mlp" (MLPClassifier), or "rf"
        (RandomForestClassifier).
    """
    from sklearn.metrics import roc_auc_score

    if classifier == "xgb":
        try:
            from xgboost import XGBClassifier  # type: ignore[import]
            clf = XGBClassifier(
                n_estimators=100,
                random_state=seed,
                verbosity=0,
                eval_metric="logloss",
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(n_estimators=100, random_state=seed)
    elif classifier == "mlp":
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=(100, 100),
            max_iter=300,
            random_state=seed,
        )
    elif classifier == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier '{classifier}'. Choose from: xgb, mlp, rf")

    n_classes = len(np.unique(y_test))
    clf.fit(X_syn, y_syn)
    y_proba = clf.predict_proba(X_test)

    if n_classes == 2:
        return float(roc_auc_score(y_test, y_proba[:, 1]))
    return float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))


def compute_tstr_regression(
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    regressor: str = "xgb",
) -> dict[str, float]:
    """Train on Synthetic, Test on Real (TSTR) for regression tasks.

    Trains a downstream regressor *exclusively* on synthetic data, then
    evaluates it on the held-out real test set.  Returns both R² and RMSE.

    Supported regressors:
      "xgb" — XGBRegressor (fallback: sklearn GradientBoostingRegressor)
      "mlp" — MLPRegressor(100, 100), max_iter=300
      "rf"  — RandomForestRegressor(100 trees)

    Returns
    -------
    {"r2": float, "rmse": float}
        r2   : (-∞, 1], higher is better
        rmse : [0, ∞),  lower is better
    """
    from sklearn.metrics import r2_score, root_mean_squared_error

    if regressor == "xgb":
        try:
            from xgboost import XGBRegressor  # type: ignore[import]
            reg = XGBRegressor(
                n_estimators=100,
                random_state=seed,
                verbosity=0,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            reg = GradientBoostingRegressor(n_estimators=100, random_state=seed)
    elif regressor == "mlp":
        from sklearn.neural_network import MLPRegressor
        reg = MLPRegressor(
            hidden_layer_sizes=(100, 100),
            max_iter=300,
            random_state=seed,
        )
    elif regressor == "rf":
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=100, random_state=seed)
    else:
        raise ValueError(f"Unknown regressor '{regressor}'. Choose from: xgb, mlp, rf")

    reg.fit(X_syn, y_syn)
    y_pred = reg.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
    }


# ─── Density / Coverage metric ───────────────────────────────────────────────

def compute_alpha_precision(
    X_real: np.ndarray,
    X_syn: np.ndarray,
    seed: int = 42,
    max_samples: int = 2000,
    n_steps: int = 30,
) -> dict[str, float]:
    """Alpha-precision and Beta-recall (Alaa et al., ICML 2022).

    Standalone numpy/sklearn implementation — no synthcity dependency.

    Overview
    --------
    The method sweeps over a radius parameter alpha ∈ [0, 1] and constructs
    two curves evaluated at `n_steps` evenly-spaced alpha values:

      α-precision curve : fraction of synthetic points inside the alpha-quantile
                          hypersphere of the real data distribution.
                          A perfect generator → curve lies on the diagonal.
      β-recall curve    : fraction of real points whose nearest synthetic
                          neighbor is "close enough" (within the beta-quantile
                          radius of the synthetic distribution).
                          A generator with full coverage → curve on diagonal.

    The reported scalar for each curve is NOT the raw AUC.  It is:

        delta = 1 − (L1 deviation from perfect diagonal) / norm_factor

    where norm_factor = sum(alpha_steps).  This "closeness to diagonal" score
    equals 1 for a perfect generator and approaches 0 for a degenerate one.

    Both arrays are subsampled to `max_samples` and MinMax-normalized using
    real-data statistics before evaluation (following the original paper).

    Reference: Alaa et al., "How Faithful is your Synthetic Data?
               Sample-level Metrics for Evaluating and Auditing
               Generative Models", ICML 2022.

    Range     : [0, 1]   (1 = curves lie perfectly on the diagonal)
    Direction : higher is better (for both alpha_precision and beta_recall)
    Features  : ALL columns (subsampled and MinMax-normalized)

    Returns
    -------
    {"alpha_precision": float, "beta_recall": float}
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import MinMaxScaler

    rng = np.random.default_rng(seed)
    n = min(max_samples, len(X_real), len(X_syn))
    idx_r = rng.choice(len(X_real), n, replace=False)
    idx_s = rng.choice(len(X_syn),  n, replace=False)

    scaler = MinMaxScaler().fit(X_real)
    X_r = scaler.transform(X_real[idx_r]).astype(np.float64)
    X_s = scaler.transform(X_syn[idx_s]).astype(np.float64)

    try:
        alphas = np.linspace(0, 1, n_steps)

        # Real-data center and per-point distances to center
        emb_center = X_r.mean(axis=0)
        real_to_center = np.linalg.norm(X_r - emb_center, axis=1)
        Radii = np.quantile(real_to_center, alphas)

        # Synthetic distances to real center
        synth_center = X_s.mean(axis=0)
        synth_to_center = np.linalg.norm(X_s - emb_center, axis=1)

        # For beta-recall: real→real NN (excluding self) and real→synth NN
        nbrs_real = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(X_r)
        real_to_real_dists, _ = nbrs_real.kneighbors(X_r)
        real_to_real = real_to_real_dists[:, 1]          # exclude self (col 0)

        nbrs_syn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(X_s)
        real_to_syn_dists, real_to_syn_args = nbrs_syn.kneighbors(X_r)
        real_to_syn = real_to_syn_dists[:, 0]

        # Distance of the closest-synth-to-real point to synth center
        closest_syn_to_syn_center = np.linalg.norm(
            X_s[real_to_syn_args[:, 0]] - synth_center, axis=1
        )
        closest_syn_Radii = np.quantile(closest_syn_to_syn_center, alphas)

        # Build curves
        alpha_precision_curve = []
        beta_coverage_curve = []
        for k in range(n_steps):
            # Fraction of synthetic points inside radius Radii[k]
            alpha_precision_curve.append(float(np.mean(synth_to_center <= Radii[k])))
            # Fraction of real points whose nearest synthetic is "close enough"
            beta_cov = np.mean(
                (real_to_syn <= real_to_real)
                & (closest_syn_to_syn_center[real_to_syn_args[:, 0]] <= closest_syn_Radii[k])
            )
            beta_coverage_curve.append(float(beta_cov))

        alpha_arr = np.array(alphas)
        ap_arr    = np.array(alpha_precision_curve)
        br_arr    = np.array(beta_coverage_curve)

        # Delta = 1 - (L1 deviation from perfect diagonal) / norm_factor
        norm = float(np.sum(alpha_arr))
        delta_ap = float(1.0 - np.sum(np.abs(alpha_arr - ap_arr)) / norm)
        delta_br = float(1.0 - np.sum(np.abs(alpha_arr - br_arr)) / norm)

        # Clamp to [0, 1] (floating-point rounding guard)
        delta_ap = max(0.0, min(1.0, delta_ap))
        delta_br = max(0.0, min(1.0, delta_br))

        return {"alpha_precision": delta_ap, "beta_recall": delta_br}

    except Exception as exc:
        warnings.warn(f"alpha_precision computation failed: {exc}")
        return {"alpha_precision": float("nan"), "beta_recall": float("nan")}


# ─── Unified entry point ─────────────────────────────────────────────────────

def compute_all_fidelity_metrics(
    X_real: np.ndarray,
    X_syn: np.ndarray,
    y_syn: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    seed: int = 42,
    cat_unique_threshold: int = 20,
    wd_sliced_n_projections: int = 50,
    mmd_max_samples: int = 2000,
    ap_max_samples: int = 2000,
    task_type: str = "classification",
) -> dict[str, float]:
    """Compute the full EXP-3.1 fidelity metric suite.

    All metrics are stored with prefix `fid_` in the returned dict.
    NaN is used for metrics that cannot be computed (missing dep / no columns).

    Parameters
    ----------
    X_real, X_syn : numpy arrays, shape (n, p), already ordinal-encoded, no NaNs
    y_syn         : synthetic labels — needed for TSTR
    X_test, y_test: real test set — needed for TSTR
    seed          : RNG seed for stochastic metrics (MMD subsample, SWD, AP)
    cat_unique_threshold : columns with ≤ this many unique values → categorical
    task_type     : "classification" or "regression" — routes TSTR to correct variant
    """
    out: dict[str, float] = {}

    # Marginal
    out["fid_ks"]    = _safe(compute_ks,    X_real, X_syn, cat_unique_threshold)
    out["fid_jsd"]   = _safe(compute_jsd,   X_real, X_syn, cat_unique_threshold)
    out["fid_wd_1d"] = _safe(compute_wd_1d, X_real, X_syn)

    # Joint
    out["fid_wd_sliced"] = _safe(
        compute_wd_sliced, X_real, X_syn,
        wd_sliced_n_projections, seed, 5000,
    )
    out["fid_mmd"] = _safe(compute_mmd, X_real, X_syn, seed, mmd_max_samples)

    # Correlation structure
    out["fid_nfn"] = _safe(compute_nfn, X_real, X_syn)

    # Utility — route to classification (AUC) or regression (R² + RMSE)
    _nan_tstr = {
        "fid_tstr_auc": float("nan"), "fid_tstr_mlp": float("nan"),
        "fid_tstr_rf": float("nan"),
        "fid_tstr_rmse_xgb": float("nan"), "fid_tstr_rmse_mlp": float("nan"),
        "fid_tstr_rmse_rf": float("nan"),
    }
    if y_syn is not None and X_test is not None and y_test is not None:
        if task_type == "regression":
            for reg_name, key_suffix in [("xgb", "auc"), ("mlp", "mlp"), ("rf", "rf")]:
                reg_result = _safe_dict(
                    compute_tstr_regression, X_syn, y_syn, X_test, y_test, seed, reg_name,
                    default={"r2": float("nan"), "rmse": float("nan")},
                )
                out[f"fid_tstr_{key_suffix}"] = reg_result["r2"]
                out[f"fid_tstr_rmse_{reg_name}"] = reg_result["rmse"]
        else:
            out["fid_tstr_auc"] = _safe(
                compute_tstr_auc, X_syn, y_syn, X_test, y_test, seed, "xgb"
            )
            out["fid_tstr_mlp"] = _safe(
                compute_tstr_auc, X_syn, y_syn, X_test, y_test, seed, "mlp"
            )
            out["fid_tstr_rf"] = _safe(
                compute_tstr_auc, X_syn, y_syn, X_test, y_test, seed, "rf"
            )
            out["fid_tstr_rmse_xgb"] = float("nan")
            out["fid_tstr_rmse_mlp"] = float("nan")
            out["fid_tstr_rmse_rf"]  = float("nan")
    else:
        out.update(_nan_tstr)

    # Density / Coverage
    ap = _safe_dict(
        compute_alpha_precision, X_real, X_syn, seed, ap_max_samples,
        default={"alpha_precision": float("nan"), "beta_recall": float("nan")},
    )
    out["fid_alpha_precision"] = ap["alpha_precision"]
    out["fid_beta_recall"]     = ap["beta_recall"]

    return out


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe(fn: Callable, *args, **kwargs) -> float:
    """Call fn(*args, **kwargs), return NaN on any exception."""
    try:
        result = fn(*args, **kwargs)
        return float("nan") if result is None else float(result)
    except Exception as exc:
        warnings.warn(f"{fn.__name__} failed: {exc}")
        return float("nan")


def _safe_dict(
    fn: Callable,
    *args,
    default: dict[str, float],
    **kwargs,
) -> dict[str, float]:
    """Call fn(*args, **kwargs) returning a dict; return `default` on error."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        warnings.warn(f"{fn.__name__} failed: {exc}")
        return default


# ─── Metric registry (for extensibility) ─────────────────────────────────────

#: Ordered list of fidelity metric keys produced by compute_all_fidelity_metrics.
#: Add new metric keys here when extending the suite.
FIDELITY_METRIC_KEYS: list[str] = [
    "fid_ks",
    "fid_jsd",
    "fid_wd_1d",
    "fid_wd_sliced",
    "fid_mmd",
    "fid_nfn",
    "fid_tstr_auc",
    "fid_tstr_mlp",
    "fid_tstr_rf",
    "fid_tstr_rmse_xgb",
    "fid_tstr_rmse_mlp",
    "fid_tstr_rmse_rf",
    "fid_alpha_precision",
    "fid_beta_recall",
]

#: Human-readable labels and "direction" for each metric key.
FIDELITY_METRIC_META: dict[str, dict] = {
    "fid_ks":              {"label": "KS (marginal, all)",         "better": "lower"},
    "fid_jsd":             {"label": "JSD (marginal, cat)",        "better": "lower"},
    "fid_wd_1d":           {"label": "WD-1D (marginal, all)",      "better": "lower"},
    "fid_wd_sliced":       {"label": "SWD (joint approx, all)",    "better": "lower"},
    "fid_mmd":             {"label": "MMD (joint, all)",           "better": "lower"},
    "fid_nfn":             {"label": "NFN (correlation, all)",     "better": "lower"},
    "fid_tstr_auc":        {"label": "TSTR-XGB (utility, AUC/R²)", "better": "higher"},
    "fid_tstr_mlp":        {"label": "TSTR-MLP (utility, AUC/R²)", "better": "higher"},
    "fid_tstr_rf":         {"label": "TSTR-RF (utility, AUC/R²)",  "better": "higher"},
    "fid_tstr_rmse_xgb":   {"label": "TSTR-XGB RMSE (reg only)",  "better": "lower"},
    "fid_tstr_rmse_mlp":   {"label": "TSTR-MLP RMSE (reg only)",  "better": "lower"},
    "fid_tstr_rmse_rf":    {"label": "TSTR-RF RMSE (reg only)",   "better": "lower"},
    "fid_alpha_precision": {"label": "α-Precision (density)",      "better": "higher"},
    "fid_beta_recall":     {"label": "β-Recall (diversity)",       "better": "higher"},
}


# ─── Legacy aliases (kept for backward compatibility) ─────────────────────────

def marginal_fidelity(X_real: np.ndarray, X_syn: np.ndarray) -> float:
    """Legacy: 1 - mean(KS). Higher = better. Use compute_ks() instead."""
    return 1.0 - compute_ks(X_real, X_syn, cat_unique_threshold=0)  # all cols


def compute_prdc_metrics(
    X_real: np.ndarray,
    X_syn: np.ndarray,
    nearest_k: int = 5,
) -> dict[str, float]:
    """Legacy: PRDC via the `prdc` package. Use compute_alpha_precision() instead."""
    try:
        from prdc import compute_prdc  # type: ignore[import]
    except ImportError:
        return {k: float("nan") for k in ("precision", "recall", "density", "coverage")}
    metrics = compute_prdc(X_real, X_syn, nearest_k=nearest_k)
    return {k: float(v) for k, v in metrics.items()}


# Old name kept for callers in exp1/exp5
tstr_auc = compute_tstr_auc
