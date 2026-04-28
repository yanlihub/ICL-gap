"""
Script: corruption.py
"""
from __future__ import annotations

import numpy as np


def corrupt_marginal(
    X: np.ndarray,
    y: np.ndarray,
    corruption_level: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corrupt marginal distributions.
    With probability `corruption_level`, replace each feature value with a
    uniform draw in [col_min, col_max]. Preserves sample count and label.
    """
    rng = np.random.default_rng(seed)
    X_c = X.copy()
    mask = rng.random(X_c.shape) < corruption_level
    for j in range(X_c.shape[1]):
        n_corrupt = mask[:, j].sum()
        if n_corrupt > 0:
            X_c[mask[:, j], j] = rng.uniform(
                X_c[:, j].min(), X_c[:, j].max(), size=n_corrupt
            )
    return X_c, y.copy()


def corrupt_feature_correlation(
    X: np.ndarray,
    y: np.ndarray,
    fraction: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Destroy feature-feature correlations.
    Permute `fraction` of columns independently.
    Preserves marginal distributions and label structure.
    """
    rng = np.random.default_rng(seed)
    X_c = X.copy()
    n_corrupt = max(1, int(fraction * X_c.shape[1]))
    cols = rng.choice(X_c.shape[1], size=n_corrupt, replace=False)
    for j in cols:
        X_c[:, j] = rng.permutation(X_c[:, j])
    return X_c, y.copy()


def corrupt_label_conditional(
    X: np.ndarray,
    y: np.ndarray,
    fraction: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Destroy P(X|Y) structure.
    Randomly reassign labels to `fraction` of samples.
    Preserves label marginal distribution and feature marginals.
    """
    rng = np.random.default_rng(seed)
    y_c = y.copy()
    n_corrupt = max(1, int(fraction * len(y_c)))
    idx = rng.choice(len(y_c), size=n_corrupt, replace=False)
    classes = np.unique(y)
    y_c[idx] = rng.choice(classes, size=n_corrupt)
    return X.copy(), y_c


def corrupt_label_noise(
    X: np.ndarray,
    y: np.ndarray,
    noise_rate: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standard symmetric label noise.
    Flip labels of `noise_rate` fraction of samples to a *different* class.
    """
    rng = np.random.default_rng(seed)
    y_c = y.copy()
    classes = np.unique(y)
    n_corrupt = max(1, int(noise_rate * len(y_c)))
    idx = rng.choice(len(y_c), size=n_corrupt, replace=False)
    for i in idx:
        other = [c for c in classes if c != y_c[i]]
        if other:
            y_c[i] = rng.choice(other)
    return X.copy(), y_c


def corrupt_class_balance(
    X: np.ndarray,
    y: np.ndarray,
    target_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample majority class to reach target_ratio = minority/majority.
    target_ratio=1.0 → perfectly balanced; <1.0 → increasingly imbalanced.
    Returns a subset (fewer samples than input when target_ratio < natural ratio).
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    max_count = counts.max()

    target_majority = int(min_count / target_ratio) if target_ratio > 0 else max_count
    target_majority = min(target_majority, max_count)

    idx_keep = []
    for cls, count in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        if count == min_count:
            idx_keep.append(cls_idx)
        else:
            n_keep = min(target_majority, count)
            idx_keep.append(rng.choice(cls_idx, size=n_keep, replace=False))

    idx_all = np.concatenate(idx_keep)
    rng.shuffle(idx_all)
    return X[idx_all].copy(), y[idx_all].copy()


def corrupt_outliers(
    X: np.ndarray,
    y: np.ndarray,
    outlier_fraction: float = 0.1,
    n_sigma: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject outliers: replace `outlier_fraction` of samples with extreme values.
    Each corrupted sample has all features set to mean ± n_sigma * std
    (direction chosen randomly per feature).
    """
    rng = np.random.default_rng(seed)
    X_c = X.copy()
    n_corrupt = max(1, int(outlier_fraction * len(X_c)))
    idx = rng.choice(len(X_c), size=n_corrupt, replace=False)
    col_mean = X_c.mean(axis=0)
    col_std = X_c.std(axis=0) + 1e-8
    for i in idx:
        signs = rng.choice([-1.0, 1.0], size=X_c.shape[1])
        X_c[i] = col_mean + signs * n_sigma * col_std
    return X_c, y.copy()


def corrupt_shuffle_all(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extreme corruption: independently permute every feature column and labels.
    Equivalent to L0 random baseline. Used as lower bound for ICL-Gap.
    """
    rng = np.random.default_rng(seed)
    X_c = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    y_c = rng.permutation(y)
    return X_c.astype(X.dtype), y_c


# Registry for sweep experiments (EXP-5.1)
CORRUPTION_REGISTRY = {
    "marginal": corrupt_marginal,
    "feature_correlation": corrupt_feature_correlation,
    "label_conditional": corrupt_label_conditional,
    "label_noise": corrupt_label_noise,
    "class_balance": corrupt_class_balance,
    "outliers": corrupt_outliers,
    "shuffle_all": corrupt_shuffle_all,
}
