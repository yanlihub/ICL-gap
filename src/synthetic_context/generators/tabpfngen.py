"""
Script: tabpfngen.py
"""
from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np

from .base import BaseGenerator

log = logging.getLogger(__name__)


def _safe_proba(p: np.ndarray) -> np.ndarray:
    """Normalize a probability vector to exactly sum to 1.0 (float64).
    Fixes floating-point precision errors from predict_proba outputs."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)          # remove any tiny negatives from fp errors
    total = p.sum()
    if total <= 0:
        return np.ones(len(p)) / len(p) # uniform fallback
    return p / total


class TabPFNGenGenerator(BaseGenerator):
    """L8: TabPFN-conditioned generative model (unsupervised extension)."""

    name = "tabpfngen"
    level = "L8"

    def __init__(
        self,
        seed: int = 42,
        t: float = 1.0,
        n_permutations: int = 1,
        max_fit_samples: int = 2000,
        y_mode: Literal["sample", "argmax"] = "sample",
    ) -> None:
        self._seed = seed
        self._t = t
        self._n_permutations = n_permutations
        self._max_fit_samples = max_fit_samples
        self._y_mode = y_mode
        self._rng = np.random.default_rng(seed)

        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._X_ctx: np.ndarray | None = None   # subsampled context for generation
        self._y_ctx: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._is_regression: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNGenGenerator":
        """Store training data. Subsamples to max_fit_samples for the feature models."""
        self._X_train = X.astype(np.float32)
        self._y_train = y.copy()
        self._classes = np.unique(y)
        self._is_regression = (np.issubdtype(y.dtype, np.floating) and
                               len(self._classes) > 20)

        n = min(self._max_fit_samples, len(X))
        if n < len(X):
            idx = self._rng.choice(len(X), n, replace=False)
            self._X_ctx = self._X_train[idx]
            self._y_ctx = self._y_train[idx]
            log.info(
                f"[tabpfngen] max_fit_samples={self._max_fit_samples} < n_train={len(X)}; "
                f"subsampling to {n} for feature models."
            )
        else:
            self._X_ctx = self._X_train
            self._y_ctx = self._y_train

        return self

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate n_samples (X_syn, y_syn) pairs.

        X_syn is sampled from TabPFN's conditional distribution over features.
        y_syn is sampled from TabPFN's predictive distribution P(y | X_syn, D_ctx).
        """
        assert self._X_ctx is not None, "Call fit() before generate()."

        try:
            import torch
            from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as e:
            raise ImportError(
                "tabpfn-extensions required for TabPFNGen. "
                "Install with: pip install tabpfn-extensions"
            ) from e

        log.info(
            f"[tabpfngen] Generating {n_samples} samples via unsupervised TabPFN "
            f"(n_ctx={len(self._X_ctx)}, n_features={self._X_ctx.shape[1]}, "
            f"t={self._t}, n_permutations={self._n_permutations})"
        )

        # ── Step 1: Generate X_syn via chain-rule conditional sampling ──────
        # Retry up to MAX_RETRIES times: generate_synthetic_data() can fail
        # stochastically with "Input X contains infinity or a value too large
        # for dtype('float32')" when internal chain-rule conditionals overflow.
        MAX_RETRIES = 5
        X_syn = None
        for attempt in range(MAX_RETRIES):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf_gen = TabPFNClassifier()
                    reg_gen = TabPFNRegressor()
                    unsup = TabPFNUnsupervisedModel(tabpfn_clf=clf_gen, tabpfn_reg=reg_gen)
                    unsup.fit(torch.tensor(self._X_ctx, dtype=torch.float32))
                    X_syn_tensor = unsup.generate_synthetic_data(
                        n_samples=n_samples,
                        t=self._t,
                        n_permutations=self._n_permutations,
                    )

                X_syn = X_syn_tensor.numpy().astype(np.float64)

                # Sanitize: replace NaN/Inf and clip extreme values
                n_bad = np.count_nonzero(~np.isfinite(X_syn))
                if n_bad > 0:
                    log.warning(f"[tabpfngen] Replacing {n_bad} non-finite values in X_syn")
                for j in range(X_syn.shape[1]):
                    col = X_syn[:, j]
                    bad_mask = ~np.isfinite(col)
                    if bad_mask.any():
                        col_median = np.nanmedian(self._X_ctx[:, j])
                        col[bad_mask] = col_median
                f32_max = np.finfo(np.float32).max / 10
                X_syn = np.clip(X_syn, -f32_max, f32_max).astype(np.float32)
                break  # success
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    log.warning(
                        f"[tabpfngen] Generation attempt {attempt+1}/{MAX_RETRIES} failed: {e}. "
                        f"Retrying with perturbed random state..."
                    )
                    # Perturb random state so next attempt samples differently
                    torch.manual_seed(self._seed + 1000 * (attempt + 1))
                    np.random.seed(self._seed + 1000 * (attempt + 1))
                    continue
                raise  # exhausted retries

        log.info(f"[tabpfngen] X_syn generated: shape={X_syn.shape}")

        # ── Step 2: Label X_syn with TabPFN's predictive distribution ───────
        if self._is_regression:
            y_syn = self._assign_labels_regression(X_syn)
            log.info(
                f"[tabpfngen] y_syn assigned (regression): "
                f"mean={y_syn.mean():.3f}, std={y_syn.std():.3f}"
            )
        else:
            n_classes = len(self._classes)
            y_syn = self._assign_labels(X_syn, n_classes)
            log.info(
                f"[tabpfngen] y_syn assigned (mode={self._y_mode}): "
                f"class distribution={np.bincount(y_syn.astype(int))}"
            )

        return X_syn, y_syn

    def _assign_labels(self, X_syn: np.ndarray, n_classes: int) -> np.ndarray:
        """Assign labels to X_syn using TabPFN's predictive distribution."""
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as e:
            raise ImportError("tabpfn required for label assignment.") from e

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = TabPFNClassifier()
            clf.fit(self._X_ctx, self._y_ctx)
            y_proba = clf.predict_proba(X_syn)   # shape: (n_samples, n_classes)

        if self._y_mode == "argmax":
            y_syn = np.argmax(y_proba, axis=1)
        else:
            # Sample from the predicted distribution (more diverse, less deterministic)
            # Explicit re-normalize to float64 to fix floating-point precision errors
            # where predict_proba rows may sum to 1.0 ± epsilon.
            rng = np.random.default_rng(self._seed)
            y_syn = np.array([
                rng.choice(n_classes, p=_safe_proba(p))
                for p in y_proba
            ])

        # Map back to original class labels (in case they aren't 0..K-1)
        return self._classes[y_syn].copy()

    def _assign_labels_regression(self, X_syn: np.ndarray) -> np.ndarray:
        """Assign continuous labels to X_syn using TabPFN's regression prediction."""
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as e:
            raise ImportError("tabpfn required for regression label assignment.") from e

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg = TabPFNRegressor()
            reg.fit(self._X_ctx, self._y_ctx)
            y_syn = reg.predict(X_syn)

        # Add small noise to avoid degenerate predictions
        rng = np.random.default_rng(self._seed)
        y_std = np.std(self._y_ctx)
        noise = rng.normal(0, y_std * 0.01, size=len(y_syn))
        y_syn = (y_syn + noise).astype(np.float32)

        return y_syn
