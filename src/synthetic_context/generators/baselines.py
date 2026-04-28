import logging
import warnings

import numpy as np
from sklearn.mixture import GaussianMixture
from .base import BaseGenerator

log = logging.getLogger(__name__)


def _is_regression_target(y: np.ndarray) -> bool:
    """Return True if y looks like a regression target (float with many unique values)."""
    return (np.issubdtype(y.dtype, np.floating) and
            len(np.unique(y)) > 20)


class RandomGenerator(BaseGenerator):
    """
    L0 baseline: X ~ N(0,1) per feature, y drawn from class prior.

    For regression: y ~ Uniform(y_min, y_max).
    No structure is preserved; serves as the lower bound for synthetic
    context quality.
    """

    name = "random"
    level = "L0"

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._n_features: int | None = None
        self._classes: np.ndarray | None = None
        self._class_probs: np.ndarray | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._is_regression: bool = False
        self._y_min: float = 0.0
        self._y_max: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomGenerator":
        self._X_train = X.copy()
        self._y_train = y.copy()
        self._n_features = X.shape[1]
        self._is_regression = _is_regression_target(y)
        if self._is_regression:
            self._y_min = float(y.min())
            self._y_max = float(y.max())
        else:
            self._classes, counts = np.unique(y, return_counts=True)
            self._class_probs = counts / counts.sum()
        return self

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        X_syn = self._rng.standard_normal((n_samples, self._n_features))
        if self._is_regression:
            y_syn = self._rng.uniform(
                self._y_min, self._y_max, n_samples
            ).astype(np.float32)
        else:
            y_syn = self._rng.choice(self._classes, size=n_samples, p=self._class_probs)
        return X_syn.astype(np.float32), y_syn


class MarginalGenerator(BaseGenerator):
    """
    L1 baseline: bootstrap each column independently from X_train.

    Preserves marginal per-feature distributions but destroys all
    inter-feature correlations. y is drawn i.i.d. from y_train
    (works for both classification and regression).
    """

    name = "marginal"
    level = "L1"

    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._class_probs: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MarginalGenerator":
        self._X_train = X.copy()
        self._y_train = y.copy()
        if not _is_regression_target(y):
            self._classes, counts = np.unique(y, return_counts=True)
            self._class_probs = counts / counts.sum()
        return self

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        X_syn = np.column_stack([
            self._rng.choice(self._X_train[:, j], size=n_samples, replace=True)
            for j in range(self._X_train.shape[1])
        ])
        # Bootstrap from y_train directly — preserves dtype for both clf and reg
        y_syn = self._rng.choice(self._y_train, size=n_samples, replace=True)
        return X_syn.astype(np.float32), y_syn


class GMMGenerator(BaseGenerator):
    """
    L2 baseline: Gaussian Mixture Model.

    Classification: fit one GMM per class, sample proportional to class priors.
    Regression:     fit a single GMM on [X | y] jointly, sample and split.

    n_components = min(5, max(1, n_class_samples // 10)) per class (classification).
    For regression a fixed n_components=5 is used (or min(5, n_samples // 10)).
    """

    name = "gmm"
    level = "L2"

    def __init__(self, n_components: int = 5, seed: int = 42):
        self._n_components = n_components
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._gmms: dict = {}
        self._classes: np.ndarray | None = None
        self._class_probs: np.ndarray | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._is_regression: bool = False
        self._gmm_joint: GaussianMixture | None = None  # for regression

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GMMGenerator":
        self._X_train = X.copy()
        self._y_train = y.copy()
        self._is_regression = _is_regression_target(y)

        if self._is_regression:
            # Fit a single GMM on the full data [X, y] jointly
            n = len(y)
            n_comp = min(self._n_components, max(1, n // 10))
            XY = np.column_stack([X, y.reshape(-1, 1)])
            self._gmm_joint = GaussianMixture(
                n_components=n_comp, random_state=self._seed
            )
            self._gmm_joint.fit(XY)
        else:
            self._classes, counts = np.unique(y, return_counts=True)
            self._class_probs = counts / counts.sum()
            for cls, count in zip(self._classes, counts):
                X_cls = X[y == cls]
                n_comp = min(self._n_components, max(1, count // 10))
                gmm = GaussianMixture(n_components=n_comp, random_state=self._seed)
                gmm.fit(X_cls)
                self._gmms[cls] = gmm
        return self

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if self._is_regression:
            # Sample from joint GMM, last column is y
            XY_syn, _ = self._gmm_joint.sample(n_samples)
            X_syn = XY_syn[:, :-1].astype(np.float32)
            y_syn = XY_syn[:, -1].astype(np.float32)
            return X_syn, y_syn
        else:
            n_per_class = self._rng.multinomial(n_samples, self._class_probs)
            X_parts, y_parts = [], []
            for cls, n in zip(self._classes, n_per_class):
                if n == 0:
                    continue
                X_cls, _ = self._gmms[cls].sample(n)
                X_parts.append(X_cls)
                y_parts.append(np.full(n, cls, dtype=int))
            X_syn = np.vstack(X_parts).astype(np.float32)
            y_syn = np.concatenate(y_parts)
            # Shuffle to mix classes
            idx = self._rng.permutation(len(y_syn))
            return X_syn[idx], y_syn[idx]


class SMOTEGenerator(BaseGenerator):
    """
    L3 baseline: SMOTE oversampling followed by stratified subsampling.

    Classification: Uses imblearn SMOTE with k_neighbors = min(5, min_class_count - 1).
    Oversamples minority classes to balance, then subsamples to n_samples
    with class stratification matching the balanced distribution.

    Regression: k-NN interpolation on ALL training samples (not class-based).
    For each synthetic sample:
      1. Pick a random training point x_i
      2. Pick one of its k nearest neighbors x_j
      3. X_syn = x_i + λ * (x_j - x_i), y_syn = y_i + λ * (y_j - y_i)
    where λ ~ U(0, 1). This preserves local joint structure (feature + target).
    """

    name = "smote"
    level = "L3"

    def __init__(self, k_neighbors: int = 5, seed: int = 42):
        self._k_neighbors = k_neighbors
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._class_probs: np.ndarray | None = None
        self._X_resampled: np.ndarray | None = None
        self._y_resampled: np.ndarray | None = None
        self._is_regression: bool = False
        self._nn_indices: np.ndarray | None = None  # for regression k-NN

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SMOTEGenerator":
        self._X_train = X.copy()
        self._y_train = y.copy()
        self._is_regression = _is_regression_target(y)

        if self._is_regression:
            # Precompute k-NN for regression SMOTE
            from sklearn.neighbors import NearestNeighbors
            k = min(self._k_neighbors, len(X) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
            nn.fit(X)
            # Shape: (n_train, k+1) — column 0 is self, columns 1..k are neighbors
            self._nn_indices = nn.kneighbors(X, return_distance=False)[:, 1:]  # exclude self
            return self

        # Classification: standard imblearn SMOTE
        from imblearn.over_sampling import SMOTE

        self._classes, counts = np.unique(y, return_counts=True)
        self._class_probs = counts / counts.sum()

        min_count = counts.min()
        k = min(self._k_neighbors, max(1, min_count - 1))
        smote = SMOTE(k_neighbors=k, random_state=self._seed)
        self._X_resampled, self._y_resampled = smote.fit_resample(X, y)
        return self

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if self._is_regression:
            return self._generate_regression(n_samples)
        return self._generate_classification(n_samples)

    def _generate_regression(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """k-NN interpolation for regression: preserves local joint structure."""
        n_train = len(self._X_train)
        k = self._nn_indices.shape[1]

        # Pick random anchor points
        anchor_idx = self._rng.integers(0, n_train, size=n_samples)
        # Pick random neighbor index (0..k-1) for each anchor
        nn_choice = self._rng.integers(0, k, size=n_samples)
        neighbor_idx = self._nn_indices[anchor_idx, nn_choice]
        # Interpolation weight
        lam = self._rng.random(n_samples).astype(np.float32)

        X_a = self._X_train[anchor_idx]
        X_b = self._X_train[neighbor_idx]
        y_a = self._y_train[anchor_idx]
        y_b = self._y_train[neighbor_idx]

        X_syn = X_a + lam[:, None] * (X_b - X_a)
        y_syn = y_a + lam * (y_b - y_a)

        return X_syn.astype(np.float32), y_syn.astype(np.float32)

    def _generate_classification(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Standard imblearn SMOTE subsampling for classification."""
        classes, counts = np.unique(self._y_resampled, return_counts=True)
        class_probs = counts / counts.sum()

        # Stratified subsample: draw n_per_class proportional to balanced priors
        n_per_class = self._rng.multinomial(n_samples, class_probs)
        idx = []
        for cls, n in zip(classes, n_per_class):
            cls_idx = np.where(self._y_resampled == cls)[0]
            chosen = self._rng.choice(
                cls_idx,
                size=min(n, len(cls_idx)) if n <= len(cls_idx) else n,
                replace=(n > len(cls_idx)),
            )
            idx.extend(chosen.tolist())

        idx = np.array(idx, dtype=np.intp)
        self._rng.shuffle(idx)
        return self._X_resampled[idx].astype(np.float32), self._y_resampled[idx]
