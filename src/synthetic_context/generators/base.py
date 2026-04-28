from abc import ABC, abstractmethod
import numpy as np


class BaseGenerator(ABC):
    """Abstract base class for all synthetic data generators."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseGenerator":
        """Fit the generator on training data."""
        ...

    @abstractmethod
    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate n_samples synthetic (X, y) pairs."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Generator name (e.g. 'tabsyn')."""
        ...

    @property
    @abstractmethod
    def level(self) -> str:
        """Quality level label (e.g. 'L0', 'L6')."""
        ...

    def fit_generate(
        self, X: np.ndarray, y: np.ndarray, n_samples: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: fit then generate."""
        n = n_samples if n_samples is not None else len(X)
        self.fit(X, y)
        return self.generate(n)
