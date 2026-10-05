"""
Script: syn_cache.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..generators.base import BaseGenerator

log = logging.getLogger(__name__)


def _snap_labels(y_syn: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    For classification tasks: snap float y_syn values to the nearest valid class label.

    Synthcity generators (CTGAN, TVAE, PATEGAN) treat y as a continuous column
    and may return float values (e.g. 2.7 for a 7-class problem). TabPFN/TabICL
    require integer class labels matching those seen in training. This function
    maps each generated y value to the nearest class in y_train.

    For regression (float y_train with many unique values), returns y_syn as-is.
    """
    classes = np.unique(y_train)
    # Heuristic: classification if y_train is integer dtype OR has <=30 unique values
    is_classification = (
        np.issubdtype(y_train.dtype, np.integer)
        or (np.issubdtype(y_train.dtype, np.floating) and len(classes) <= 30)
    )
    if not is_classification:
        return y_syn
    # Snap each generated value to the nearest valid class
    y_snapped = classes[np.argmin(
        np.abs(y_syn.reshape(-1, 1).astype(float) - classes.reshape(1, -1).astype(float)),
        axis=1
    )]
    return y_snapped.astype(y_train.dtype)


class SyntheticDataCache:
    """
    Persistent cache for generated synthetic datasets.

    Saves (X_syn, y_syn) as compressed .npz files keyed by
    (generator_name, openml_id, n_train, gen_seed).  On cache hit the
    generator is never fitted — important for expensive models like CTGAN.
    """

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_or_generate(
        self,
        generator: BaseGenerator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        openml_id: int,
        gen_seed: int = 0,
        n_samples: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return cached synthetic data if available, otherwise fit the generator,
        generate, cache, and return.

        Parameters
        ----------
        generator   : fitted generator instance (fit() will be called if cache miss)
        X_train     : real training features used to fit the generator
        y_train     : real training labels
        openml_id   : OpenML dataset id (part of the cache key)
        gen_seed    : seed used when constructing the generator
        n_samples   : number of synthetic samples (default = len(y_train))
        """
        n = n_samples if n_samples is not None else len(y_train)
        key = self._key(generator.name, openml_id, len(y_train), gen_seed)
        cache_path = self.cache_dir / f"{key}.npz"

        if cache_path.exists():
            log.info(f"[SynCache] HIT  {key}  ({cache_path})")
            data = np.load(cache_path, allow_pickle=True)
            X_syn, y_syn = data["X_syn"], data["y_syn"]
            # If more samples needed than cached, generate the difference
            if len(y_syn) < n:
                log.warning(
                    f"[SynCache] cached {len(y_syn)} samples < requested {n}; "
                    "generating remainder without re-fitting."
                )
                if not hasattr(generator, '_n_features') or generator._n_features is None:
                    generator.fit(X_train, y_train)
                X_extra, y_extra = generator.generate(n - len(y_syn))
                X_syn = np.vstack([X_syn, X_extra])
                y_syn = np.concatenate([y_syn, y_extra])
            return X_syn[:n], _snap_labels(y_syn[:n], y_train)

        log.info(f"[SynCache] MISS {key} — fitting {generator.name} …")
        generator.fit(X_train, y_train)
        X_syn, y_syn = generator.generate(n)
        np.savez_compressed(cache_path, X_syn=X_syn, y_syn=y_syn)
        log.info(f"[SynCache] saved {len(y_syn)} samples → {cache_path}")
        return X_syn, _snap_labels(y_syn, y_train)

    def exists(
        self, generator_name: str, openml_id: int, n_train: int, gen_seed: int
    ) -> bool:
        return (self.cache_dir / f"{self._key(generator_name, openml_id, n_train, gen_seed)}.npz").exists()

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _key(generator_name: str, openml_id: int, n_train: int, gen_seed: int) -> str:
        return f"{generator_name}_{openml_id}_n{n_train}_seed{gen_seed}"
