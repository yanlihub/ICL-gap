"""
Script: synthcity_generators.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseGenerator

log = logging.getLogger(__name__)

# synthcity creates a 'workspace/' dir in CWD; point it to writable scratch
_SYNTHCITY_WS = Path("./synthcity_ws")

# Column name used for the label inside synthcity DataFrames
_LABEL_COL = "__label__"


def _to_df(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X.astype(np.float32), columns=cols)
    df[_LABEL_COL] = y.astype(int)
    return df


def _from_df(df: pd.DataFrame, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    feat_cols = [f"f{i}" for i in range(n_features)]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df[_LABEL_COL].to_numpy().astype(int)
    return X, y


class _SynthcityGenerator(BaseGenerator):
    """Base for synthcity plugin wrappers."""

    _plugin_name: str = ""
    _level: str = ""

    def __init__(self, n_iter: int = 500, batch_size: int = 500,
                 seed: int = 42, **kwargs: Any):
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.seed = seed
        self._extra_kwargs = kwargs
        self._plugin = None
        self._n_features: int | None = None
        self._n_classes: int | None = None

    @property
    def name(self) -> str:
        return self._plugin_name

    @property
    def level(self) -> str:
        return self._level

    def _build_plugin(self):
        try:
            from synthcity.plugins import Plugins  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "synthcity is not installed. "
                "Run: bash scripts/install_deps.sh"
            ) from e
        return Plugins().get(
            self._plugin_name,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            random_state=self.seed,
            **self._extra_kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SynthcityGenerator":
        self._n_features = X.shape[1]
        self._n_classes = int(np.unique(y).shape[0])
        df = _to_df(X, y)

        # synthcity writes a 'workspace/' dir relative to CWD — must be writable
        _SYNTHCITY_WS.mkdir(parents=True, exist_ok=True)
        _old_cwd = os.getcwd()
        os.chdir(_SYNTHCITY_WS)
        try:
            log.info(f"[{self.name}] Fitting on {len(y)} samples …")
            self._plugin = self._build_plugin()
            from synthcity.plugins.core.dataloader import GenericDataLoader  # type: ignore
            loader = GenericDataLoader(df, target_column=_LABEL_COL)
            self._plugin.fit(loader)
            log.info(f"[{self.name}] Fit done.")
        finally:
            os.chdir(_old_cwd)
        return self

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if self._plugin is None:
            raise RuntimeError("Call fit() before generate().")

        log.info(f"[{self.name}] Generating {n_samples} samples …")
        syn = self._plugin.generate(count=n_samples)
        # synthcity returns a DataLoader; get the underlying DataFrame
        syn_df = syn.dataframe() if hasattr(syn, "dataframe") else syn.data

        X_syn, y_syn = _from_df(syn_df, self._n_features)

        # Clip labels to valid range (model may produce out-of-range values)
        y_syn = np.clip(y_syn, 0, self._n_classes - 1)
        log.info(f"[{self.name}] Generated {len(y_syn)} samples.")
        return X_syn, y_syn


class CTGANGenerator(_SynthcityGenerator):
    """CTGAN — Conditional Tabular GAN (Xu et al., 2019). Level L4."""
    _plugin_name = "ctgan"
    _level = "L4"


class TVAEGenerator(_SynthcityGenerator):
    """TVAE — Tabular Variational Autoencoder (Xu et al., 2019). Level L5."""
    _plugin_name = "tvae"
    _level = "L5"


class PATEGANGenerator(_SynthcityGenerator):
    """
    PATE-GAN — Differentially Private GAN (Jordon et al., 2018). Level L7.

    Uses PATE (Private Aggregation of Teachers' Ensembles) to achieve
    (epsilon, delta)-differential privacy during training.
    """
    _plugin_name = "pategan"
    _level = "L7"

    def __init__(
        self,
        n_iter: int = 500,
        batch_size: int = 500,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(n_iter=n_iter, batch_size=batch_size, seed=seed, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self._extra_kwargs["epsilon"] = epsilon
        self._extra_kwargs["delta"] = delta


class DPGANGenerator(_SynthcityGenerator):
    """
    DPGAN — Differentially Private GAN (Xie et al., 2018). Level L4-DP.

    Uses DP-SGD to train the discriminator under (epsilon, delta)-DP.
    Synthcity plugin name: 'dpgan'.
    """
    _plugin_name = "dpgan"
    _level = "L4-DP"

    def __init__(
        self,
        n_iter: int = 500,
        batch_size: int = 500,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        seed: int = 42,
        **kwargs: Any,
    ):
        super().__init__(n_iter=n_iter, batch_size=batch_size, seed=seed, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self._extra_kwargs["epsilon"] = epsilon
        self._extra_kwargs["delta"] = delta
