"""
Script: tabsyn.py
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
import types
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import BaseGenerator

log = logging.getLogger(__name__)

TABSYN_ROOT = Path("./tabsyn")


# ─── sys.path / import helpers ────────────────────────────────────────────────

def _ensure_tabsyn_on_path() -> None:
    root_str = str(TABSYN_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if not TABSYN_ROOT.exists():
        raise ImportError(
            f"TabSyn not found at {TABSYN_ROOT}. "
            "Run: bash scripts/clone_tabsyn.sh"
        )
    try:
        import tabsyn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"TabSyn import failed after adding {TABSYN_ROOT} to sys.path. "
            f"Error: {e}"
        ) from e


@contextlib.contextmanager
def _cwd(path: Path):
    """Temporarily change CWD — TabSyn resolves 'data/' relative to CWD."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ─── Data format helpers ───────────────────────────────────────────────────────

def _is_regression_target(y: np.ndarray) -> bool:
    """Return True if y looks like a regression target."""
    return (np.issubdtype(y.dtype, np.floating) and
            len(np.unique(y)) > 20)


def _write_tabsyn_data(
    data_dir: Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Write arrays into TabSyn's expected directory format and return info dict.

    For classification: ALL features are numerical, label is the sole 'categorical'
    column. TabSyn's concat_y_to_X prepends y as X_cat.

    For regression: ALL columns (features + target) are numerical.
    TabSyn handles regression by treating the target as one more numerical column.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    n_features = X_train.shape[1]
    is_regression = _is_regression_target(y_train)

    if is_regression:
        task_type = "regression"
        n_classes = 0  # not applicable

        # For regression: X_num = features only (NOT target).
        # TabSyn's make_dataset calls concat_y_to_X(X_num, y) which prepends y
        # to X_num during loading. Writing y separately avoids double-inclusion.
        np.save(data_dir / "X_num_train.npy", X_train.astype(np.float32))
        np.save(data_dir / "X_num_test.npy",  X_test.astype(np.float32))

        # Labels as float32 for regression
        np.save(data_dir / "y_train.npy", y_train.astype(np.float32))
        np.save(data_dir / "y_test.npy",  y_test.astype(np.float32))

        # TabSyn requires X_cat to exist and have >=1 column (OrdinalEncoder
        # fails on (n,0); preprocess crashes on None). Write a single dummy
        # constant column — TabSyn will encode it as 1 category with 0 capacity.
        # We discard it when parsing the output CSV.
        np.save(data_dir / "X_cat_train.npy", np.zeros((len(y_train), 1), dtype=np.int64))
        np.save(data_dir / "X_cat_test.npy",  np.zeros((len(y_test),  1), dtype=np.int64))

        # Column layout: features are numerical, dummy is categorical, target is separate.
        # After concat_y_to_X on X_num: internal num = [target, feature_0, ..., feature_{n-1}]
        # X_cat = [dummy_col] stays as-is for regression (no concat on cat side).
        # Use n_features for dummy col idx, n_features+1 for target.
        num_col_idx = list(range(n_features))
        cat_col_idx = [n_features]       # dummy categorical column
        target_col_idx = [n_features + 1]

    else:
        n_classes = int(np.unique(y_train).shape[0])
        task_type = "binclass" if n_classes == 2 else "multiclass"

        # Numerical features
        np.save(data_dir / "X_num_train.npy", X_train.astype(np.float32))
        np.save(data_dir / "X_num_test.npy",  X_test.astype(np.float32))

        # Labels (1-D int arrays)
        np.save(data_dir / "y_train.npy", y_train.astype(int))
        np.save(data_dir / "y_test.npy",  y_test.astype(int))

        # Empty (n, 0) X_cat files
        np.save(data_dir / "X_cat_train.npy", np.empty((len(y_train), 0), dtype=np.int64))
        np.save(data_dir / "X_cat_test.npy",  np.empty((len(y_test),  0), dtype=np.int64))

        num_col_idx = list(range(n_features))
        cat_col_idx = []
        target_col_idx = [n_features]

    # ── info.json ────────────────────────────────────────────────────────────
    n_total_cols = len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)
    idx_mapping: dict[int, int] = {i: i for i in range(n_total_cols)}
    inverse_idx_mapping = {v: k for k, v in idx_mapping.items()}
    idx_name_mapping = {i: f"feature_{i}" for i in range(n_features)}
    if is_regression:
        idx_name_mapping[n_features] = "_dummy_cat"  # dummy categorical column
        idx_name_mapping[n_features + 1] = "target"
    else:
        idx_name_mapping[n_features] = "target"

    info = {
        "task_type": task_type,
        "n_classes": n_classes,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "idx_mapping": {str(k): v for k, v in idx_mapping.items()},
        "inverse_idx_mapping": {str(k): v for k, v in inverse_idx_mapping.items()},
        "idx_name_mapping": {str(k): v for k, v in idx_name_mapping.items()},
    }

    with open(data_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    return info


# ─── Training helpers ──────────────────────────────────────────────────────────

def _train_vae(dataname: str, device: str, num_epochs: int) -> None:
    """Stage 1: train VAE. Must be called with CWD=TABSYN_ROOT."""
    from tabsyn.vae.main import main as vae_main  # type: ignore[import]
    # NOTE: TabSyn's vae/main.py hardcodes num_epochs=4000 as a local variable;
    # the args.num_epochs we pass is ignored by the original code.
    # FIX: args.gpu must be set — TabSyn checks `args.gpu != -1` to pick cuda device.
    args = types.SimpleNamespace(
        dataname=dataname,
        device=device,
        gpu=0,           # triggers args.device = 'cuda:0' inside vae main
        max_beta=1e-3,
        min_beta=1e-5,
        lambd=0.7,
    )
    t0 = time.time()
    log.info(f"[TabSyn] Stage 1 VAE '{dataname}' (4000 epochs hardcoded) …")
    vae_main(args)
    log.info(f"[TabSyn] VAE done in {(time.time()-t0)/60:.1f} min")


def _train_diffusion(dataname: str, device: str, num_epochs: int) -> None:
    """Stage 2: train diffusion model. Must be called with CWD=TABSYN_ROOT."""
    from tabsyn.main import main as diff_main  # type: ignore[import]
    # NOTE: TabSyn's main.py hardcodes num_epochs=10001; args.num_epochs is ignored.
    # FIX: args.gpu must be set.
    args = types.SimpleNamespace(
        dataname=dataname,
        device=device,
        gpu=0,           # triggers args.device = 'cuda:0' inside diff main
    )
    t0 = time.time()
    log.info(f"[TabSyn] Stage 2 Diffusion '{dataname}' (10001 epochs hardcoded) …")
    diff_main(args)
    log.info(f"[TabSyn] Diffusion done in {(time.time()-t0)/60:.1f} min")


def _run_sample(dataname: str, device: str, n_samples: int, save_path: str) -> None:
    """Sampling step. Must be called with CWD=TABSYN_ROOT.

    NOTE: TabSyn's sample.py always generates train_z.shape[0] samples
    (the training set size), ignoring any n_samples argument.
    Caller must subsample/resample the output CSV as needed.
    FIX: args.gpu must be set.
    """
    from tabsyn.sample import main as sample_main  # type: ignore[import]
    args = types.SimpleNamespace(
        dataname=dataname,
        device=device,
        gpu=0,           # triggers args.device = 'cuda:0' inside sample main
        steps=50,
        save_path=save_path,
    )
    sample_main(args)


# ─── Generator class ──────────────────────────────────────────────────────────

class TabSynGenerator(BaseGenerator):
    """
    TabSyn wrapper — two-stage VAE + score-based diffusion on tabular data.

    Data format conversion: our skrub-encoded numpy arrays → TabSyn's .npy
    directory format with info.json metadata. ALL features are treated as
    numerical; the label is the sole 'categorical' column as TabSyn expects.

    Requires:  bash scripts/clone_tabsyn.sh
    """

    name = "tabsyn"
    level = "L6"

    def __init__(
        self,
        vae_epochs: int = 4000,
        diff_epochs: int = 10000,
        seed: int = 0,
        device: str = "cuda",
        stable_name: str | None = None,
        **kwargs: Any,
    ):
        self.vae_epochs = vae_epochs
        self.diff_epochs = diff_epochs
        self.seed = seed
        self.device = device
        # stable_name: if set, use as checkpoint directory name (enables reuse
        # across experiment runs).  If None, a random UUID is used (old behavior).
        self.stable_name = stable_name
        self._dataname: str | None = None
        self._n_features: int | None = None
        self._n_classes: int | None = None
        self._is_regression: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabSynGenerator":
        _ensure_tabsyn_on_path()

        self._n_features = X.shape[1]
        self._n_classes = int(np.unique(y).shape[0])
        self._is_regression = _is_regression_target(y)

        # Determine checkpoint name: stable (reusable) or one-off UUID
        self._dataname = self.stable_name if self.stable_name else f"synctx_{uuid.uuid4().hex[:8]}"

        # TabSyn VAE needs a validation split to compute val loss.
        # Always write the data dir — sampling also reads training data.
        from sklearn.model_selection import train_test_split as tts
        val_frac = max(0.1, min(200 / len(y), 0.2))
        stratify_arg = None if self._is_regression else y
        X_tr, X_te, y_tr, y_te = tts(
            X, y, test_size=val_frac, stratify=stratify_arg, random_state=self.seed
        )
        data_dir = TABSYN_ROOT / "data" / self._dataname
        _write_tabsyn_data(data_dir, X_tr, y_tr, X_te, y_te)
        log.info(
            f"TabSyn data written → {data_dir} "
            f"(n_train={len(y_tr)}, n_val={len(y_te)}, "
            f"n_feat={self._n_features}, n_cls={self._n_classes})"
        )

        # Check for existing checkpoint — skip training if both stages done
        vae_ckpt  = TABSYN_ROOT / "tabsyn" / "vae" / "ckpt" / self._dataname / "model.pt"
        diff_ckpt = TABSYN_ROOT / "tabsyn" / "ckpt" / self._dataname / "model.pt"
        if vae_ckpt.exists() and diff_ckpt.exists():
            log.info(
                f"[TabSyn] Checkpoint found for '{self._dataname}' — skipping training."
            )
            return self

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        with _cwd(TABSYN_ROOT):
            _train_vae(self._dataname, self.device, self.vae_epochs)
            _train_diffusion(self._dataname, self.device, self.diff_epochs)

        return self

    def generate(self, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        if self._dataname is None:
            raise RuntimeError("Call fit() before generate().")
        _ensure_tabsyn_on_path()

        import pandas as pd

        save_path = str(TABSYN_ROOT / "data" / self._dataname / "syn_samples.csv")

        with _cwd(TABSYN_ROOT):
            _run_sample(self._dataname, self.device, n_samples, save_path)

        syn_df = pd.read_csv(save_path)

        feature_cols = [f"feature_{i}" for i in range(self._n_features)]
        X_syn = syn_df[feature_cols].to_numpy(dtype=np.float32)
        if self._is_regression:
            y_syn = syn_df["target"].to_numpy(dtype=np.float32)
        else:
            y_syn = np.clip(
                syn_df["target"].to_numpy().astype(int), 0, self._n_classes - 1
            )

        # TabSyn always generates train_set_size samples, not n_samples.
        # Subsample or resample with replacement to match requested n_samples.
        n_generated = len(y_syn)
        if n_generated != n_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n_generated, size=n_samples, replace=(n_samples > n_generated))
            X_syn, y_syn = X_syn[idx], y_syn[idx]
            log.info(
                f"TabSyn: resampled {n_generated} → {n_samples} "
                f"({'with' if n_samples > n_generated else 'without'} replacement)"
            )

        log.info(f"TabSyn generated {len(y_syn)} samples.")
        return X_syn, y_syn
