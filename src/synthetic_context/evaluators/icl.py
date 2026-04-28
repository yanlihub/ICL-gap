"""
Script: icl.py
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Literal

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# NaN sentinel result dicts per task type
# ---------------------------------------------------------------------------

_NAN_RESULT_BINARY: dict[str, float] = {
    "roc_auc": float("nan"),
    "balanced_accuracy": float("nan"),
    "log_loss": float("nan"),
    "inference_time_s": float("nan"),
}

_NAN_RESULT_MULTICLASS: dict[str, float] = {
    "roc_auc": float("nan"),
    "accuracy": float("nan"),
    "macro_f1": float("nan"),
    "log_loss": float("nan"),
    "inference_time_s": float("nan"),
}

_NAN_RESULT_REGRESSION: dict[str, float] = {
    "rmse": float("nan"),
    "mae": float("nan"),
    "r2": float("nan"),
    "inference_time_s": float("nan"),
}


def _nan_result(task_type: str) -> dict[str, float]:
    if task_type == "regression":
        return dict(_NAN_RESULT_REGRESSION)
    elif task_type == "multiclass":
        return dict(_NAN_RESULT_MULTICLASS)
    else:
        return dict(_NAN_RESULT_BINARY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_icl(
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model: Literal["tabpfn", "tabicl"] = "tabpfn",
    seed: int = 42,
    device: str = "cuda",
    task_type: str = "binary",
) -> dict[str, float]:
    """
    Evaluate an ICL model using (X_ctx, y_ctx) as in-context examples.

    The model is fitted on the context set and evaluated on the held-out
    test set.  No separate training split is created; the entire context is
    treated as the "training" data visible to the ICL model.

    Parameters
    ----------
    X_ctx:
        Feature matrix for the context (in-context training) examples.
        Shape: ``(n_ctx, n_features)``.
    y_ctx:
        Labels/targets for the context examples. Shape: ``(n_ctx,)``.
    X_test:
        Feature matrix for the test examples. Shape: ``(n_test, n_features)``.
    y_test:
        True labels/targets for the test examples. Shape: ``(n_test,)``.
    model:
        Which ICL backbone to use.  One of ``"tabpfn"`` or ``"tabicl"``.
    seed:
        Random seed forwarded to the model constructor.
    device:
        Device string (``"cuda"`` or ``"cpu"``) forwarded to the model.
    task_type:
        One of ``"binary"``, ``"multiclass"``, or ``"regression"``.

    Returns
    -------
    dict with keys depending on task_type:

    Binary classification:
        ``roc_auc``, ``balanced_accuracy``, ``log_loss``, ``inference_time_s``
    Multiclass classification:
        ``roc_auc`` (macro OvR), ``accuracy``, ``macro_f1``, ``log_loss``,
        ``inference_time_s``
    Regression:
        ``rmse``, ``mae``, ``r2``, ``inference_time_s``
    """
    nan_result = _nan_result(task_type)

    # ------------------------------------------------------------------
    # 1. Instantiate the model
    # ------------------------------------------------------------------
    mdl = _build_model(model=model, seed=seed, device=device, task_type=task_type)
    if mdl is None:
        return nan_result

    # ------------------------------------------------------------------
    # 2. Fit on context (with automatic CPU fallback if GPU unavailable)
    # ------------------------------------------------------------------
    try:
        mdl.fit(X_ctx, y_ctx)
    except Exception as exc:
        if device != "cpu" and (
            "GPU" in str(exc) or "CUDA" in str(exc) or "HIP" in str(exc)
            or "cuda" in str(exc).lower() or "hip" in str(exc).lower()
        ):
            warnings.warn(
                f"[evaluate_icl] {model} GPU unavailable ({exc}), retrying on CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            mdl = _build_model(model=model, seed=seed, device="cpu", task_type=task_type)
            if mdl is None:
                return nan_result
            try:
                mdl.fit(X_ctx, y_ctx)
            except Exception as exc2:
                warnings.warn(
                    f"[evaluate_icl] {model} CPU fallback also failed: {exc2}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return nan_result
        else:
            warnings.warn(
                f"[evaluate_icl] {model} fit() raised {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return nan_result

    # ------------------------------------------------------------------
    # 3. Predict on test set (timed)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        if task_type == "regression":
            y_pred = mdl.predict(X_test)
            y_proba = None
        else:
            y_proba = mdl.predict_proba(X_test)
            y_pred = mdl.predict(X_test)
    except Exception as exc:
        warnings.warn(
            f"[evaluate_icl] {model} predict() raised {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return nan_result
    inference_time_s = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 4. Sanity-check predictions
    # ------------------------------------------------------------------
    if np.any(np.isnan(y_pred)):
        warnings.warn(
            f"[evaluate_icl] {model} returned NaN predictions.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {**nan_result, "inference_time_s": inference_time_s}

    if y_proba is not None and np.any(np.isnan(y_proba)):
        warnings.warn(
            f"[evaluate_icl] {model} returned NaN probabilities.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {**nan_result, "inference_time_s": inference_time_s}

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    if task_type == "regression":
        return _compute_regression_metrics(y_test, y_pred, inference_time_s)
    else:
        classes = np.unique(np.concatenate([y_ctx, y_test]))
        n_classes = len(classes)
        if task_type == "multiclass" or n_classes > 2:
            return _compute_multiclass_metrics(
                y_test, y_pred, y_proba, classes, mdl, inference_time_s
            )
        else:
            return _compute_binary_metrics(
                y_test, y_pred, y_proba, classes, mdl, inference_time_s
            )


# ---------------------------------------------------------------------------
# Private metric helpers
# ---------------------------------------------------------------------------


def _compute_binary_metrics(
    y_test, y_pred, y_proba, classes, mdl, inference_time_s: float
) -> dict[str, float]:
    try:
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    except Exception:
        bal_acc = float("nan")

    try:
        ll = float(log_loss(y_test, y_proba, labels=classes))
    except Exception:
        ll = float("nan")

    try:
        clf_classes = list(getattr(mdl, "classes_", classes))
        pos_idx = clf_classes.index(classes[1]) if classes[1] in clf_classes else 1
        roc = float(roc_auc_score(y_test, y_proba[:, pos_idx]))
    except Exception:
        roc = float("nan")

    return {
        "roc_auc": roc,
        "balanced_accuracy": bal_acc,
        "log_loss": ll,
        "inference_time_s": inference_time_s,
    }


def _compute_multiclass_metrics(
    y_test, y_pred, y_proba, classes, mdl, inference_time_s: float
) -> dict[str, float]:
    try:
        roc = float(
            roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=classes,
            )
        )
    except Exception:
        roc = float("nan")

    try:
        acc = float(np.mean(y_pred == y_test))
    except Exception:
        acc = float("nan")

    try:
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", labels=classes))
    except Exception:
        macro_f1 = float("nan")

    try:
        ll = float(log_loss(y_test, y_proba, labels=classes))
    except Exception:
        ll = float("nan")

    return {
        "roc_auc": roc,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "log_loss": ll,
        "inference_time_s": inference_time_s,
    }


def _compute_regression_metrics(
    y_test, y_pred, inference_time_s: float
) -> dict[str, float]:
    try:
        rmse = float(math.sqrt(mean_squared_error(y_test, y_pred)))
    except Exception:
        rmse = float("nan")

    try:
        mae = float(mean_absolute_error(y_test, y_pred))
    except Exception:
        mae = float("nan")

    try:
        r2 = float(r2_score(y_test, y_pred))
    except Exception:
        r2 = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "inference_time_s": inference_time_s,
    }


# ---------------------------------------------------------------------------
# Private model builder
# ---------------------------------------------------------------------------


def _build_model(
    model: str,
    seed: int,
    device: str,
    task_type: str = "binary",
):
    """
    Instantiate the requested ICL model (classifier or regressor).

    Returns ``None`` and emits a warning if the required package is missing
    or if the model/task_type combination is not supported.
    """
    is_regression = (task_type == "regression")

    if model == "tabpfn":
        if is_regression:
            try:
                from tabpfn import TabPFNRegressor  # type: ignore[import]
            except ImportError:
                warnings.warn(
                    "TabPFNRegressor is not available in the installed tabpfn version. "
                    "Update tabpfn to use regression support.",
                    ImportWarning,
                    stacklevel=3,
                )
                return None
            return TabPFNRegressor(
                device=device,
                random_state=seed,
            )
        else:
            try:
                from tabpfn import TabPFNClassifier  # type: ignore[import]
            except ImportError:
                warnings.warn(
                    "TabPFN is not installed. "
                    "Install it with: pip install tabpfn",
                    ImportWarning,
                    stacklevel=3,
                )
                return None
            return TabPFNClassifier(
                device=device,
                random_state=seed,
            )

    elif model == "tabicl":
        if is_regression:
            try:
                from tabicl import TabICLRegressor  # type: ignore[import]
                return TabICLRegressor(
                    device=device,
                    random_state=seed,
                )
            except ImportError:
                warnings.warn(
                    "TabICLRegressor is not available in the installed tabicl version. "
                    "Returning NaN for regression with TabICL.",
                    ImportWarning,
                    stacklevel=3,
                )
                return None
        else:
            try:
                from tabicl import TabICLClassifier  # type: ignore[import]
            except ImportError:
                warnings.warn(
                    "TabICL is not installed. "
                    "Install it with: pip install tabicl",
                    ImportWarning,
                    stacklevel=3,
                )
                return None
            return TabICLClassifier(
                device=device,
                random_state=seed,
            )

    else:
        raise ValueError(
            f"Unknown ICL model '{model}'. Supported options: 'tabpfn', 'tabicl'."
        )


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

def _build_classifier(model: str, seed: int, device: str):
    """Deprecated alias for _build_model. Use _build_model() instead."""
    return _build_model(model=model, seed=seed, device=device, task_type="binary")
