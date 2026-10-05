"""
Script: loader.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from skrub import TableVectorizer

log = logging.getLogger(__name__)

# Raw data directory: pre-downloaded CSV files (california_housing, news)
_RAW_DATA_DIR = Path("./data/raw")
_RAW_DATA_DIR_FALLBACK = Path("./data/raw")

# Local datasets loaded from pre-downloaded CSV files (not on OpenML).
# Both CSVs were saved by scripts/download_datasets.py with the target renamed to "target".
_LOCAL_DATASETS: dict[str, dict] = {
    "california_housing": {"target_col": "target"},
    "news": {"target_col": "target"},
}

# Canonical benchmark dataset registry: name → OpenML ID
DATASET_REGISTRY: dict[str, int] = {
    # ── 10-dataset final benchmark (see experiment_standards.md §1.1) ────
    # Large (>10K rows)
    "adult": 1590,
    "default": 42477,        # Default of Credit Card Clients (UCI/OpenML)
    "australian": 40981,     # Statlog Australian Credit Approval (n=690, d=14)
    "shoppers": 40981,       # legacy alias of "australian" (same OpenML id); NOT Online Shoppers
    "magic": 1120,           # Magic Gamma Telescope
    # "california_housing" → sklearn built-in, no OpenML ID, handled by download_datasets.py
    # "news"    → UCI ZIP, no OpenML ID, handled by download_datasets.py
    # Small (<10K rows)
    "breast_cancer": 15,
    "diabetes": 37,          # Pima Indians Diabetes
    "credit-g": 31,          # German Credit
    "abalone": 183,          # Abalone (regression, target=Rings)
    # ── Legacy / exploratory datasets ────────────────────────────────────
    "heart-statlog": 53,
    "blood-transfusion": 1464,
    "phoneme": 1489,
    "bank-marketing": 1461,
    "nomao": 1486,
    "pc4": 1049,
    "kc1": 1067,
    "covertype": 293,
    "higgs": 23512,
    "shuttle": 40685,
    "sick": 38,
    "oil-spill": 40701,
    "mammography": 310,
    "gisette": 41026,
    "vehicle": 54,
    "segment": 36,
    "kin8nm": 189,
    "house_16H": 574,
    "bodyfat": 560,
}

# Task type for each dataset: "binary", "multiclass", or "regression"
DATASET_TASK_TYPE: dict[str, str] = {
    # ── Final 10-dataset benchmark ────────────────────────────────────────
    "adult": "binary",
    "default": "binary",
    "australian": "binary",
    "shoppers": "binary",    # legacy alias of "australian"
    "magic": "binary",
    "california_housing": "regression",
    "news": "regression",
    "breast_cancer": "binary",
    "diabetes": "binary",
    "credit-g": "binary",
    "abalone": "regression",
    # ── Legacy datasets ───────────────────────────────────────────────────
    "heart-statlog": "binary",
    "blood-transfusion": "binary",
    "bank-marketing": "binary",
    "nomao": "binary",
    "phoneme": "binary",
    "pc4": "binary",
    "kc1": "binary",
    "sick": "binary",
    "oil-spill": "binary",
    "mammography": "binary",
    "gisette": "binary",
    "higgs": "binary",
    "covertype": "multiclass",
    "vehicle": "multiclass",
    "segment": "multiclass",
    "shuttle": "multiclass",
    "kin8nm": "regression",
    "house_16H": "regression",
    "bodyfat": "regression",
}

# EXP-1.1 sanity check subset (legacy, kept for backward compatibility)
EXP1_DATASETS = ["adult", "diabetes", "phoneme", "credit-g", "breast_cancer"]

# ── Final 10-dataset benchmark (experiment_standards.md §1.1) ─────────────
# "beijing" and "news" must be loaded via download_datasets.py (UCI source),
# then passed as raw DataFrames; they are NOT loadable via DatasetLoader.load().
MAIN_DATASETS = [
    # Large (>10K rows)
    "adult",         # Binary Clf,  ~48K rows
    "default",       # Binary Clf,  ~30K rows
    "australian",    # Binary Clf,  690 rows (OpenML 40981)
    "magic",         # Binary Clf,  ~19K rows
    # "california_housing" and "news" loaded separately (sklearn/UCI, regression)
    # Small (<10K rows)
    "breast_cancer", # Binary Clf,  ~569 rows
    "diabetes",      # Binary Clf,  ~768 rows
    "credit-g",      # Binary Clf,  ~1K rows
    "kin8nm",        # Regression,  ~8.2K rows (robot arm kinematics, 8 float features)
]

MAIN_DATASETS_CLF = ["adult", "default", "australian", "magic",
                     "breast_cancer", "diabetes", "credit-g"]
MAIN_DATASETS_REG = ["kin8nm"]  # + california_housing, news (loaded externally)

# Legacy EXP-2 dataset list (kept for backward compatibility)
EXP2_DATASETS = [
    "breast_cancer", "vehicle", "credit-g", "bodyfat",
    "segment", "phoneme", "kin8nm", "adult", "house_16H", "shuttle",
]


def get_task_type(name: str) -> str:
    """Return the task type for a registered dataset name.

    Parameters
    ----------
    name:
        Dataset name, must be in DATASET_TASK_TYPE.

    Returns
    -------
    One of "binary", "multiclass", or "regression".
    """
    if name not in DATASET_TASK_TYPE:
        raise ValueError(
            f"Unknown task type for dataset '{name}'. "
            f"Add it to DATASET_TASK_TYPE. Available: {sorted(DATASET_TASK_TYPE)}"
        )
    return DATASET_TASK_TYPE[name]


class DatasetLoader:
    """Loads, preprocesses, and caches OpenML datasets."""

    def __init__(
        self,
        cache_dir: Path | str,
        train_size: float = 0.8,
        random_state: int = 42,
    ):
        self.cache_dir = Path(cache_dir)
        self.train_size = train_size
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset by name.

        Routes to local CSV loading for california_housing and news,
        and to OpenML for all other registered datasets.
        """
        if name in _LOCAL_DATASETS:
            return self._load_local_csv(name)
        if name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}"
            )
        return self.load_by_id(DATASET_REGISTRY[name], name=name)

    def load_by_id(
        self, openml_id: int, name: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset by OpenML ID, using disk cache when available."""
        tag = name or str(openml_id)
        train_path = self.cache_dir / f"{openml_id}_train.npz"
        test_path = self.cache_dir / f"{openml_id}_test.npz"

        if train_path.exists() and test_path.exists():
            log.info(f"Loading {tag} from cache ({train_path})")
            train = np.load(train_path, allow_pickle=True)
            test = np.load(test_path, allow_pickle=True)
            return train["X"], test["X"], train["y"], test["y"]

        log.info(f"Downloading {tag} from OpenML (id={openml_id})")
        X_train, X_test, y_train, y_test = self._download_and_preprocess(
            openml_id, name=name
        )
        self._save_to_cache(openml_id, X_train, X_test, y_train, y_test, name=name)
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_local_csv(
        self, name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load a local CSV dataset (california_housing or news) with npz caching.

        Cache key is name-based (e.g. california_housing_train.npz) since
        these datasets have no OpenML ID.
        """
        train_path = self.cache_dir / f"{name}_train.npz"
        test_path = self.cache_dir / f"{name}_test.npz"

        if train_path.exists() and test_path.exists():
            log.info(f"Loading {name} from cache ({train_path})")
            train = np.load(train_path, allow_pickle=True)
            test = np.load(test_path, allow_pickle=True)
            return train["X"], test["X"], train["y"], test["y"]

        # Locate CSV: prefer scratch, fall back to projappl mirror
        csv_path = _RAW_DATA_DIR / f"{name}.csv"
        if not csv_path.exists():
            csv_path = _RAW_DATA_DIR_FALLBACK / f"{name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV for '{name}' not found at {_RAW_DATA_DIR}/{name}.csv. "
                "Run: python scripts/download_datasets.py"
            )

        log.info(f"Loading {name} from CSV ({csv_path})")
        df = pd.read_csv(csv_path)

        target_col = _LOCAL_DATASETS[name]["target_col"]
        y_series = df[target_col]
        X_df = df.drop(columns=[target_col])

        task_type = DATASET_TASK_TYPE.get(name, "regression")
        if task_type == "regression":
            y = y_series.astype(float).values.astype(np.float32)
        else:
            le = LabelEncoder()
            y = le.fit_transform(y_series.astype(str))

        ordinal = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        vectorizer = TableVectorizer(
            low_cardinality=ordinal,
            high_cardinality=ordinal,
        )
        X_encoded = vectorizer.fit_transform(X_df)
        if hasattr(X_encoded, "toarray"):
            X_encoded = X_encoded.toarray()
        X = np.asarray(X_encoded, dtype=np.float32)

        col_medians = np.nanmedian(X, axis=0)
        nan_mask = ~np.isfinite(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        # Regression: no stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_size, random_state=self.random_state
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(train_path, X=X_train, y=y_train, task_type=np.array([task_type]))
        np.savez_compressed(test_path, X=X_test, y=y_test, task_type=np.array([task_type]))
        log.info(f"Cached {name} ({task_type}) → {train_path}")
        return X_train, X_test, y_train, y_test

    def _download_and_preprocess(
        self, openml_id: int, name: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Determine task type (default to classification if unknown)
        task_type = DATASET_TASK_TYPE.get(name, "binary") if name else "binary"

        dataset = openml.datasets.get_dataset(
            openml_id,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=False,
        )
        X_df, y_series, _, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format="dataframe",
        )
        X_df = X_df.reset_index(drop=True)
        y_series = y_series.reset_index(drop=True)

        if task_type == "regression":
            # For regression: keep y as float32, no LabelEncoder
            y = y_series.astype(float).values.astype(np.float32)
        else:
            # For classification: encode target with LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y_series.astype(str))

        # Use skrub TableVectorizer configured for ordinal encoding
        # (no one-hot: tabular foundation models handle their own encoding)
        # skrub 0.7.x renamed the transformer params to low_cardinality / high_cardinality
        ordinal = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        vectorizer = TableVectorizer(
            low_cardinality=ordinal,
            high_cardinality=ordinal,
        )
        X_encoded = vectorizer.fit_transform(X_df)

        # Convert sparse to dense if needed
        if hasattr(X_encoded, "toarray"):
            X_encoded = X_encoded.toarray()
        X = np.asarray(X_encoded, dtype=np.float32)

        # Replace any remaining NaN/Inf with column medians
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = ~np.isfinite(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        if task_type == "regression":
            # Do NOT stratify for regression targets
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=self.train_size,
                random_state=self.random_state,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=self.train_size,
                stratify=y,
                random_state=self.random_state,
            )
        return X_train, X_test, y_train, y_test

    def _save_to_cache(
        self,
        openml_id: int,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        name: str | None = None,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        task_type = DATASET_TASK_TYPE.get(name, "binary") if name else "binary"
        np.savez_compressed(
            self.cache_dir / f"{openml_id}_train.npz",
            X=X_train,
            y=y_train,
            task_type=np.array([task_type]),
        )
        np.savez_compressed(
            self.cache_dir / f"{openml_id}_test.npz",
            X=X_test,
            y=y_test,
            task_type=np.array([task_type]),
        )
        log.info(f"Cached {openml_id} ({task_type}) to {self.cache_dir}")
