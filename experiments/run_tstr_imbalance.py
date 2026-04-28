"""
Script: run_tstr_imbalance.py
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from synthetic_context.data.loader import DatasetLoader, get_task_type
from synthetic_context.data.syn_store import SyntheticDataStore
from synthetic_context.evaluators.icl import evaluate_icl
from synthetic_context.utils.seeds import set_all_seeds

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("exp_tstr_imbalance")

# ── Constants ────────────────────────────────────────────────────────────────

EXP_ID = "EXP-TSTR-IMB-"

RESULTS_BASE = Path("./results")
DATA_DIR = Path("./data")

# 7 binary classification datasets
ALL_DATASETS = [
    "breast_cancer", "diabetes", "credit-g",
    "shoppers", "magic", "default", "adult",
]

# 10 experimental conditions
ALL_GENERATORS = [
    "real",
    "random", "marginal", "gmm",
    "smote", "smote_preserve_rho",
    "ctgan", "tvae", "tabsyn", "tabpfngen",
]

ALL_RHOS = [0.30, 0.15, 0.05, 0.02]
ALL_SEEDS = [0, 1, 2, 3, 4]

# Generators that MUST be fit on D_imb (not cache_downsample):
#   - smote: its rebalancing to 1:1 is the P(Y)-confounding mechanism
#   - smote_preserve_rho: ablation that preserves P(Y) after SMOTE fit
FIT_ON_IMB_GENERATORS = {"smote", "smote_preserve_rho"}

# Generators that use cache_downsample from SyntheticDataStore
CACHE_GENERATORS = {"random", "marginal", "gmm", "ctgan", "tvae", "tabsyn", "tabpfngen"}

# Max context size for TFM evaluation (consistent with EXP-A)
TABPFN_MAX_CTX = 10_000


# ── Imbalance construction ──────────────────────────────────────────────────

def make_controlled_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    target_rho: float,
    seed: int,
    min_samples_per_class: int = 5,
    minority_class: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Subsample to minority-class fraction ≈ target_rho.

    When minority_class is provided (cache-downsample path), the caller pins
    the minority label so real and synthetic contexts share the same positive class.

    Returns (X_imb, y_imb, meta_dict).
    """
    classes, counts = np.unique(y, return_counts=True)
    if minority_class is None:
        order = np.argsort(counts)
        minority_class = int(classes[order[0]])
        majority_class = int(classes[order[-1]])
    else:
        minority_class = int(minority_class)
        non_min = [(int(c), int(n)) for c, n in zip(classes, counts) if int(c) != minority_class]
        if not non_min:
            raise ValueError(f"No non-minority class (minority_class={minority_class})")
        majority_class = max(non_min, key=lambda cn: cn[1])[0]

    n_min_orig = int(np.sum(y == minority_class))
    n_maj_orig = int(np.sum(y == majority_class))
    rho_orig = n_min_orig / max(n_min_orig + n_maj_orig, 1)

    if target_rho <= rho_orig:
        n_maj_new = n_maj_orig
        n_min_new = int(round(target_rho * n_maj_orig / max(1.0 - target_rho, 1e-6)))
        n_min_new = max(min_samples_per_class, min(n_min_new, n_min_orig))
    else:
        n_min_new = n_min_orig
        n_maj_new = int(round(n_min_orig * (1.0 - target_rho) / max(target_rho, 1e-6)))
        n_maj_new = max(min_samples_per_class, min(n_maj_new, n_maj_orig))

    rng = np.random.RandomState(seed)
    idx_min = np.where(y == minority_class)[0]
    idx_maj = np.where(y == majority_class)[0]
    keep_min = rng.choice(idx_min, n_min_new, replace=False)
    keep_maj = rng.choice(idx_maj, n_maj_new, replace=False)
    idx_kept = np.concatenate([keep_maj, keep_min])
    rng.shuffle(idx_kept)

    meta = {
        "minority_class": minority_class,
        "majority_class": majority_class,
        "n_minority": n_min_new,
        "n_majority": n_maj_new,
        "n_total": n_min_new + n_maj_new,
        "rho_original": float(rho_orig),
        "rho_actual": float(n_min_new / (n_min_new + n_maj_new)),
    }
    return X[idx_kept], y[idx_kept], meta


# ── Synthetic data builders ─────────────────────────────────────────────────

def build_syn_cache_downsample(
    gen_name: str,
    dataset: str,
    seed: int,
    rho: float,
    minority_class: int,
    store: SyntheticDataStore,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load from SyntheticDataStore and downsample to target ρ."""
    sample_idx = seed % 5
    X_cache, y_cache = store.load(gen_name, dataset, sample_idx)
    y_cache = y_cache.astype(int)

    X_syn, y_syn, _ = make_controlled_imbalance(
        X_cache, y_cache, rho, seed, minority_class=minority_class,
    )
    return X_syn, y_syn, "cache_downsample"


def build_syn_smote(
    X_imb: np.ndarray,
    y_imb: np.ndarray,
    seed: int,
    minority_class: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """SMOTE fit on D_imb: default rebalancing to 1:1 (the P(Y)-confounding source)."""
    from imblearn.over_sampling import SMOTE as _SMOTE

    min_count = int(np.sum(y_imb == minority_class))
    k = min(5, max(1, min_count - 1))
    sm = _SMOTE(k_neighbors=k, random_state=seed)
    X_syn, y_syn = sm.fit_resample(X_imb, y_imb)
    return X_syn, y_syn, "fit_on_imb_rebalanced"


def build_syn_smote_preserve_rho(
    X_imb: np.ndarray,
    y_imb: np.ndarray,
    seed: int,
    rho: float,
    minority_class: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """SMOTE fit on D_imb, then resample output at original ρ (ablation).

    This isolates SMOTE's interpolation quality from its P(Y) distortion.
    """
    from imblearn.over_sampling import SMOTE as _SMOTE

    min_count = int(np.sum(y_imb == minority_class))
    k = min(5, max(1, min_count - 1))
    sm = _SMOTE(k_neighbors=k, random_state=seed)
    X_bal, y_bal = sm.fit_resample(X_imb, y_imb)  # 1:1 oversampled pool

    # Resample from the balanced pool at the original ratio ρ
    rng = np.random.RandomState(seed + 9999)
    n_total = len(X_imb)
    n_min_target = max(1, int(round(n_total * rho)))
    n_maj_target = n_total - n_min_target

    min_pool = np.where(y_bal == minority_class)[0]
    maj_pool = np.where(y_bal != minority_class)[0]
    idx_min = rng.choice(min_pool, n_min_target, replace=n_min_target > len(min_pool))
    idx_maj = rng.choice(maj_pool, n_maj_target, replace=n_maj_target > len(maj_pool))
    idx = np.concatenate([idx_min, idx_maj])
    rng.shuffle(idx)

    return X_bal[idx], y_bal[idx], "fit_on_imb_preserve_rho"


def build_syn(
    gen_name: str,
    X_imb: np.ndarray,
    y_imb: np.ndarray,
    seed: int,
    rho: float,
    dataset: str,
    minority_class: int,
    store: SyntheticDataStore,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Route to the correct synthetic data builder."""
    if gen_name == "real":
        return X_imb, y_imb, "real_imb"

    if gen_name == "smote":
        return build_syn_smote(X_imb, y_imb, seed, minority_class)

    if gen_name == "smote_preserve_rho":
        return build_syn_smote_preserve_rho(X_imb, y_imb, seed, rho, minority_class)

    # All other generators: cache_downsample
    return build_syn_cache_downsample(gen_name, dataset, seed, rho, minority_class, store)


# ── TSTR evaluators ──────────────────────────────────────────────────────────

def _roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if proba.ndim == 2 and proba.shape[1] == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    return float("nan")


def tstr_xgb(X_syn, y_syn, X_test, y_test, seed, balanced=False) -> float:
    """Train XGBoost on synthetic, test on real. Returns AUC."""
    if len(np.unique(y_syn)) < 2:
        return float("nan")
    try:
        from xgboost import XGBClassifier
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=200, random_state=seed)
        clf.fit(X_syn, y_syn)
        return _roc_auc(y_test, clf.predict_proba(X_test))

    kwargs = dict(n_estimators=200, random_state=seed, verbosity=0, eval_metric="logloss")
    if balanced:
        counts = np.bincount(y_syn.astype(int))
        if len(counts) >= 2 and counts[1] > 0:
            kwargs["scale_pos_weight"] = float(counts[0]) / float(counts[1])
    try:
        clf = XGBClassifier(**kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_syn, y_syn)
        return _roc_auc(y_test, clf.predict_proba(X_test))
    except Exception as e:
        log.warning(f"TSTR-XGB failed: {e}")
        return float("nan")


def tstr_rf(X_syn, y_syn, X_test, y_test, seed, balanced=False) -> float:
    """Train RandomForest on synthetic, test on real. Returns AUC."""
    from sklearn.ensemble import RandomForestClassifier
    if len(np.unique(y_syn)) < 2:
        return float("nan")
    kwargs = dict(n_estimators=200, random_state=seed, n_jobs=4)
    if balanced:
        kwargs["class_weight"] = "balanced"
    try:
        clf = RandomForestClassifier(**kwargs)
        clf.fit(X_syn, y_syn)
        return _roc_auc(y_test, clf.predict_proba(X_test))
    except Exception as e:
        log.warning(f"TSTR-RF failed: {e}")
        return float("nan")


def tstr_mlp(X_syn, y_syn, X_test, y_test, seed) -> float:
    """Train MLP on synthetic, test on real. Returns AUC."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y_syn)) < 2:
        return float("nan")
    try:
        scaler = StandardScaler()
        X_syn_s = scaler.fit_transform(X_syn)
        X_test_s = scaler.transform(X_test)
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500,
            random_state=seed, early_stopping=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_syn_s, y_syn)
        return _roc_auc(y_test, clf.predict_proba(X_test_s))
    except Exception as e:
        log.warning(f"TSTR-MLP failed: {e}")
        return float("nan")


def _clip_context(X, y, seed, max_ctx=TABPFN_MAX_CTX):
    """Subsample to max_ctx if needed (keeps TFM memory-safe)."""
    if len(X) <= max_ctx:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), max_ctx, replace=False)
    return X[idx], y[idx]


def icl_auc(X_ctx, y_ctx, X_test, y_test, seed, device, model="tabpfn") -> float:
    """Evaluate ICL model and return AUC."""
    X_c, y_c = _clip_context(X_ctx, y_ctx, seed)
    try:
        m = evaluate_icl(X_c, y_c, X_test, y_test,
                         model=model, seed=seed, device=device, task_type="binary")
        return float(m.get("roc_auc", float("nan")))
    except Exception as e:
        log.warning(f"ICL-{model} failed: {e}")
        return float("nan")


# ── Main experiment ──────────────────────────────────────────────────────────

def evaluate_all(
    X_syn: np.ndarray, y_syn: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    seed: int, device: str,
) -> Dict[str, float]:
    """Evaluate D_syn with all 7 evaluators. Returns metrics dict."""
    m: Dict[str, float] = {}

    # ICL evaluators
    m["icl_tabpfn_auc"] = icl_auc(X_syn, y_syn, X_test, y_test, seed, device, "tabpfn")
    m["icl_tabicl_auc"] = icl_auc(X_syn, y_syn, X_test, y_test, seed, device, "tabicl")

    # TSTR evaluators
    m["tstr_xgb_naive_auc"]    = tstr_xgb(X_syn, y_syn, X_test, y_test, seed, balanced=False)
    m["tstr_xgb_balanced_auc"] = tstr_xgb(X_syn, y_syn, X_test, y_test, seed, balanced=True)
    m["tstr_rf_naive_auc"]     = tstr_rf(X_syn, y_syn, X_test, y_test, seed, balanced=False)
    m["tstr_rf_balanced_auc"]  = tstr_rf(X_syn, y_syn, X_test, y_test, seed, balanced=True)
    m["tstr_mlp_naive_auc"]    = tstr_mlp(X_syn, y_syn, X_test, y_test, seed)

    return m


def run_experiment(
    datasets: List[str],
    generators: List[str],
    rhos: List[float],
    seeds: List[int],
    device: str,
    results_dir: Path | None = None,
) -> List[Dict]:
    loader = DatasetLoader(cache_dir=DATA_DIR)
    store = SyntheticDataStore()
    records: List[Dict] = []

    for dataset_name in datasets:
        log.info(f"\n{'='*65}")
        log.info(f"Dataset: {dataset_name}")

        try:
            X_train, X_test, y_train, y_test = loader.load(dataset_name)
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        except Exception as e:
            log.error(f"Failed to load {dataset_name}: {e}")
            continue

        task_type = get_task_type(dataset_name)
        if task_type == "regression":
            log.warning(f"  Skipping {dataset_name}: regression dataset")
            continue
        n_train = len(y_train)
        log.info(f"  n_train={n_train}  n_test={len(y_test)}")

        # Verify cache availability for cache generators
        cache_gens_in_run = [g for g in generators if g in CACHE_GENERATORS]
        for g in cache_gens_in_run:
            if not store.all_exist(g, dataset_name, n_samples=5):
                log.warning(f"  Cache missing for {g} × {dataset_name}")

        for rho in rhos:
            # Create imbalanced real training set
            X_imb, y_imb, imb_meta = make_controlled_imbalance(
                X_train, y_train, rho, seed=0  # seed=0 for deterministic D_imb per ρ
            )
            minority_class = int(imb_meta["minority_class"])

            log.info(f"\n  ρ={rho:.2f}  n_imb={imb_meta['n_total']}  "
                     f"n_min={imb_meta['n_minority']}  n_maj={imb_meta['n_majority']}")

            # ── Compute real baseline AUC for ICL-Gap ────────────────────
            real_auc: Dict[str, Dict[int, float]] = defaultdict(dict)

            for seed in seeds:
                set_all_seeds(seed)

                for gen_name in generators:
                    t0 = time.perf_counter()

                    # Re-create D_imb with per-seed randomization for SMOTE
                    # (real D_imb uses seed for consistency)
                    X_imb_s, y_imb_s, imb_meta_s = make_controlled_imbalance(
                        X_train, y_train, rho, seed,
                    )
                    minority_class_s = int(imb_meta_s["minority_class"])

                    try:
                        X_syn, y_syn, syn_source = build_syn(
                            gen_name, X_imb_s, y_imb_s, seed,
                            rho, dataset_name, minority_class_s, store,
                        )
                    except Exception as e:
                        log.warning(f"    {gen_name} seed={seed} build failed: {e}")
                        records.append({
                            "exp_id": EXP_ID,
                            "dataset": dataset_name,
                            "rho_target": float(rho),
                            "generator": gen_name,
                            "seed": seed,
                            "failed": True,
                            "error": str(e),
                            **{k: v for k, v in imb_meta_s.items()},
                        })
                        continue

                    # Compute minority fraction in synthetic data
                    n_syn = len(y_syn)
                    min_frac_syn = float(np.mean(y_syn == minority_class_s))

                    log.info(
                        f"    {gen_name:20s} seed={seed} n_syn={n_syn:5d} "
                        f"min_frac={min_frac_syn:.3f} src={syn_source}"
                    )

                    # Evaluate with all evaluators
                    metrics = evaluate_all(X_syn, y_syn, X_test, y_test, seed, device)
                    wall_s = round(time.perf_counter() - t0, 2)

                    # Store real AUC for ICL-Gap computation
                    if gen_name == "real":
                        real_auc["tabpfn"][seed] = metrics["icl_tabpfn_auc"]
                        real_auc["tabicl"][seed] = metrics["icl_tabicl_auc"]
                        for tstr_key in ["tstr_xgb_naive_auc", "tstr_xgb_balanced_auc",
                                         "tstr_rf_naive_auc", "tstr_rf_balanced_auc",
                                         "tstr_mlp_naive_auc"]:
                            real_auc[tstr_key][seed] = metrics[tstr_key]

                    # Compute gaps
                    icl_gap_tabpfn = _gap(real_auc["tabpfn"].get(seed), metrics["icl_tabpfn_auc"])
                    icl_gap_tabicl = _gap(real_auc["tabicl"].get(seed), metrics["icl_tabicl_auc"])
                    tstr_gap_xgb_naive = _gap(
                        real_auc["tstr_xgb_naive_auc"].get(seed), metrics["tstr_xgb_naive_auc"])
                    tstr_gap_xgb_balanced = _gap(
                        real_auc["tstr_xgb_balanced_auc"].get(seed), metrics["tstr_xgb_balanced_auc"])

                    records.append({
                        "exp_id":             EXP_ID,
                        "dataset":            dataset_name,
                        "rho_target":         float(rho),
                        "rho_actual":         float(imb_meta_s["rho_actual"]),
                        "generator":          gen_name,
                        "seed":               seed,
                        "syn_source":         syn_source,
                        "n_imb":              int(imb_meta_s["n_total"]),
                        "n_minority":         int(imb_meta_s["n_minority"]),
                        "n_majority":         int(imb_meta_s["n_majority"]),
                        "minority_class":     int(minority_class_s),
                        "n_syn":              n_syn,
                        "minority_frac_syn":  min_frac_syn,
                        "failed":             False,
                        # Raw evaluator AUCs
                        **metrics,
                        # Gaps (real - syn; negative = syn beats real = rank inversion)
                        "icl_gap_tabpfn":     icl_gap_tabpfn,
                        "icl_gap_tabicl":     icl_gap_tabicl,
                        "tstr_gap_xgb_naive": tstr_gap_xgb_naive,
                        "tstr_gap_xgb_balanced": tstr_gap_xgb_balanced,
                        "wall_time_s":        wall_s,
                    })

            # Checkpoint after each (dataset, rho)
            if results_dir is not None:
                _checkpoint(records, results_dir / "results_partial.json")
                log.info(f"  [checkpoint] {len(records)} records saved")

    return records


def _gap(real_val: float | None, syn_val: float) -> float:
    """Compute gap = real - syn. Returns NaN if either is missing."""
    if real_val is None or not np.isfinite(real_val) or not np.isfinite(syn_val):
        return float("nan")
    return float(real_val - syn_val)


def _checkpoint(records: List[Dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(records: List[Dict]) -> List[Dict]:
    """Aggregate across seeds per (dataset, generator, rho)."""
    metric_keys = [
        "icl_tabpfn_auc", "icl_tabicl_auc",
        "tstr_xgb_naive_auc", "tstr_xgb_balanced_auc",
        "tstr_rf_naive_auc", "tstr_rf_balanced_auc",
        "tstr_mlp_naive_auc",
        "icl_gap_tabpfn", "icl_gap_tabicl",
        "tstr_gap_xgb_naive", "tstr_gap_xgb_balanced",
    ]
    buckets: Dict[Tuple[str, str, float], List[Dict]] = defaultdict(list)
    for r in records:
        if r.get("failed"):
            continue
        buckets[(r["dataset"], r["generator"], r["rho_target"])].append(r)

    summary = []
    for (ds, gen, rho), rs in sorted(buckets.items()):
        row = {"dataset": ds, "generator": gen, "rho_target": rho, "n_seeds": len(rs)}
        for k in metric_keys:
            vals = np.array([r.get(k, float("nan")) for r in rs], dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{k}_mean"] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
            row[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(row)
    return summary


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP-TSTR-IMBALANCE : P(Y)-Confounding under Class Imbalance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets", nargs="*", default=ALL_DATASETS)
    p.add_argument("--generators", nargs="*", default=ALL_GENERATORS)
    p.add_argument("--rhos", type=float, nargs="*", default=ALL_RHOS)
    p.add_argument("--seeds", type=int, nargs="*", default=ALL_SEEDS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--smoke_test", action="store_true",
                    help="Run with minimal config for quick validation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        log.info(" SMOKE TEST MODE")
        if args.datasets == ALL_DATASETS:
            args.datasets = ["breast_cancer"]
        if args.generators == ALL_GENERATORS:
            args.generators = ["real", "smote", "marginal"]
        if args.rhos == ALL_RHOS:
            args.rhos = [0.15]
        if args.seeds == ALL_SEEDS:
            args.seeds = [0]

    config = {
        "exp_id":         EXP_ID,
        "datasets":       args.datasets,
        "generators":     args.generators,
        "rhos":           args.rhos,
        "seeds":          args.seeds,
        "device":         args.device,
        "smoke_test":     args.smoke_test,
        "script":         str(Path(__file__).resolve()),
        "git_commit":     get_git_commit(),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "python":         sys.version,
        "hostname":       platform.node(),
        # Protocol documentation
        "protocol": {
            "cache_downsample": sorted(CACHE_GENERATORS),
            "fit_on_imb":       sorted(FIT_ON_IMB_GENERATORS),
            "explanation": (
                "cache_downsample: pre-generated from full balanced D_train, "
                "then downsampled to target rho. Isolates evaluator P(Y) sensitivity. "
                "fit_on_imb: SMOTE fit on D_imb(rho). "
                "smote rebalances to 1:1 (P(Y)-confounding source). "
                "smote_preserve_rho resamples at rho (ablation control)."
            ),
        },
        "evaluators": {
            "icl_tabpfn_auc":        "TabPFN  AUC on D_test",
            "icl_tabicl_auc":        "TabICL  AUC on D_test",
            "tstr_xgb_naive_auc":    "XGBoost (no class weight) AUC on D_test",
            "tstr_xgb_balanced_auc": "XGBoost (scale_pos_weight) AUC on D_test",
            "tstr_rf_naive_auc":     "RandomForest (no class weight) AUC on D_test",
            "tstr_rf_balanced_auc":  "RandomForest (class_weight='balanced') AUC on D_test",
            "tstr_mlp_naive_auc":    "MLP (no class weight) AUC on D_test",
        },
        "gap_definition": "gap = AUC(real) - AUC(syn); negative = rank inversion",
    }

    log.info("=" * 70)
    log.info(f"{EXP_ID}: P(Y)-Confounding under Class Imbalance")
    log.info(f"  Datasets   : {args.datasets}")
    log.info(f"  Generators : {args.generators}")
    log.info(f"  ρ values   : {args.rhos}")
    log.info(f"  Seeds      : {args.seeds}")
    log.info(f"  Device     : {args.device}")
    log.info(f"  Git commit : {config['git_commit']}")
    log.info("=" * 70)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_BASE / "exp_tstr_imbalance_v3" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    records = run_experiment(
        datasets=args.datasets,
        generators=args.generators,
        rhos=args.rhos,
        seeds=args.seeds,
        device=args.device,
        results_dir=out_dir,
    )
    summary = aggregate(records)

    # Save
    with open(out_dir / "results.json", "w") as f:
        json.dump(records, f, indent=2, default=_json_default)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=_json_default)

    # Remove partial
    partial = out_dir / "results_partial.json"
    if partial.exists():
        partial.unlink()

    log.info(f"\nResults → {out_dir}")
    log.info(f"  results.json : {len(records)} records")
    log.info(f"  summary.json : {len(summary)} aggregates")

    # Quick summary table
    _print_summary(summary, args.rhos)


def _print_summary(summary: List[Dict], rhos: List[float]) -> None:
    """Print condensed gap table."""
    print(f"\n{'='*80}")
    print(f"{EXP_ID} — ICL-Gap (TabPFN) vs TSTR-Gap (XGB naive) pooled across datasets")
    print(f"{'='*80}")
    print(f"{'Generator':20s}", end="")
    for rho in rhos:
        print(f"  ρ={rho:.2f} ICL/TSTR", end="")
    print()
    print("-" * 80)

    # Pool across datasets
    from collections import defaultdict
    pooled: Dict[Tuple[str, float], List[Tuple[float, float]]] = defaultdict(list)
    for row in summary:
        gen = row["generator"]
        rho = row["rho_target"]
        icl = row.get("icl_gap_tabpfn_mean", float("nan"))
        tstr = row.get("tstr_gap_xgb_naive_mean", float("nan"))
        pooled[(gen, rho)].append((icl, tstr))

    for gen in ALL_GENERATORS:
        if gen == "real":
            continue
        print(f"{gen:20s}", end="")
        for rho in rhos:
            vals = pooled.get((gen, rho), [])
            if vals:
                icl_vals = [v[0] for v in vals if np.isfinite(v[0])]
                tstr_vals = [v[1] for v in vals if np.isfinite(v[1])]
                icl_m = np.mean(icl_vals) if icl_vals else float("nan")
                tstr_m = np.mean(tstr_vals) if tstr_vals else float("nan")
                print(f"  {icl_m:+.3f}/{tstr_m:+.3f}  ", end="")
            else:
                print(f"  {'N/A':>15s}  ", end="")
        print()


if __name__ == "__main__":
    main()
