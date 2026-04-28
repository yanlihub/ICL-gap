"""
Script: run_quality_spectrum.py
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from synthetic_context.data.loader import DatasetLoader, DATASET_REGISTRY, get_task_type
from synthetic_context.data.syn_store import SyntheticDataStore
from synthetic_context.evaluators.icl import evaluate_icl
from synthetic_context.utils.metrics import compute_all_fidelity_metrics
from synthetic_context.utils.seeds import set_all_seeds

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("exp_quality_spectrum")

# ── Constants ────────────────────────────────────────────────────────────────

EXP_ID = "EXP-A-"

RESULTS_BASE = Path("./results")
DATA_DIR = Path("./data")

ALL_DATASETS = [
    "breast_cancer", "diabetes", "credit-g",          # small, binary
    "kin8nm",                                          # small, regression
    "shoppers", "magic",                               # medium, binary
    "california_housing",                              # medium, regression
    "default", "adult",                                # large, binary
    "news",                                            # large, regression
]

ALL_GENERATORS = [
    "random", "marginal", "gmm", "smote",
    "ctgan", "tvae", "tabsyn", "tabpfngen",
]

ALL_SEEDS = [0, 1, 2, 3, 4]
ICL_MODELS = ["tabpfn", "tabicl"]

# Generator quality level (for ordering in plots)
GENERATOR_LEVEL: dict[str, int] = {
    "random": 0, "marginal": 1, "gmm": 2, "smote": 3,
    "ctgan": 4, "tvae": 5, "tabsyn": 6, "tabpfngen": 7,
}

# ── NEW metric key names (no ambiguous fid_ prefix) ─────────────────────────

# These are the 14 fidelity metric keys produced by compute_all_fidelity_metrics.
# We remap from old `fid_*` names to clear, unambiguous names.
#
# For TSTR metrics:
#   - Classification: tstr_xgb_auc, tstr_mlp_auc, tstr_rf_auc
#   - Regression:     tstr_xgb_r2,  tstr_mlp_r2,  tstr_rf_r2
#                     tstr_xgb_rmse, tstr_mlp_rmse, tstr_rf_rmse

_FIDELITY_RENAME_CLF: dict[str, str] = {
    "fid_ks":              "ks",
    "fid_jsd":             "jsd",
    "fid_wd_1d":           "wd_1d",
    "fid_wd_sliced":       "swd",
    "fid_mmd":             "mmd",
    "fid_nfn":             "nfn",
    "fid_tstr_auc":        "tstr_xgb_auc",
    "fid_tstr_mlp":        "tstr_mlp_auc",
    "fid_tstr_rf":         "tstr_rf_auc",
    "fid_tstr_rmse_xgb":   "tstr_xgb_rmse",   # NaN for classification
    "fid_tstr_rmse_mlp":   "tstr_mlp_rmse",    # NaN for classification
    "fid_tstr_rmse_rf":    "tstr_rf_rmse",      # NaN for classification
    "fid_alpha_precision": "alpha_precision",
    "fid_beta_recall":     "beta_recall",
}

_FIDELITY_RENAME_REG: dict[str, str] = {
    "fid_ks":              "ks",
    "fid_jsd":             "jsd",
    "fid_wd_1d":           "wd_1d",
    "fid_wd_sliced":       "swd",
    "fid_mmd":             "mmd",
    "fid_nfn":             "nfn",
    "fid_tstr_auc":        "tstr_xgb_r2",      # regression: this field stores R²
    "fid_tstr_mlp":        "tstr_mlp_r2",       # regression: this field stores R²
    "fid_tstr_rf":         "tstr_rf_r2",        # regression: this field stores R²
    "fid_tstr_rmse_xgb":   "tstr_xgb_rmse",
    "fid_tstr_rmse_mlp":   "tstr_mlp_rmse",
    "fid_tstr_rmse_rf":    "tstr_rf_rmse",
    "fid_alpha_precision": "alpha_precision",
    "fid_beta_recall":     "beta_recall",
}


def rename_fidelity(raw: dict[str, float], task_type: str) -> dict[str, float]:
    """Rename fidelity metric keys from old fid_* to clear names."""
    rmap = _FIDELITY_RENAME_REG if task_type == "regression" else _FIDELITY_RENAME_CLF
    out: dict[str, float] = {}
    for old_key, new_key in rmap.items():
        out[new_key] = raw.get(old_key, float("nan"))
    return out


def primary_metric_name(task_type: str) -> str:
    """Return the primary ICL metric name for logging."""
    return "R²" if task_type == "regression" else "AUC"


def primary_metric_value(m: dict, task_type: str) -> float:
    """Extract the primary quality metric from ICL evaluation result."""
    if task_type == "regression":
        return float(m.get("r2", float("nan")))
    return float(m.get("roc_auc", float("nan")))


def get_git_commit() -> str:
    """Get current git commit hash for provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:12]
    except Exception:
        return "unknown"


# ── Main experiment ──────────────────────────────────────────────────────────

def run_experiment(
    datasets: list[str],
    generators: list[str],
    icl_models: list[str],
    seeds: list[int],
    device: str,
    results_dir: Path | None = None,
) -> list[dict]:
    loader = DatasetLoader(cache_dir=DATA_DIR)
    store = SyntheticDataStore()
    results: list[dict] = []

    for dataset_name in datasets:
        log.info(f"\n{'='*65}")
        log.info(f"Dataset: {dataset_name}")

        try:
            X_train, X_test, y_train, y_test = loader.load(dataset_name)
        except Exception as e:
            log.error(f"Failed to load {dataset_name}: {e}")
            continue

        task_type = get_task_type(dataset_name)
        pm_name = primary_metric_name(task_type)
        openml_id = DATASET_REGISTRY.get(dataset_name, -1)
        n_train = len(y_train)
        n_features = X_train.shape[1]
        log.info(f"  task={task_type}  n_train={n_train}  n_test={len(y_test)}  n_feat={n_features}")

        record_base = {
            "exp_id":     EXP_ID,
            "dataset":    dataset_name,
            "openml_id":  openml_id,
            "n_train":    n_train,
            "n_test":     len(y_test),
            "n_features": n_features,
            "task_type":  task_type,
        }

        # ── Real baseline: D_train as context ────────────────────────────
        real_pm: dict[tuple[str, int], float] = {}  # (model, seed) → primary metric

        for model in icl_models:
            log.info(f"  [REAL] model={model}")
            for seed in seeds:
                set_all_seeds(seed)
                t0 = time.perf_counter()
                try:
                    m = evaluate_icl(
                        X_train, y_train, X_test, y_test,
                        model=model, seed=seed, device=device,
                        task_type=task_type,
                    )
                    pm_val = primary_metric_value(m, task_type)
                except Exception as e:
                    log.warning(f"    real eval failed: {e}")
                    m = {}
                    pm_val = float("nan")
                wall_s = round(time.perf_counter() - t0, 3)

                real_pm[(model, seed)] = pm_val
                log.info(f"    seed={seed}  {pm_name}={pm_val:.4f}  t={wall_s:.1f}s")

                results.append({
                    **record_base,
                    "generator":  "real",
                    "icl_model":  model,
                    "seed":       seed,
                    "syn_source": "real_train",
                    # ICL metrics
                    "icl_primary_metric": pm_name,
                    "icl_primary_value":  pm_val,
                    "icl_gap":            0.0,
                    # Full ICL result
                    **{f"icl_{k}": v for k, v in m.items()},
                    "wall_time_s":        wall_s,
                })

        # ── Synthetic: load from SyntheticDataStore ─────────────────────
        for gen_name in generators:
            log.info(f"\n  [GEN] {gen_name}")

            if not store.all_exist(gen_name, dataset_name, len(seeds)):
                log.warning(
                    f"  {gen_name} × {dataset_name}: cache incomplete "
                    f"(need {len(seeds)} samples) — emitting NaN records"
                )
                for seed in seeds:
                    for model in icl_models:
                        results.append({
                            **record_base,
                            "generator": gen_name, "icl_model": model,
                            "seed": seed, "syn_source": "cache_missing",
                            "icl_primary_metric": pm_name,
                            "icl_primary_value": float("nan"),
                            "icl_gap": float("nan"),
                            "wall_time_s": float("nan"),
                        })
                continue

            for seed in seeds:
                set_all_seeds(seed)

                # Load synthetic data (seed → sample_idx)
                try:
                    X_syn, y_syn = store.load(gen_name, dataset_name, seed)
                except Exception as e:
                    log.warning(f"    {gen_name} seed={seed} load failed: {e}")
                    for model in icl_models:
                        results.append({
                            **record_base,
                            "generator": gen_name, "icl_model": model,
                            "seed": seed, "syn_source": "cache_load_error",
                            "icl_primary_metric": pm_name,
                            "icl_primary_value": float("nan"),
                            "icl_gap": float("nan"),
                            "wall_time_s": float("nan"),
                        })
                    continue

                n_syn = len(y_syn)

                # ── Fidelity metrics (once per sample, shared across ICL models)
                log.info(f"    {gen_name} seed={seed}  n_syn={n_syn}  computing fidelity ...")
                t_fid = time.perf_counter()
                raw_fidelity = compute_all_fidelity_metrics(
                    X_train, X_syn, y_syn, X_test, y_test,
                    seed=seed, task_type=task_type,
                )
                fidelity = rename_fidelity(raw_fidelity, task_type)
                fid_time = round(time.perf_counter() - t_fid, 2)

                fid_summary = "  ".join(
                    f"{k}={v:.3f}" for k, v in fidelity.items()
                    if isinstance(v, float) and np.isfinite(v)
                )
                log.info(f"    fidelity ({fid_time:.1f}s): {fid_summary}")

                # ── ICL evaluation per model ────────────────────────────
                for model in icl_models:
                    t0 = time.perf_counter()
                    try:
                        m = evaluate_icl(
                            X_syn, y_syn, X_test, y_test,
                            model=model, seed=seed, device=device,
                            task_type=task_type,
                        )
                        pm_syn = primary_metric_value(m, task_type)
                    except Exception as e:
                        log.warning(f"    {gen_name} seed={seed} model={model} ICL failed: {e}")
                        m = {}
                        pm_syn = float("nan")
                    wall_s = round(time.perf_counter() - t0, 3)

                    pm_real = real_pm.get((model, seed), float("nan"))
                    icl_gap = (
                        float(pm_real - pm_syn)
                        if np.isfinite(pm_real) and np.isfinite(pm_syn)
                        else float("nan")
                    )
                    log.info(
                        f"    {gen_name} seed={seed} model={model}  "
                        f"{pm_name}_syn={pm_syn:.4f}  gap={icl_gap:+.4f}"
                    )

                    results.append({
                        **record_base,
                        "generator":          gen_name,
                        "icl_model":          model,
                        "seed":               seed,
                        "syn_source":         "SyntheticDataStore",
                        "n_syn":              n_syn,
                        # ICL metrics
                        "icl_primary_metric": pm_name,
                        "icl_primary_value":  pm_syn,
                        "icl_gap":            icl_gap,
                        # Full ICL result
                        **{f"icl_{k}": v for k, v in m.items()},
                        # Fidelity (shared across ICL models for same sample)
                        **fidelity,
                        "fidelity_time_s":    fid_time,
                        "wall_time_s":        wall_s,
                    })

            # ── Checkpoint after each (dataset, generator) ──────────────
            if results_dir is not None:
                _checkpoint(results, results_dir / "results_partial.json")
                log.info(f"  [checkpoint] {len(results)} records saved")

    return results


def _checkpoint(records: list[dict], path: Path) -> None:
    """Save intermediate results for crash recovery."""
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=_json_default)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP-A : Generator Quality Spectrum (canonical rewrite)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets", nargs="*", default=ALL_DATASETS,
                    help="Datasets to evaluate")
    p.add_argument("--generators", nargs="*", default=ALL_GENERATORS,
                    help="Generators to evaluate")
    p.add_argument("--icl_models", nargs="*", default=ICL_MODELS,
                    help="ICL models to evaluate")
    p.add_argument("--seeds", type=int, nargs="*", default=ALL_SEEDS,
                    help="Random seeds (also used as SyntheticDataStore sample indices)")
    p.add_argument("--device", default="cuda",
                    help="Device for ICL models (cuda or cpu)")
    p.add_argument("--out_dir", default=None,
                    help="Override output directory (default: auto-generated)")
    p.add_argument("--smoke_test", action="store_true",
                    help="Run with minimal config for quick validation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        log.info(" SMOKE TEST MODE — minimal configuration")
        if args.datasets == ALL_DATASETS:
            args.datasets = ["breast_cancer"]
        if args.generators == ALL_GENERATORS:
            args.generators = ["marginal"]
        if args.seeds == ALL_SEEDS:
            args.seeds = [0]

    # Build config dict for provenance
    config = {
        "exp_id":      EXP_ID,
        "datasets":    args.datasets,
        "generators":  args.generators,
        "icl_models":  args.icl_models,
        "seeds":       args.seeds,
        "device":      args.device,
        "smoke_test":  args.smoke_test,
        # Provenance
        "script":      str(Path(__file__).resolve()),
        "git_commit":  get_git_commit(),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "python":      sys.version,
        "hostname":    platform.node(),
        "syn_store":   str(SyntheticDataStore().base_dir),
        "data_dir":    str(DATA_DIR),
        # Metric definitions
        "icl_gap_clf": "AUC(real) - AUC(syn)",
        "icl_gap_reg": "R2(real) - R2(syn)",
        "tstr_clf":    "tstr_{xgb,mlp,rf}_auc = AUC (higher is better)",
        "tstr_reg":    "tstr_{xgb,mlp,rf}_r2 = R² (higher is better); tstr_*_rmse also recorded",
    }

    log.info("=" * 70)
    log.info(f"{EXP_ID}: Generator Quality Spectrum")
    log.info(f"  Datasets   : {args.datasets}")
    log.info(f"  Generators : {args.generators}")
    log.info(f"  ICL models : {args.icl_models}")
    log.info(f"  Seeds      : {args.seeds}")
    log.info(f"  Device     : {args.device}")
    log.info(f"  Git commit : {config['git_commit']}")
    log.info("=" * 70)

    # Run experiment
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_BASE / "exp_quality_spectrum_v3" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_experiment(
        datasets=args.datasets,
        generators=args.generators,
        icl_models=args.icl_models,
        seeds=args.seeds,
        device=args.device,
        results_dir=out_dir,
    )

    # ── Save results ─────────────────────────────────────────────────────
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    log.info(f"Results → {results_path}  ({len(results)} records)")

    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=_json_default)
    log.info(f"Config  → {config_path}")

    # Remove partial checkpoint
    partial = out_dir / "results_partial.json"
    if partial.exists():
        partial.unlink()

    # ── Quick summary ────────────────────────────────────────────────────
    _print_summary(results)

    log.info(f"\nAll outputs → {out_dir}")


def _print_summary(results: list[dict]) -> None:
    """Print a quick ICL-Gap summary table."""
    from collections import defaultdict

    gaps: dict[str, list[float]] = defaultdict(list)
    for r in results:
        gen = r.get("generator", "")
        gap = r.get("icl_gap")
        if gen != "real" and gap is not None and np.isfinite(gap):
            gaps[gen].append(gap)

    if not gaps:
        return

    print(f"\n{'='*60}")
    print(f"{EXP_ID} — ICL-Gap Summary (mean ± std across all datasets × seeds × models)")
    print(f"{'='*60}")
    print(f"{'Generator':12s}  {'mean gap':>10s}  {'std':>8s}  {'n':>4s}")
    print("-" * 40)

    for gen in ALL_GENERATORS:
        vals = gaps.get(gen, [])
        if vals:
            arr = np.array(vals)
            print(f"{gen:12s}  {arr.mean():>10.4f}  {arr.std():>8.4f}  {len(arr):>4d}")


if __name__ == "__main__":
    main()
