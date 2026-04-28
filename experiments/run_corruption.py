"""
Script: run_corruption.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from synthetic_context.data.loader import DatasetLoader, DATASET_REGISTRY, get_task_type
from synthetic_context.data.syn_store import SyntheticDataStore
from synthetic_context.evaluators.icl import evaluate_icl
from synthetic_context.utils.corruption import CORRUPTION_REGISTRY
from synthetic_context.utils.seeds import set_all_seeds

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("exp_corruption")

# ── Constants ────────────────────────────────────────────────────────────────

EXP_ID = "EXP-C-"

RESULTS_BASE = Path("./results")
DATA_DIR = Path("./data")

# Max context size (stratified subsample for compute efficiency)
MAX_CTX = 2000

# 7 binary classification datasets
ALL_DATASETS = [
    "breast_cancer", "diabetes", "credit-g",
    "shoppers", "magic", "default", "adult",
]

# 8 synthetic generators (loaded from SyntheticDataStore)
ALL_CONTEXT_GENERATORS = [
    "random", "marginal", "gmm", "smote",
    "ctgan", "tvae", "tabsyn", "tabpfngen",
]

ICL_MODELS = ["tabpfn", "tabicl"]
ALL_SEEDS = [0, 1, 2, 3, 4]

# ── Corruption schedule ──────────────────────────────────────────────────────

CORRUPTION_SCHEDULE: dict[str, list[float]] = {
    "marginal":            [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    "feature_correlation": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
    "label_conditional":   [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    "label_noise":         [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    "class_balance":       [1.0, 0.7,  0.5, 0.3, 0.1, 0.02],  # target_ratio
    "outliers":            [0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
    "shuffle_all":         [1.0],
}

CORRUPTION_LABELS: dict[str, str] = {
    "marginal":            "C1: Marginal noise",
    "feature_correlation": "C2: Feature correlation",
    "label_conditional":   "C3: Label-conditional",
    "label_noise":         "C4: Label noise",
    "class_balance":       "C5: Class balance",
    "outliers":            "C6: Outliers",
    "shuffle_all":         "C7: Shuffle all",
}

# ── Context style definitions (all 9 context types) ───────────────────────

CONTEXT_STYLES: dict[str, dict] = {
    "real":       {"color": "#0072B2", "ls": "-",  "marker": "o",  "label": "Real"},
    "random":     {"color": "#999999", "ls": ":",  "marker": "x",  "label": "Random"},
    "marginal":   {"color": "#E69F00", "ls": "-.", "marker": "D",  "label": "Marginal"},
    "gmm":        {"color": "#56B4E9", "ls": "-.", "marker": "P",  "label": "GMM"},
    "smote":      {"color": "#009E73", "ls": "--", "marker": "v",  "label": "SMOTE"},
    "ctgan":      {"color": "#CC79A7", "ls": "--", "marker": "s",  "label": "CTGAN"},
    "tvae":       {"color": "#D55E00", "ls": ":",  "marker": "^",  "label": "TVAE"},
    "tabsyn":     {"color": "#661100", "ls": "-",  "marker": "h",  "label": "TabSyn"},
    "tabpfngen":  {"color": "#332288", "ls": "-",  "marker": "*",  "label": "TabPFNGen"},
}
CTX_ORDER = list(CONTEXT_STYLES.keys())
MODEL_DISPLAY = {"tabpfn": "TabPFN ", "tabicl": "TabICL "}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _stratified_subsample(
    X: np.ndarray, y: np.ndarray, n: int, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified subsample to at most n samples."""
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    n_per = rng.multinomial(n, probs)
    idx = []
    for cls, k in zip(classes, n_per):
        cls_idx = np.where(y == cls)[0]
        k = min(k, len(cls_idx))
        idx.extend(rng.choice(cls_idx, size=k, replace=False).tolist())
    idx = np.array(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _apply_corruption(
    X: np.ndarray, y: np.ndarray,
    corruption_name: str, raw_level: float, seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply corruption, return (X_c, y_c, stored_level)."""
    fn = CORRUPTION_REGISTRY[corruption_name]
    if corruption_name == "class_balance":
        X_c, y_c = fn(X, y, target_ratio=raw_level, seed=seed)
        stored = round(1.0 - raw_level, 4)  # convert to "imbalance severity"
    elif corruption_name == "shuffle_all":
        X_c, y_c = fn(X, y, seed=seed)
        stored = 1.0
    elif corruption_name == "marginal":
        X_c, y_c = fn(X, y, corruption_level=raw_level, seed=seed)
        stored = raw_level
    elif corruption_name == "feature_correlation":
        X_c, y_c = fn(X, y, fraction=raw_level, seed=seed)
        stored = raw_level
    elif corruption_name == "label_conditional":
        X_c, y_c = fn(X, y, fraction=raw_level, seed=seed)
        stored = raw_level
    elif corruption_name == "label_noise":
        X_c, y_c = fn(X, y, noise_rate=raw_level, seed=seed)
        stored = raw_level
    elif corruption_name == "outliers":
        X_c, y_c = fn(X, y, outlier_fraction=raw_level, seed=seed)
        stored = raw_level
    else:
        raise ValueError(f"Unknown corruption: {corruption_name}")
    return X_c, y_c, stored


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


# ── Main experiment ──────────────────────────────────────────────────────────

def run_experiment(
    datasets: List[str],
    context_generators: List[str],
    icl_models: List[str],
    seeds: List[int],
    device: str,
    results_dir: Path | None = None,
) -> List[Dict]:
    loader = DatasetLoader(cache_dir=DATA_DIR)
    store = SyntheticDataStore()
    results: List[Dict] = []

    for dataset_name in datasets:
        log.info(f"\n{'='*65}")
        log.info(f"Dataset: {dataset_name}")

        try:
            X_train, X_test, y_train, y_test = loader.load(dataset_name)
        except Exception as e:
            log.error(f"Failed to load {dataset_name}: {e}")
            continue

        task_type = get_task_type(dataset_name)
        if task_type == "regression":
            log.warning(f"  Skipping {dataset_name}: regression dataset")
            continue

        openml_id = DATASET_REGISTRY.get(dataset_name, -1)
        log.info(f"  task={task_type}  n_train={len(y_train)}  n_test={len(y_test)}  "
                 f"n_feat={X_train.shape[1]}")

        # ── Build contexts per seed ─────────────────────────────────────
        # For each seed, we use a different synthetic sample (seed → sample_idx)
        # This gives proper variability across seeds.

        for seed in seeds:
            set_all_seeds(seed)

            # Real context (subsampled, same for all seeds within a dataset
            # to isolate evaluator randomness, but with seed-specific subsampling)
            X_real, y_real = _stratified_subsample(X_train, y_train, MAX_CTX, seed=seed)

            contexts: dict[str, Tuple[np.ndarray, np.ndarray]] = {"real": (X_real, y_real)}

            for gen_name in context_generators:
                sample_idx = seed % 5
                if not store.all_exist(gen_name, dataset_name, sample_idx + 1):
                    log.warning(f"  {gen_name} sample {sample_idx}: not in store — skip")
                    continue
                try:
                    X_syn, y_syn = store.load(gen_name, dataset_name, sample_idx)
                    X_syn, y_syn = _stratified_subsample(X_syn, y_syn, MAX_CTX, seed=seed)
                    contexts[gen_name] = (X_syn, y_syn)
                except Exception as e:
                    log.error(f"  {gen_name} load failed: {e}")

            log.info(f"  seed={seed}  contexts: {list(contexts.keys())} "
                     f"({len(contexts)} types)")

            # ── Uncorrupted baselines ───────────────────────────────────
            baselines: dict[Tuple[str, str], float] = {}  # (ctx, model) → AUC

            for ctx_name, (X_ctx, y_ctx) in contexts.items():
                for model in icl_models:
                    t0 = time.perf_counter()
                    try:
                        m = evaluate_icl(X_ctx, y_ctx, X_test, y_test,
                                         model=model, seed=seed, device=device,
                                         task_type="binary")
                        auc = float(m["roc_auc"])
                    except Exception as e:
                        log.warning(f"    baseline {ctx_name} {model} failed: {e}")
                        auc = float("nan")
                    elapsed = round(time.perf_counter() - t0, 2)

                    baselines[(ctx_name, model)] = auc
                    real_auc = baselines.get(("real", model), float("nan"))
                    gap = float(real_auc - auc) if np.isfinite(real_auc) and np.isfinite(auc) else float("nan")

                    log.info(f"    {ctx_name:<12} {model:<8} AUC={auc:.4f}  "
                             f"gap={gap:+.4f}  t={elapsed}s")

                    results.append({
                        "exp_id":           EXP_ID,
                        "dataset":          dataset_name,
                        "openml_id":        openml_id,
                        "context_type":     ctx_name,
                        "corruption_type":  "none",
                        "corruption_level": 0.0,
                        "icl_model":        model,
                        "seed":             seed,
                        "sample_idx":       seed % 5,
                        "n_ctx":            len(y_ctx),
                        "roc_auc":          auc,
                        "corruption_delta": 0.0,
                        "gap_from_real":    gap,
                        "wall_time_s":      elapsed,
                    })

            # ── Corruption sweep ────────────────────────────────────────
            for ctx_name, (X_ctx, y_ctx) in contexts.items():
                log.info(f"\n    [Corruption sweep — {ctx_name}, seed={seed}]")

                for corruption_name, raw_levels in CORRUPTION_SCHEDULE.items():
                    for raw_level in raw_levels:
                        if raw_level == 0.0:
                            continue  # baseline already stored

                        try:
                            X_c, y_c, stored_lv = _apply_corruption(
                                X_ctx, y_ctx, corruption_name, raw_level, seed)
                        except Exception as e:
                            log.warning(f"      {corruption_name}@{raw_level} failed: {e}")
                            continue
                        if len(y_c) == 0:
                            continue

                        for model in icl_models:
                            t0 = time.perf_counter()
                            try:
                                m = evaluate_icl(X_c, y_c, X_test, y_test,
                                                 model=model, seed=seed, device=device,
                                                 task_type="binary")
                                auc = float(m["roc_auc"])
                            except Exception as e:
                                log.warning(f"      {corruption_name}@{stored_lv} "
                                            f"{model} failed: {e}")
                                auc = float("nan")
                            elapsed = round(time.perf_counter() - t0, 2)

                            b_ctx = baselines.get((ctx_name, model), float("nan"))
                            b_real = baselines.get(("real", model), float("nan"))
                            delta = float(b_ctx - auc) if np.isfinite(b_ctx) and np.isfinite(auc) else float("nan")
                            gap = float(b_real - auc) if np.isfinite(b_real) and np.isfinite(auc) else float("nan")

                            log.info(
                                f"      {ctx_name:<12} {corruption_name:<22} "
                                f"lv={stored_lv:.3f} {model:<8} "
                                f"AUC={auc:.4f}  delta={delta:+.4f}"
                            )
                            results.append({
                                "exp_id":           EXP_ID,
                                "dataset":          dataset_name,
                                "openml_id":        openml_id,
                                "context_type":     ctx_name,
                                "corruption_type":  corruption_name,
                                "corruption_level": stored_lv,
                                "icl_model":        model,
                                "seed":             seed,
                                "sample_idx":       seed % 5,
                                "n_ctx":            len(y_c),
                                "roc_auc":          auc,
                                "corruption_delta": delta,
                                "gap_from_real":    gap,
                                "wall_time_s":      elapsed,
                            })

            # Checkpoint after each (dataset, seed)
            if results_dir is not None:
                with open(results_dir / "results_partial.json", "w") as f:
                    json.dump(results, f, indent=2, default=_json_default)
                log.info(f"  [checkpoint] {len(results)} records saved")

    return results


# ── Visualisation ────────────────────────────────────────────────────────────

def _active_contexts(records: List[Dict], synthetic_only: bool = False) -> List[str]:
    """Return ordered list of context types present in records."""
    present = {r["context_type"] for r in records}
    ordered = [c for c in CTX_ORDER if c in present]
    if synthetic_only:
        ordered = [c for c in ordered if c != "real"]
    return ordered


def _aggregate(
    records: List[Dict], corruption_name: str, icl_model: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return {ctx: (levels, mean_delta, std_delta)}, pooled over datasets × seeds."""
    data: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["corruption_type"] != corruption_name or r["icl_model"] != icl_model:
            continue
        data[r["context_type"]][float(r["corruption_level"])].append(
            float(r["corruption_delta"]))
    return {
        ctx: (
            np.array(sorted(lv_dict)),
            np.array([np.nanmean(lv_dict[l]) for l in sorted(lv_dict)]),
            np.array([np.nanstd(lv_dict[l]) for l in sorted(lv_dict)]),
        )
        for ctx, lv_dict in data.items()
    }


def plot_sensitivity_curves(records: List[Dict], out_dir: Path) -> None:
    """Sensitivity curves: AUC-drop vs corruption level, one panel per corruption type."""
    icl_models = sorted({r["icl_model"] for r in records})
    corrupt_types = [c for c in CORRUPTION_SCHEDULE if c != "shuffle_all"]
    n_ds = len({r["dataset"] for r in records})
    n_sd = len({r["seed"] for r in records})

    for model in icl_models:
        fig, axes = plt.subplots(1, len(corrupt_types),
                                 figsize=(3.4 * len(corrupt_types), 4.5),
                                 sharey=False)
        active_ctxs = _active_contexts(records)
        for ax, corr in zip(axes, corrupt_types):
            agg = _aggregate(records, corr, model)
            for ctx_name in active_ctxs:
                if ctx_name not in agg:
                    continue
                s = CONTEXT_STYLES[ctx_name]
                levels, means, stds = agg[ctx_name]
                ax.plot(levels, means, color=s["color"], linestyle=s["ls"],
                        marker=s["marker"], markersize=5, linewidth=1.8,
                        label=s["label"], zorder=3)
                ax.fill_between(levels, means - stds, means + stds,
                                color=s["color"], alpha=0.12, zorder=2)

            # shuffle_all endpoint
            for ctx_name in active_ctxs:
                s = CONTEXT_STYLES[ctx_name]
                sa = [r["corruption_delta"] for r in records
                      if r["corruption_type"] == "shuffle_all"
                      and r["icl_model"] == model
                      and r["context_type"] == ctx_name]
                if sa:
                    ax.axhline(np.nanmean(sa), color=s["color"],
                               linestyle=(0, (2, 4)), linewidth=1.0, alpha=0.45)

            ax.axhline(0, color="gray", linewidth=0.7, linestyle=":")
            ax.set_xlabel(CORRUPTION_LABELS.get(corr, corr), fontsize=7.5)
            ax.set_title(corr.replace("_", "\n"), fontsize=8, pad=3)
            ax.set_ylim(bottom=-0.05)
            if ax is axes[0]:
                ax.set_ylabel("AUC drop from uncorrupted baseline\n(higher = worse)",
                              fontsize=8.5)

        handles = [
            plt.Line2D([0], [0], color=CONTEXT_STYLES[c]["color"],
                       linestyle=CONTEXT_STYLES[c]["ls"],
                       marker=CONTEXT_STYLES[c]["marker"],
                       label=CONTEXT_STYLES[c]["label"])
            for c in active_ctxs
        ]
        handles.append(plt.Line2D([0], [0], color="gray",
                                  linestyle=(0, (2, 4)), linewidth=1.2,
                                  label="shuffle_all endpoint"))
        ncols = min(len(handles), 6)
        fig.legend(handles=handles, loc="upper center", ncol=ncols,
                   fontsize=8, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(
            f"{EXP_ID}  Corruption Sensitivity — {MODEL_DISPLAY.get(model, model)}\n"
            f"mean ± 1σ  ({n_ds} datasets × {n_sd} seeds each)",
            fontsize=9, y=1.07,
        )
        fig.tight_layout()
        path = out_dir / f"sensitivity_curves_{model}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Plot: {path}")


def plot_sensitivity_heatmap(records: List[Dict], out_dir: Path) -> None:
    """Heatmap: rows=context, cols=corruption. Cell=mean AUC drop at max level."""
    icl_models = sorted({r["icl_model"] for r in records})
    ctx_types = _active_contexts(records)
    corrupt_names = list(CORRUPTION_SCHEDULE.keys())

    for model in icl_models:
        matrix = np.full((len(ctx_types), len(corrupt_names)), np.nan)
        for i, ctx in enumerate(ctx_types):
            for j, corr in enumerate(corrupt_names):
                sub = [r for r in records
                       if r["icl_model"] == model
                       and r["context_type"] == ctx
                       and r["corruption_type"] == corr]
                if not sub:
                    continue
                max_lv = max(r["corruption_level"] for r in sub)
                vals = [r["corruption_delta"] for r in sub
                        if r["corruption_level"] == max_lv]
                matrix[i, j] = float(np.nanmean(vals))

        vmax = float(np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 0.5
        fig, ax = plt.subplots(
            figsize=(len(corrupt_names) * 1.6 + 1.5, len(ctx_types) * 0.8 + 2.0))
        im = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("AUC drop at max corruption\n(higher = more sensitive)", fontsize=9)

        ax.set_xticks(range(len(corrupt_names)))
        ax.set_xticklabels([c.replace("_", "\n") for c in corrupt_names], fontsize=8)
        ax.set_yticks(range(len(ctx_types)))
        ax.set_yticklabels([CONTEXT_STYLES.get(c, {}).get("label", c) for c in ctx_types],
                           fontsize=9)
        for i in range(len(ctx_types)):
            for j in range(len(corrupt_names)):
                v = matrix[i, j]
                if not np.isnan(v):
                    tc = "white" if v > vmax * 0.55 else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=8.5, color=tc, fontweight="bold")

        ax.set_title(
            f"Corruption Sensitivity Heatmap — {MODEL_DISPLAY.get(model, model)}\n"
            "(AUC drop at max level; higher = more sensitive)", fontsize=10)
        fig.tight_layout()
        path = out_dir / f"sensitivity_heatmap_{model}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Plot: {path}")


def generate_plots(records: List[Dict], out_dir: Path) -> None:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"\n  Generating plots → {plots_dir}")
    plot_sensitivity_curves(records, plots_dir)
    plot_sensitivity_heatmap(records, plots_dir)
    log.info("  Plots done.")


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(records: List[Dict]) -> None:
    """Print key findings: sensitivity ratio between C1+C2 vs C3."""
    icl_models = sorted({r["icl_model"] for r in records})
    corrupt_types = [c for c in CORRUPTION_SCHEDULE if c != "shuffle_all"]

    print(f"\n{'='*70}")
    print(f"{EXP_ID} — Sensitivity at max corruption level (pooled across datasets × seeds)")
    print(f"{'='*70}")

    for model in icl_models:
        print(f"\n  {MODEL_DISPLAY.get(model, model)}:")
        print(f"  {'Context':<14}", end="")
        for c in corrupt_types:
            print(f"  {c[:8]:>8}", end="")
        print()
        print("  " + "-" * (14 + len(corrupt_types) * 10))

        for ctx in CTX_ORDER:
            vals = {}
            for corr in corrupt_types:
                sub = [r["corruption_delta"] for r in records
                       if r["icl_model"] == model
                       and r["context_type"] == ctx
                       and r["corruption_type"] == corr
                       and r["corruption_level"] == max(
                           CORRUPTION_SCHEDULE[corr])]
                # For class_balance, max raw level is 0.02 → stored as 0.98
                if corr == "class_balance":
                    sub = [r["corruption_delta"] for r in records
                           if r["icl_model"] == model
                           and r["context_type"] == ctx
                           and r["corruption_type"] == corr]
                    if sub:
                        sub = [r["corruption_delta"] for r in records
                               if r["icl_model"] == model
                               and r["context_type"] == ctx
                               and r["corruption_type"] == corr
                               and abs(r["corruption_level"] - 0.98) < 0.05]
                if sub:
                    vals[corr] = float(np.nanmean(sub))
                else:
                    vals[corr] = float("nan")

            if not any(np.isfinite(v) for v in vals.values()):
                continue
            print(f"  {ctx:<14}", end="")
            for c in corrupt_types:
                v = vals.get(c, float("nan"))
                if np.isfinite(v):
                    print(f"  {v:>8.3f}", end="")
                else:
                    print(f"  {'N/A':>8}", end="")
            print()

        # Sensitivity ratio for real context
        real_c1c2 = []
        real_c3 = []
        for r in records:
            if r["icl_model"] != model or r["context_type"] != "real":
                continue
            if r["corruption_type"] in ("marginal", "feature_correlation"):
                if r["corruption_level"] >= 0.9:  # near-max
                    real_c1c2.append(r["corruption_delta"])
            elif r["corruption_type"] == "label_conditional":
                if r["corruption_level"] >= 0.45:  # near-max
                    real_c3.append(r["corruption_delta"])
        if real_c1c2 and real_c3:
            ratio = np.nanmean(real_c1c2) / max(np.nanmean(real_c3), 1e-6)
            print(f"\n  Real context: C1+C2/C3 sensitivity ratio = {ratio:.1f}×")


# ── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EXP-C : Controlled Corruption Decomposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets", nargs="*", default=ALL_DATASETS)
    p.add_argument("--context_generators", nargs="*", default=ALL_CONTEXT_GENERATORS)
    p.add_argument("--icl_models", nargs="*", default=ICL_MODELS)
    p.add_argument("--seeds", type=int, nargs="*", default=ALL_SEEDS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    p.add_argument("--smoke_test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        log.info(" SMOKE TEST MODE")
        if args.datasets == ALL_DATASETS:
            args.datasets = ["breast_cancer"]
        if args.context_generators == ALL_CONTEXT_GENERATORS:
            args.context_generators = ["marginal"]
        if args.seeds == ALL_SEEDS:
            args.seeds = [0]

    config = {
        "exp_id":              EXP_ID,
        "datasets":            args.datasets,
        "context_generators":  args.context_generators,
        "icl_models":          args.icl_models,
        "seeds":               args.seeds,
        "device":              args.device,
        "smoke_test":          args.smoke_test,
        "max_ctx":             MAX_CTX,
        "corruption_schedule": {k: v for k, v in CORRUPTION_SCHEDULE.items()},
        "script":              str(Path(__file__).resolve()),
        "git_commit":          get_git_commit(),
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "python":              sys.version,
        "hostname":            platform.node(),
        "syn_store":           str(SyntheticDataStore().base_dir),
        # Key definitions
        "metric_definitions": {
            "roc_auc":          "Raw AUC after applying corruption to context",
            "corruption_delta": "AUC(uncorrupted_same_ctx) - AUC(corrupted) = AUC drop",
            "gap_from_real":    "AUC(real_uncorrupted) - AUC(current) = total gap from ideal",
        },
        "sample_idx_note": (
            "Each seed maps to a different synthetic sample: sample_idx = seed % 5. "
            "This provides proper variability across seeds, unlike  which used sample 0 only."
        ),
    }

    log.info("=" * 70)
    log.info(f"{EXP_ID}: Controlled Corruption Decomposition")
    log.info(f"  Datasets           : {args.datasets}")
    log.info(f"  Context generators : {args.context_generators}")
    log.info(f"  ICL models         : {args.icl_models}")
    log.info(f"  Seeds              : {args.seeds}")
    log.info(f"  MAX_CTX            : {MAX_CTX}")
    log.info(f"  Device             : {args.device}")
    log.info(f"  Git commit         : {config['git_commit']}")
    log.info("=" * 70)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_BASE / "exp_corruption_v3" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    records = run_experiment(
        datasets=args.datasets,
        context_generators=args.context_generators,
        icl_models=args.icl_models,
        seeds=args.seeds,
        device=args.device,
        results_dir=out_dir,
    )

    # Save
    with open(out_dir / "results.json", "w") as f:
        json.dump(records, f, indent=2, default=_json_default)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=_json_default)

    # Remove partial
    partial = out_dir / "results_partial.json"
    if partial.exists():
        partial.unlink()

    log.info(f"\nResults → {out_dir}  ({len(records)} records)")

    # Print summary
    print_summary(records)

    # Plots
    if not args.no_plots and len(records) > 10:
        generate_plots(records, out_dir)

    log.info(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
