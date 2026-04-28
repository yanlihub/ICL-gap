"""
Script: run_pda.py
"""
import argparse
import json
import logging
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synthetic_context.data.loader import DatasetLoader, get_task_type
from synthetic_context.data.syn_store import SyntheticDataStore
from synthetic_context.utils.seeds import set_all_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_BASE = Path("./results")
DATA_DIR     = Path("./data")

# Final 10-dataset benchmark (fixed 2026-03-13)
ALL_DATASETS = [
    "adult", "breast_cancer", "california_housing", "credit-g",
    "default", "diabetes", "kin8nm", "magic", "news", "shoppers",
]
ALL_GENERATORS = [
    "random", "marginal", "gmm", "smote",
    "ctgan", "tvae", "tabsyn", "tabpfngen",
]
ALL_SEEDS = [0, 1, 2, 3, 4]

# Regression PDA: quantile levels (49 values, finer approximation)
# Same for both TabPFN and TabICL → fair comparison
REG_ALPHAS = np.array(
    [0.01, 0.03, 0.05, 0.07, 0.09,
     0.11, 0.13, 0.15, 0.17, 0.19,
     0.21, 0.23, 0.25, 0.27, 0.29,
     0.31, 0.33, 0.35, 0.37, 0.39,
     0.41, 0.43, 0.45, 0.47, 0.49,
     0.51, 0.53, 0.55, 0.57, 0.59,
     0.61, 0.63, 0.65, 0.67, 0.69,
     0.71, 0.73, 0.75, 0.77, 0.79,
     0.81, 0.83, 0.85, 0.87, 0.89,
     0.91, 0.93, 0.95, 0.97, 0.99],
    dtype=np.float64,
)  # 50 quantile levels → 51 bins on common grid

N_PMF_BINS = 100   # bins for the common uniform grid (regression PMF)
TABPFN_MAX_CTX = 10_000

ICL_MODELS = ["tabpfn", "tabicl"]


# ── PDA computation ────────────────────────────────────────────────────────────

def js_divergence_row(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence between two probability vectors (log2 base → [0, 1])."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))


def compute_pda(proba_real: np.ndarray, proba_syn: np.ndarray) -> dict:
    """Compute PDA and JS divergence statistics (works for clf and reg).

    Parameters
    ----------
    proba_real : (n_test, K)  probability vectors with real context
    proba_syn  : (n_test, K)  probability vectors with syn context

    Returns
    -------
    dict with mean_js, max_js, std_js, frac_high_js, pda
    """
    n = len(proba_real)
    js_vals = np.array([js_divergence_row(proba_real[i], proba_syn[i]) for i in range(n)])
    return {
        "mean_js":      float(np.mean(js_vals)),
        "max_js":       float(np.max(js_vals)),
        "std_js":       float(np.std(js_vals)),
        "frac_high_js": float(np.mean(js_vals > 0.1)),
        "pda":          float(1.0 - np.mean(js_vals)),
    }


# ── Quantile → PMF conversion (regression) ────────────────────────────────────

def make_bin_edges(y_test: np.ndarray, n_bins: int = N_PMF_BINS) -> np.ndarray:
    """Create common uniform bin edges for a regression target.

    Range: [mean − 4*std, mean + 4*std] to capture tails,
    padded slightly beyond the observed range.
    """
    mu, sigma = float(y_test.mean()), float(y_test.std())
    sigma = max(sigma, 1e-6)
    lo = min(float(y_test.min()) - sigma, mu - 4 * sigma)
    hi = max(float(y_test.max()) + sigma, mu + 4 * sigma)
    return np.linspace(lo, hi, n_bins + 1)


def quantiles_to_pmf_matrix(
    quantile_matrix: np.ndarray,
    alphas: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Convert quantile predictions to a PMF matrix on a common uniform grid.

    For each test point, the quantile values define a piecewise-linear CDF.
    We evaluate this CDF at the bin edges to get probability masses.

    Parameters
    ----------
    quantile_matrix : (n_test, n_alphas)  quantile predictions
    alphas          : (n_alphas,)          probability levels
    bin_edges       : (n_bins+1,)          uniform grid edges

    Returns
    -------
    pmf_matrix : (n_test, n_bins)  probability mass functions summing to 1
    """
    n_test = quantile_matrix.shape[0]
    n_bins = len(bin_edges) - 1
    pmf_matrix = np.zeros((n_test, n_bins), dtype=np.float64)

    for i in range(n_test):
        q_vals = quantile_matrix[i]  # (n_alphas,)
        # Sort (should already be sorted, but ensure)
        sort_idx = np.argsort(q_vals)
        q_sorted = q_vals[sort_idx]
        a_sorted = alphas[sort_idx]

        # Piecewise-linear CDF at bin edges
        cdf_at_edges = np.interp(bin_edges, q_sorted, a_sorted,
                                 left=0.0, right=1.0)
        pmf = np.diff(cdf_at_edges)          # probability mass per bin
        pmf = np.clip(pmf, 1e-10, None)      # no zero-probability bins
        pmf /= pmf.sum()                      # renormalize to sum to 1
        pmf_matrix[i] = pmf

    return pmf_matrix


# ── TFM evaluation wrappers ────────────────────────────────────────────────────

def _clip_context(X: np.ndarray, y: np.ndarray, seed: int = 42):
    if len(X) <= TABPFN_MAX_CTX:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), TABPFN_MAX_CTX, replace=False)
    return X[idx], y[idx]


def _predict_chunked(reg, X: np.ndarray, chunk_size: int = 500, **kwargs) -> np.ndarray:
    """Call reg.predict in chunks to avoid GPU OOM on large test sets.

    TabPFN loads context + test batch jointly into GPU memory; for large test
    sets (>4k rows) this exceeds LUMI GPU memory.  Chunking keeps each call
    manageable while preserving the fitted context.

    TabPFN quantile predict returns a Python list of n_quantiles 1-D arrays
    (one array per quantile level, each of shape (n_samples,)).  We convert
    each chunk to (n_quantiles, chunk_size), concatenate along axis=1, then
    transpose to (n_samples, n_quantiles) for use by quantiles_to_pmf_matrix.
    Regular predict returns a 1-D ndarray (n_samples,) and is concatenated
    along axis=0 as usual.
    """
    parts = []
    for start in range(0, len(X), chunk_size):
        out = reg.predict(X[start: start + chunk_size], **kwargs)
        # Normalise: list of arrays → 2-D ndarray (n_quantiles, chunk_size)
        if isinstance(out, list):
            out = np.array(out)   # (n_quantiles, chunk_size)
        parts.append(out)

    if parts[0].ndim == 1:
        # mean / median / mode: (n_samples,) per chunk
        return np.concatenate(parts, axis=0)
    else:
        # quantiles: (n_quantiles, chunk_size) per chunk
        # → concat axis=1 → (n_quantiles, n_test) → .T → (n_test, n_quantiles)
        return np.concatenate(parts, axis=1).T


def eval_tabpfn_clf(X_ctx, y_ctx, X_test, y_test, seed, device
                    ) -> tuple[float, np.ndarray | None]:
    """TabPFN classification: returns (roc_auc, proba (n_test, n_classes))."""
    from sklearn.metrics import roc_auc_score
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return float("nan"), None

    X_ctx, y_ctx = _clip_context(X_ctx, y_ctx, seed)
    for dev in ([device] if device != "cuda" else ["cuda", "cpu"]):
        try:
            clf = TabPFNClassifier(device=dev, n_estimators=32,
                                   random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_ctx, y_ctx)
                proba = clf.predict_proba(X_test)
            classes = clf.classes_
            pos_idx = list(classes).index(1) if 1 in classes else 1
            auc = float(roc_auc_score(y_test, proba[:, pos_idx]))
            return auc, proba
        except Exception as e:
            if dev == "cuda" and any(k in str(e) for k in ("CUDA", "HIP", "GPU")):
                log.warning(f"  GPU failed, retry CPU: {e}")
                continue
            log.warning(f"  TabPFN-clf failed ({dev}): {e}")
            return float("nan"), None
    return float("nan"), None


def eval_tabpfn_reg(X_ctx, y_ctx, X_test, y_test, seed, device, bin_edges
                    ) -> tuple[float, float, np.ndarray | None]:
    """TabPFN regression: returns (r2, rmse, pmf_matrix (n_test, n_bins))."""
    from sklearn.metrics import r2_score, root_mean_squared_error
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        return float("nan"), float("nan"), None

    X_ctx, y_ctx = _clip_context(X_ctx, y_ctx, seed)
    for dev in ([device] if device != "cuda" else ["cuda", "cpu"]):
        try:
            reg = TabPFNRegressor(device=dev, n_estimators=32,
                                  random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg.fit(X_ctx, y_ctx)
                y_pred = _predict_chunked(reg, X_test)
                # Get quantiles for PDA — chunk to avoid GPU OOM on large test sets
                q_matrix = _predict_chunked(
                    reg, X_test,
                    output_type="quantiles",
                    quantiles=REG_ALPHAS.tolist(),
                )
                # q_matrix: (n_test, n_alphas)

            r2   = float(r2_score(y_test, y_pred))
            rmse = float(root_mean_squared_error(y_test, y_pred))
            pmf  = quantiles_to_pmf_matrix(q_matrix, REG_ALPHAS, bin_edges)
            return r2, rmse, pmf
        except Exception as e:
            if dev == "cuda" and any(k in str(e) for k in ("CUDA", "HIP", "GPU")):
                log.warning(f"  GPU failed, retry CPU: {e}")
                continue
            log.warning(f"  TabPFN-reg failed ({dev}): {e}")
            return float("nan"), float("nan"), None
    return float("nan"), float("nan"), None


def eval_tabicl_clf(X_ctx, y_ctx, X_test, y_test, seed, device
                    ) -> tuple[float, np.ndarray | None]:
    """TabICL classification: returns (roc_auc, proba (n_test, n_classes))."""
    from sklearn.metrics import roc_auc_score
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        return float("nan"), None

    X_ctx, y_ctx = _clip_context(X_ctx, y_ctx, seed)
    for dev in ([device] if device != "cuda" else ["cuda", "cpu"]):
        try:
            clf = TabICLClassifier(device=dev, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_ctx, y_ctx)
                proba = clf.predict_proba(X_test)
            classes = clf.classes_
            pos_idx = list(classes).index(1) if 1 in classes else 1
            auc = float(roc_auc_score(y_test, proba[:, pos_idx]))
            return auc, proba
        except Exception as e:
            if dev == "cuda" and any(k in str(e) for k in ("CUDA", "HIP", "GPU")):
                log.warning(f"  GPU failed, retry CPU: {e}")
                continue
            log.warning(f"  TabICL-clf failed ({dev}): {e}")
            return float("nan"), None
    return float("nan"), None


def eval_tabicl_reg(X_ctx, y_ctx, X_test, y_test, seed, device, bin_edges
                    ) -> tuple[float, float, np.ndarray | None]:
    """TabICL regression (quantile-based PDA): returns (r2, rmse, pmf_matrix)."""
    from sklearn.metrics import r2_score, root_mean_squared_error
    try:
        from tabicl import TabICLRegressor
    except ImportError:
        return float("nan"), float("nan"), None

    X_ctx, y_ctx = _clip_context(X_ctx, y_ctx, seed)
    for dev in ([device] if device != "cuda" else ["cuda", "cpu"]):
        try:
            reg = TabICLRegressor(device=dev, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reg.fit(X_ctx, y_ctx)
                y_pred = reg.predict(X_test)
                # Quantile predictions for PDA
                q_matrix = reg.predict(X_test, output_type="quantiles",
                                       alphas=REG_ALPHAS.tolist())
                # q_matrix: (n_test, n_alphas)

            r2   = float(r2_score(y_test, y_pred))
            rmse = float(root_mean_squared_error(y_test, y_pred))
            pmf  = quantiles_to_pmf_matrix(q_matrix, REG_ALPHAS, bin_edges)
            return r2, rmse, pmf
        except Exception as e:
            if dev == "cuda" and any(k in str(e) for k in ("CUDA", "HIP", "GPU")):
                log.warning(f"  GPU failed, retry CPU: {e}")
                continue
            log.warning(f"  TabICL-reg failed ({dev}): {e}")
            return float("nan"), float("nan"), None
    return float("nan"), float("nan"), None


# ── Main experiment loop ───────────────────────────────────────────────────────

def run_experiment(
    datasets: list[str],
    generators: list[str],
    seeds: list[int],
    icl_models: list[str],
    device: str,
    checkpoint_path: Optional[Path] = None,
) -> list[dict]:
    loader = DatasetLoader(cache_dir=DATA_DIR)
    store  = SyntheticDataStore()
    results: list[dict] = []

    for ds_name in datasets:
        log.info(f"\n{'='*65}\nDataset: {ds_name}")
        try:
            X_train, X_test, y_train, y_test = loader.load(ds_name)
        except Exception as e:
            log.error(f"  Load failed: {e}")
            continue

        task = get_task_type(ds_name)
        log.info(f"  task={task}  n_train={len(y_train)}  n_test={len(y_test)}")

        # Common bin edges for regression PDA (computed once per dataset)
        bin_edges = make_bin_edges(y_test) if task == "regression" else None

        # ── Real context baselines (per icl_model × seed) ─────────────────
        # score_real[model][seed] = (primary_score, pmf_or_proba)
        score_real: dict[str, dict[int, tuple]] = {m: {} for m in icl_models}

        for model in icl_models:
            log.info(f"  [REAL] model={model}")
            for seed in seeds:
                set_all_seeds(seed)
                if task == "regression":
                    if model == "tabpfn":
                        r2, rmse, pmf = eval_tabpfn_reg(
                            X_train, y_train, X_test, y_test, seed, device, bin_edges)
                        score_real[model][seed] = (r2, rmse, pmf)
                        log.info(f"    seed={seed}  R²={r2:.4f}  RMSE={rmse:.4f}")
                    else:  # tabicl
                        r2, rmse, pmf = eval_tabicl_reg(
                            X_train, y_train, X_test, y_test, seed, device, bin_edges)
                        score_real[model][seed] = (r2, rmse, pmf)
                        log.info(f"    seed={seed}  R²={r2:.4f}  RMSE={rmse:.4f}")
                else:  # classification
                    if model == "tabpfn":
                        auc, proba = eval_tabpfn_clf(
                            X_train, y_train, X_test, y_test, seed, device)
                        score_real[model][seed] = (auc, proba)
                        log.info(f"    seed={seed}  AUC={auc:.4f}")
                    else:
                        auc, proba = eval_tabicl_clf(
                            X_train, y_train, X_test, y_test, seed, device)
                        score_real[model][seed] = (auc, proba)
                        log.info(f"    seed={seed}  AUC={auc:.4f}")

        # ── Synthetic contexts (from SyntheticDataStore) ───────────────────
        for gen_name in generators:
            log.info(f"\n  [GEN] {gen_name}")

            if not store.all_exist(gen_name, ds_name, len(seeds)):
                log.warning(f"  Cache incomplete for {gen_name}×{ds_name} — skip")
                continue

            for seed in seeds:
                set_all_seeds(seed)

                try:
                    X_syn, y_syn = store.load(gen_name, ds_name, seed)
                except Exception as e:
                    log.warning(f"    Load syn failed {gen_name} seed={seed}: {e}")
                    continue

                for model in icl_models:
                    rec_base = {
                        "exp_id":    "EXP-PDA-",
                        "dataset":   ds_name,
                        "generator": gen_name,
                        "icl_model": model,
                        "seed":      seed,
                        "task_type": task,
                    }

                    if task == "regression":
                        # Real baseline for this (model, seed)
                        real_tuple = score_real[model].get(seed, (float("nan"), float("nan"), None))
                        r2_real, rmse_real, pmf_real = real_tuple

                        t0 = time.perf_counter()
                        if model == "tabpfn":
                            r2_syn, rmse_syn, pmf_syn = eval_tabpfn_reg(
                                X_syn, y_syn, X_test, y_test, seed, device, bin_edges)
                        else:
                            r2_syn, rmse_syn, pmf_syn = eval_tabicl_reg(
                                X_syn, y_syn, X_test, y_test, seed, device, bin_edges)
                        wall_s = time.perf_counter() - t0

                        icl_gap = (float(r2_real - r2_syn)
                                   if np.isfinite(r2_real) and np.isfinite(r2_syn)
                                   else float("nan"))
                        rmse_ratio = (float(rmse_syn / rmse_real)
                                      if np.isfinite(rmse_real) and rmse_real > 0
                                         and np.isfinite(rmse_syn)
                                      else float("nan"))

                        if pmf_real is not None and pmf_syn is not None:
                            pda_m = compute_pda(pmf_real, pmf_syn)
                        else:
                            pda_m = {"mean_js": float("nan"), "max_js": float("nan"),
                                     "std_js": float("nan"), "frac_high_js": float("nan"),
                                     "pda": float("nan")}

                        log.info(
                            f"    {gen_name} {model} seed={seed}  "
                            f"R²_syn={r2_syn:.4f}  gap={icl_gap:+.4f}  "
                            f"RMSE_ratio={rmse_ratio:.3f}  "
                            f"mean_JS={pda_m['mean_js']:.4f}  "
                            f"max_JS={pda_m['max_js']:.4f}"
                        )
                        results.append({
                            **rec_base,
                            "r2_real":    r2_real,
                            "r2_syn":     r2_syn,
                            "rmse_real":  rmse_real,
                            "rmse_syn":   rmse_syn,
                            "rmse_ratio": rmse_ratio,
                            "icl_gap":    icl_gap,
                            "wall_time_s": round(wall_s, 2),
                            **pda_m,
                        })
                        if checkpoint_path is not None:
                            _checkpoint(results, checkpoint_path)

                    else:  # classification
                        real_tuple = score_real[model].get(seed, (float("nan"), None))
                        auc_real, proba_real = real_tuple

                        t0 = time.perf_counter()
                        if model == "tabpfn":
                            auc_syn, proba_syn = eval_tabpfn_clf(
                                X_syn, y_syn, X_test, y_test, seed, device)
                        else:
                            auc_syn, proba_syn = eval_tabicl_clf(
                                X_syn, y_syn, X_test, y_test, seed, device)
                        wall_s = time.perf_counter() - t0

                        icl_gap = (float(auc_real - auc_syn)
                                   if np.isfinite(auc_real) and np.isfinite(auc_syn)
                                   else float("nan"))

                        if proba_real is not None and proba_syn is not None:
                            pda_m = compute_pda(proba_real, proba_syn)
                        else:
                            pda_m = {"mean_js": float("nan"), "max_js": float("nan"),
                                     "std_js": float("nan"), "frac_high_js": float("nan"),
                                     "pda": float("nan")}

                        log.info(
                            f"    {gen_name} {model} seed={seed}  "
                            f"AUC_syn={auc_syn:.4f}  gap={icl_gap:+.4f}  "
                            f"mean_JS={pda_m['mean_js']:.4f}  "
                            f"max_JS={pda_m['max_js']:.4f}"
                        )
                        results.append({
                            **rec_base,
                            "roc_auc_real": auc_real,
                            "roc_auc_syn":  auc_syn,
                            "icl_gap":      icl_gap,
                            "wall_time_s":  round(wall_s, 2),
                            **pda_m,
                        })
                        if checkpoint_path is not None:
                            _checkpoint(results, checkpoint_path)

    return results


def _checkpoint(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",   nargs="*", default=ALL_DATASETS)
    parser.add_argument("--generators", nargs="*", default=ALL_GENERATORS)
    parser.add_argument("--seeds",      nargs="*", type=int, default=ALL_SEEDS)
    parser.add_argument("--icl_models", nargs="*", default=ICL_MODELS)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--out_dir",    default=str(RESULTS_BASE / "exp_pda_v3"))
    args = parser.parse_args()

    log.info("EXP-PDA- (Canonical 10 Datasets x 8 Generators x 5 Seeds)")
    log.info(f"  Datasets   : {args.datasets}")
    log.info(f"  Generators : {args.generators}")
    log.info(f"  ICL models : {args.icl_models}")
    log.info(f"  Seeds      : {args.seeds}")
    log.info(f"  Device     : {args.device}")
    log.info(f"  Regression PDA: quantile-based ({len(REG_ALPHAS)} alphas → {N_PMF_BINS} bins)")
    log.info(f"  Both TabPFN and TabICL use same ALPHAS for fair comparison")

    partial_path = Path(args.out_dir) / "results_partial.json"

    results = run_experiment(
        datasets   = args.datasets,
        generators = args.generators,
        seeds      = args.seeds,
        icl_models = args.icl_models,
        device     = args.device,
        checkpoint_path = partial_path,
    )

    # Save results
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nSaved {len(results)} records → {results_path}")

    if partial_path.exists():
        partial_path.unlink()

    # Quick summary
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        g = r["generator"]
        gap = r.get("icl_gap")
        mjs = r.get("mean_js")
        xjs = r.get("max_js")
        if gap is not None and np.isfinite(gap):   agg[g].append(("gap", gap))
        if mjs is not None and np.isfinite(mjs):   agg[g].append(("mjs", mjs))
        if xjs is not None and np.isfinite(xjs):   agg[g].append(("xjs", xjs))

    print(f"\n{'Generator':12s}  {'mean gap':>10s}  {'mean JS':>10s}  {'max JS':>10s}  n")
    print("-" * 55)
    for gen in ALL_GENERATORS:
        gaps = [v for k, v in agg[gen] if k == "gap"]
        mjss = [v for k, v in agg[gen] if k == "mjs"]
        xjss = [v for k, v in agg[gen] if k == "xjs"]
        if gaps:
            print(f"{gen:12s}  {np.mean(gaps):>10.4f}  "
                  f"{np.mean(mjss) if mjss else float('nan'):>10.4f}  "
                  f"{np.mean(xjss) if xjss else float('nan'):>10.4f}  {len(gaps)}")


if __name__ == "__main__":
    main()
