"""
Script: run_privacy.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synthetic_context.data.loader import DatasetLoader, get_task_type
from synthetic_context.generators import get_generator
from synthetic_context.utils.metrics import compute_all_fidelity_metrics, FIDELITY_METRIC_KEYS
from synthetic_context.utils.seeds import set_all_seeds

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_BASE = Path("./results")
DATA_DIR     = Path("./data")
SYN_CACHE_DIR = DATA_DIR / "syn_cache" / "exp_privacy_v3"
REAL_CACHE_DIR = RESULTS_BASE / "exp_privacy_v3" / "real_baselines"

DEFAULT_DATASETS = ["adult", "breast_cancer", "credit-g", "default", "diabetes", "magic", "shoppers"]
DEFAULT_SEEDS    = [0, 1, 2, 3, 4]
EPSILON_VALUES   = [0.1, 0.5, 1.0, 5.0, 10.0]
DP_GENERATORS    = ["dpgan", "pategan"]   # equal status
REF_GENERATORS   = ["ctgan"]             # non-private reference (eps=inf)
TABPFN_MAX_CTX   = 10_000

# All 15 metric keys (fidelity + ICL)
ALL_ICL_KEYS = ["icl_gap", "icl_gap_tabicl", "mean_js", "max_js"]
ALL_METRIC_KEYS = ALL_ICL_KEYS + FIDELITY_METRIC_KEYS  # 4 + 11 = 15


def _epsilon_label(epsilon: float) -> str:
    return f"e{epsilon}" if math.isfinite(epsilon) else "inf"


class PrivacySyntheticCache:
    """Persistent cache for EXP-PRIVACY synthetic samples."""

    def __init__(self, base_dir: Path | str = SYN_CACHE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _condition_dir(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
    ) -> Path:
        eps_label = _epsilon_label(epsilon)
        stem = (
            f"{dataset}__{eps_label}"
            f"__fitseed{fit_seed}"
            f"__n{n_train}"
            f"__iter{n_iter}"
            f"__bs{batch_size}"
        )
        return self.base_dir / generator / stem

    def sample_path(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
        sample_idx: int,
    ) -> Path:
        return self._condition_dir(
            dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size
        ) / f"sample{sample_idx}.npz"

    def meta_path(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
    ) -> Path:
        return self._condition_dir(
            dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size
        ) / "meta.json"

    def exists(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
        sample_idx: int,
    ) -> bool:
        return self.sample_path(
            dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size, sample_idx
        ).exists()

    def all_exist(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
        n_samples: int,
    ) -> bool:
        return all(
            self.exists(
                dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size, sample_idx
            )
            for sample_idx in range(n_samples)
        )

    def save(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
        sample_idx: int,
        X_syn: np.ndarray,
        y_syn: np.ndarray,
        fit_time_s: float | None = None,
        n_samples: int | None = None,
    ) -> None:
        sample_path = self.sample_path(
            dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size, sample_idx
        )
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_path, X_syn=X_syn.astype(np.float32), y_syn=y_syn)

        meta_path = self.meta_path(
            dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size
        )
        meta = {
            "dataset": dataset,
            "generator": generator,
            "epsilon": epsilon,
            "epsilon_label": _epsilon_label(epsilon),
            "fit_seed": fit_seed,
            "n_train": n_train,
            "n_iter": n_iter,
            "batch_size": batch_size,
        }
        if fit_time_s is not None:
            meta["fit_time_s"] = round(fit_time_s, 2)
        if n_samples is not None:
            meta["n_samples"] = n_samples
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        log.info(f"[SynCache] saved {sample_path}")

    def load(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
        sample_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_path = self.sample_path(
            dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size, sample_idx
        )
        data = np.load(sample_path, allow_pickle=True)
        return data["X_syn"], data["y_syn"]

    def load_meta(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
    ) -> dict | None:
        meta_path = self.meta_path(
            dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size
        )
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return json.load(f)

    def eval_path(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
        sample_idx: int,
        eval_seed: int,
        device: str,
    ) -> Path:
        safe_device = device.replace("/", "_").replace(":", "_")
        return self._condition_dir(
            dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size
        ) / f"eval_sample{sample_idx}_seed{eval_seed}_{safe_device}.json"

    def eval_exists(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
        sample_idx: int,
        eval_seed: int,
        device: str,
    ) -> bool:
        return self.eval_path(
            dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size,
            sample_idx, eval_seed, device,
        ).exists()

    def save_eval(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
        sample_idx: int,
        eval_seed: int,
        device: str,
        record: dict,
    ) -> None:
        eval_path = self.eval_path(
            dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size,
            sample_idx, eval_seed, device,
        )
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w") as f:
            json.dump(record, f, indent=2, default=str)

    def load_eval(
        self,
        dataset: str,
        generator: str,
        epsilon: float,
        fit_seed: int,
        n_train: int,
        n_iter: int,
        batch_size: int,
        sample_idx: int,
        eval_seed: int,
        device: str,
    ) -> dict:
        eval_path = self.eval_path(
            dataset, generator, epsilon, fit_seed, n_train, n_iter, batch_size,
            sample_idx, eval_seed, device,
        )
        with open(eval_path) as f:
            return json.load(f)


class RealBaselineCache:
    """Persistent cache for real-data ICL baselines."""

    def __init__(self, base_dir: Path | str = REAL_CACHE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path(self, dataset: str, seed: int, device: str) -> Path:
        safe_device = device.replace("/", "_").replace(":", "_")
        return self.base_dir / dataset / f"seed{seed}_{safe_device}.npz"

    def exists(self, dataset: str, seed: int, device: str) -> bool:
        return self.path(dataset, seed, device).exists()

    def save(
        self,
        dataset: str,
        seed: int,
        device: str,
        auc_tabpfn: float,
        proba_tabpfn: np.ndarray | None,
        auc_tabicl: float,
    ) -> None:
        cache_path = self.path(dataset, seed, device)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            auc_tabpfn=np.array([auc_tabpfn], dtype=float),
            auc_tabicl=np.array([auc_tabicl], dtype=float),
            has_proba=np.array([proba_tabpfn is not None], dtype=bool),
            proba_tabpfn=(
                proba_tabpfn
                if proba_tabpfn is not None
                else np.empty((0, 0), dtype=np.float32)
            ),
        )

    def load(
        self,
        dataset: str,
        seed: int,
        device: str,
    ) -> tuple[float, np.ndarray | None, float]:
        data = np.load(self.path(dataset, seed, device), allow_pickle=True)
        auc_tabpfn = float(data["auc_tabpfn"][0])
        auc_tabicl = float(data["auc_tabicl"][0])
        has_proba = bool(data["has_proba"][0])
        proba_tabpfn = data["proba_tabpfn"] if has_proba else None
        return auc_tabpfn, proba_tabpfn, auc_tabicl


# ── PDA helpers ────────────────────────────────────────────────────────────────

def js_divergence_row(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """JS divergence (log2) between two probability vectors. Range [0, 1]."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m))))


def compute_pda_metrics(
    proba_real: np.ndarray,
    proba_syn: np.ndarray,
) -> dict[str, float]:
    """Compute ICL-PDA metrics from two (n_test, n_classes) probability matrices."""
    js_vals = np.array([
        js_divergence_row(proba_real[i], proba_syn[i])
        for i in range(len(proba_real))
    ])
    return {
        "pda":     float(1.0 - np.mean(js_vals)),
        "mean_js": float(np.mean(js_vals)),
        "std_js":  float(np.std(js_vals)),
        "max_js":  float(np.max(js_vals)),
    }


def _nan_pda() -> dict[str, float]:
    return {"pda": float("nan"), "mean_js": float("nan"),
            "std_js": float("nan"), "max_js": float("nan")}


# ── ICL wrappers ───────────────────────────────────────────────────────────────

def _clip_ctx(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Subsample context to TABPFN_MAX_CTX if needed."""
    if len(X) <= TABPFN_MAX_CTX:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), TABPFN_MAX_CTX, replace=False)
    return X[idx], y[idx]


def run_tabpfn(
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    device: str,
) -> tuple[float, np.ndarray | None]:
    """Fit TabPFN, return (roc_auc, proba_matrix). Returns (nan, None) on failure."""
    from sklearn.metrics import roc_auc_score
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        log.error("tabpfn not installed")
        return float("nan"), None

    X_ctx, y_ctx = _clip_ctx(X_ctx, y_ctx, seed)

    for dev in ([device] if device != "cuda" else ["cuda", "cpu"]):
        try:
            clf = TabPFNClassifier(device=dev, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_ctx, y_ctx)
                proba = clf.predict_proba(X_test)
            pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
            auc = float(roc_auc_score(y_test, proba[:, pos_idx]))
            return auc, proba
        except Exception as e:
            if dev == "cuda" and any(k in str(e) for k in ("CUDA", "HIP", "GPU", "cuda", "hip")):
                log.warning(f"  TabPFN GPU failed, retrying CPU: {e}")
                continue
            log.warning(f"  TabPFN failed ({dev}): {e}")
            return float("nan"), None
    return float("nan"), None


def run_tabicl(
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    device: str,
) -> float:
    """Fit TabICL, return roc_auc. Returns nan on failure or if not installed."""
    from sklearn.metrics import roc_auc_score
    try:
        from tabicl import TabICLClassifier
    except ImportError:
        log.warning("tabicl not installed — icl_gap_tabicl will be NaN")
        return float("nan")

    for dev in ([device] if device != "cuda" else ["cuda", "cpu"]):
        try:
            clf = TabICLClassifier(device=dev, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_ctx, y_ctx)
                proba = clf.predict_proba(X_test)
            pos_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 1
            auc = float(roc_auc_score(y_test, proba[:, pos_idx]))
            return auc
        except Exception as e:
            if dev == "cuda" and any(k in str(e) for k in ("CUDA", "HIP", "GPU", "cuda", "hip")):
                log.warning(f"  TabICL GPU failed, retrying CPU: {e}")
                continue
            log.warning(f"  TabICL failed ({dev}): {e}")
            return float("nan")
    return float("nan")


# ── Main experiment loop ───────────────────────────────────────────────────────

def run_experiment(
    datasets: list[str],
    seeds: list[int],
    epsilon_values: list[float],
    device: str,
    n_iter: int,
    batch_size: int,
    fit_seed: int,
) -> list[dict]:
    loader = DatasetLoader(cache_dir=DATA_DIR)
    syn_cache = PrivacySyntheticCache()
    real_cache = RealBaselineCache()
    results: list[dict] = []

    for ds_name in datasets:
        log.info(f"\n{'='*65}\nDataset: {ds_name}")
        try:
            X_train, X_test, y_train, y_test = loader.load(ds_name)
        except Exception as e:
            log.error(f"  Load failed: {e}")
            continue

        task = get_task_type(ds_name)
        if task != "binary":
            log.warning(f"  Skipping {ds_name}: task={task} (not binary)")
            continue

        log.info(f"  n_train={len(y_train)}  n_test={len(y_test)}")

        # ── Real baseline (per seed) ──────────────────────────────────────────
        real_auc_tabpfn:  dict[int, float]             = {}
        real_proba_tabpfn: dict[int, np.ndarray | None] = {}
        real_auc_tabicl:  dict[int, float]             = {}

        for seed in seeds:
            if real_cache.exists(ds_name, seed, device):
                auc_pfn, proba_pfn, auc_icl = real_cache.load(ds_name, seed, device)
                log.info(
                    f"  [real cache] seed={seed}  "
                    f"TabPFN-AUC={auc_pfn:.4f}  TabICL-AUC={auc_icl:.4f}"
                )
            else:
                set_all_seeds(seed)
                auc_pfn, proba_pfn = run_tabpfn(X_train, y_train, X_test, y_test, seed, device)
                auc_icl = run_tabicl(X_train, y_train, X_test, y_test, seed, device)
                real_cache.save(ds_name, seed, device, auc_pfn, proba_pfn, auc_icl)
                log.info(
                    f"  [real] seed={seed}  TabPFN-AUC={auc_pfn:.4f}  TabICL-AUC={auc_icl:.4f}"
                )

            real_auc_tabpfn[seed]   = auc_pfn
            real_proba_tabpfn[seed] = proba_pfn
            real_auc_tabicl[seed]   = auc_icl

        # ── Synthetic conditions ──────────────────────────────────────────────
        conditions: list[tuple[str, float]] = []
        for gen_name in DP_GENERATORS:
            for eps in epsilon_values:
                conditions.append((gen_name, eps))
        for gen_name in REF_GENERATORS:
            conditions.append((gen_name, float("inf")))

        for (gen_name, epsilon) in conditions:
            eps_label    = _epsilon_label(epsilon)
            condition_id = f"{gen_name}_{eps_label}"
            log.info(f"\n  [GEN] {gen_name}  eps={epsilon}")
            fit_time = 0.0
            fit_performed = False
            n_samples = len(seeds)

            cache_ready = syn_cache.all_exist(
                ds_name, gen_name, epsilon, fit_seed, len(y_train), n_iter, batch_size, n_samples
            )

            if cache_ready:
                log.info(
                    f"  [CACHE] HIT  {condition_id}  fit_seed={fit_seed}  samples={n_samples}"
                )
                meta = syn_cache.load_meta(
                    ds_name, gen_name, epsilon, fit_seed, len(y_train), n_iter, batch_size
                )
                if meta is not None and "fit_time_s" in meta:
                    fit_time = float(meta["fit_time_s"])
            else:
                log.info(
                    f"  [CACHE] MISS {condition_id}  fit_seed={fit_seed}  "
                    f"→ fit once, generate {n_samples} samples"
                )
                set_all_seeds(fit_seed)
                t0 = time.perf_counter()
                gen_kwargs: dict = {
                    "seed": fit_seed,
                    "n_iter": n_iter,
                    "batch_size": batch_size,
                }
                if gen_name in DP_GENERATORS + REF_GENERATORS:
                    gen_kwargs["device"] = device
                if gen_name in ("dpgan", "pategan"):
                    gen_kwargs["epsilon"] = epsilon
                    gen_kwargs["delta"]   = 1e-5

                try:
                    gen = get_generator(gen_name, **gen_kwargs)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gen.fit(X_train, y_train)
                    fit_time = time.perf_counter() - t0
                    fit_performed = True

                    for sample_idx in range(n_samples):
                        X_syn, y_syn = gen.generate(len(X_train))
                        if not syn_cache.exists(
                            ds_name, gen_name, epsilon, fit_seed, len(y_train),
                            n_iter, batch_size, sample_idx,
                        ):
                            syn_cache.save(
                                ds_name, gen_name, epsilon, fit_seed, len(y_train),
                                n_iter, batch_size, sample_idx, X_syn, y_syn,
                                fit_time_s=fit_time, n_samples=n_samples,
                            )
                except Exception as e:
                    fit_time = time.perf_counter() - t0
                    log.warning(f"    {condition_id} fit-once failed: {e}")

            for sample_idx, seed in enumerate(seeds):
                if syn_cache.eval_exists(
                    ds_name, gen_name, epsilon, fit_seed, len(y_train),
                    n_iter, batch_size, sample_idx, seed, device,
                ):
                    cached_record = syn_cache.load_eval(
                        ds_name, gen_name, epsilon, fit_seed, len(y_train),
                        n_iter, batch_size, sample_idx, seed, device,
                    )
                    results.append(cached_record)
                    _checkpoint(results, RESULTS_BASE / "exp_privacy_v3" / "results_partial.json")
                    log.info(
                        f"    [eval cache] {condition_id} sample={sample_idx} eval_seed={seed}"
                    )
                    continue

                if not syn_cache.exists(
                    ds_name, gen_name, epsilon, fit_seed, len(y_train),
                    n_iter, batch_size, sample_idx,
                ):
                    log.warning(
                        f"    Missing cached sample for {condition_id} sample={sample_idx} "
                        f"eval_seed={seed}"
                    )
                    results.append(_nan_record(
                        ds_name, gen_name, epsilon, condition_id, seed,
                        sample_idx, fit_seed,
                        real_auc_tabpfn.get(seed, float('nan')),
                        real_auc_tabicl.get(seed, float('nan')),
                        fit_time,
                        fit_performed,
                    ))
                    _checkpoint(results, RESULTS_BASE / "exp_privacy_v3" / "results_partial.json")
                    continue

                X_syn, y_syn = syn_cache.load(
                    ds_name, gen_name, epsilon, fit_seed, len(y_train),
                    n_iter, batch_size, sample_idx,
                )
                set_all_seeds(seed)

                # ── TabPFN: AUC + proba for PDA ───────────────────────────────
                auc_syn_pfn, proba_syn_pfn = run_tabpfn(
                    X_syn, y_syn, X_test, y_test, seed, device
                )
                auc_real_pfn = real_auc_tabpfn.get(seed, float("nan"))
                icl_gap_pfn  = (
                    float(auc_real_pfn - auc_syn_pfn)
                    if math.isfinite(auc_real_pfn) and math.isfinite(auc_syn_pfn)
                    else float("nan")
                )

                # ── TabICL: AUC only ──────────────────────────────────────────
                auc_syn_icl  = run_tabicl(X_syn, y_syn, X_test, y_test, seed, device)
                auc_real_icl = real_auc_tabicl.get(seed, float("nan"))
                icl_gap_icl  = (
                    float(auc_real_icl - auc_syn_icl)
                    if math.isfinite(auc_real_icl) and math.isfinite(auc_syn_icl)
                    else float("nan")
                )

                # ── ICL-PDA ───────────────────────────────────────────────────
                proba_real_pfn = real_proba_tabpfn.get(seed)
                if proba_real_pfn is not None and proba_syn_pfn is not None:
                    pda_m = compute_pda_metrics(proba_real_pfn, proba_syn_pfn)
                else:
                    pda_m = _nan_pda()

                # ── Fidelity metrics (all 11) ─────────────────────────────────
                fidelity = compute_all_fidelity_metrics(
                    X_train, X_syn, y_syn, X_test, y_test, seed=seed
                )

                log.info(
                    f"    {condition_id} sample={sample_idx} eval_seed={seed}  "
                    f"gap_pfn={icl_gap_pfn:+.4f}  gap_icl={icl_gap_icl:+.4f}  "
                    f"PDA={pda_m['pda']:.4f}  max_JS={pda_m['max_js']:.4f}  "
                    f"KS={fidelity.get('fid_ks', float('nan')):.4f}  "
                    f"TSTR={fidelity.get('fid_tstr_auc', float('nan')):.4f}  "
                    f"NFN={fidelity.get('fid_nfn', float('nan')):.4f}"
                )

                rec = {
                    "exp_id":           "EXP-PRIVACY-",
                    "dataset":          ds_name,
                    "generator":        gen_name,
                    "epsilon":          epsilon,
                    "epsilon_label":    eps_label,
                    "condition":        condition_id,
                    "seed":             seed,
                    "sample_idx":       sample_idx,
                    "fit_seed":         fit_seed,
                    "fit_performed":    fit_performed,
                    # TabPFN
                    "roc_auc_real":     auc_real_pfn,
                    "roc_auc_syn":      auc_syn_pfn,
                    "icl_gap":          icl_gap_pfn,
                    # TabICL
                    "roc_auc_real_tabicl": auc_real_icl,
                    "roc_auc_syn_tabicl":  auc_syn_icl,
                    "icl_gap_tabicl":      icl_gap_icl,
                    # PDA
                    **pda_m,
                    # 11 fidelity metrics
                    **fidelity,
                    "fit_time_s": round(fit_time, 2),
                }
                syn_cache.save_eval(
                    ds_name, gen_name, epsilon, fit_seed, len(y_train),
                    n_iter, batch_size, sample_idx, seed, device, rec,
                )
                results.append(rec)
                _checkpoint(results, RESULTS_BASE / "exp_privacy_v3" / "results_partial.json")

    return results


def _nan_record(
    dataset: str, gen: str, epsilon: float, condition: str,
    seed: int, sample_idx: int, fit_seed: int,
    auc_real_pfn: float, auc_real_icl: float, fit_time: float, fit_performed: bool,
) -> dict:
    eps_label = _epsilon_label(epsilon)
    return {
        "exp_id": "EXP-PRIVACY-", "dataset": dataset,
        "generator": gen, "epsilon": epsilon, "epsilon_label": eps_label,
        "condition": condition, "seed": seed,
        "sample_idx": sample_idx, "fit_seed": fit_seed,
        "fit_performed": fit_performed,
        "roc_auc_real": auc_real_pfn, "roc_auc_syn": float("nan"),
        "icl_gap": float("nan"),
        "roc_auc_real_tabicl": auc_real_icl, "roc_auc_syn_tabicl": float("nan"),
        "icl_gap_tabicl": float("nan"),
        "fit_time_s": round(fit_time, 2),
        **_nan_pda(),
        **{k: float("nan") for k in FIDELITY_METRIC_KEYS},
    }


def _checkpoint(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


# ── Summary ────────────────────────────────────────────────────────────────────

# All 15 metrics with display name, key in record, and direction
METRIC_DISPLAY: list[tuple[str, str, bool]] = [
    # (display_name, record_key, higher_is_better)
    # -- ICL metrics --
    ("ICL-Gap(TabPFN)",  "icl_gap",              False),
    ("ICL-Gap(TabICL)",  "icl_gap_tabicl",        False),
    ("mean_JS",          "mean_js",               False),
    ("max_JS",           "max_js",                False),
    # -- Marginal fidelity --
    ("KS",               "fid_ks",                False),
    ("JSD",              "fid_jsd",               False),
    ("WD-1D",            "fid_wd_1d",             False),
    # -- Joint distribution --
    ("SWD",              "fid_wd_sliced",         False),
    ("MMD",              "fid_mmd",               False),
    # -- Correlation --
    ("NFN",              "fid_nfn",               False),
    # -- Utility (TSTR) --
    ("TSTR-XGB",         "fid_tstr_auc",          True),
    ("TSTR-MLP",         "fid_tstr_mlp",          True),
    ("TSTR-RF",          "fid_tstr_rf",           True),
    # -- Density/coverage --
    ("α-Precision",      "fid_alpha_precision",   True),
    ("β-Recall",         "fid_beta_recall",       True),
]


def _fmt_mean_std(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "nan"
    if not math.isfinite(std):
        return f"{mean:.4f}"
    return f"{mean:.4f}±{std:.4f}"


def _sorted_group_keys(grouped: dict[tuple, list[dict]]) -> list[tuple]:
    def sort_key(g: tuple) -> tuple:
        if len(g) == 2:
            gen, eps = g
            return (gen, eps if math.isfinite(eps) else 1e9)
        if len(g) == 3:
            ds, gen, eps = g
            return (ds, gen, eps if math.isfinite(eps) else 1e9)
        return g

    return sorted(grouped.keys(), key=sort_key)


def aggregate_results(
    results: list[dict],
    group_keys: list[str],
    expected_n: int,
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    metric_keys = [rec_key for _, rec_key, _ in METRIC_DISPLAY]

    for r in results:
        grouped[tuple(r[k] for k in group_keys)].append(r)

    rows: list[dict] = []
    for g in _sorted_group_keys(grouped):
        recs = grouped[g]
        row = {k: v for k, v in zip(group_keys, g)}
        row["n_records"] = len(recs)
        row["expected_n"] = expected_n
        row["complete"] = len(recs) == expected_n

        for rec_key in metric_keys:
            vals = [
                float(r.get(rec_key, float("nan")))
                for r in recs
                if math.isfinite(float(r.get(rec_key, float("nan"))))
            ]
            if vals:
                row[f"{rec_key}_mean"] = float(np.mean(vals))
                row[f"{rec_key}_std"] = float(np.std(vals))
            else:
                row[f"{rec_key}_mean"] = float("nan")
                row[f"{rec_key}_std"] = float("nan")
        rows.append(row)

    return rows


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_artifacts(
    results: list[dict],
    out_dir: Path,
    datasets: list[str],
    seeds: list[int],
) -> tuple[list[dict], list[dict]]:
    condition_rows = aggregate_results(
        results=results,
        group_keys=["generator", "epsilon"],
        expected_n=len(datasets) * len(seeds),
    )
    dataset_condition_rows = aggregate_results(
        results=results,
        group_keys=["dataset", "generator", "epsilon"],
        expected_n=len(seeds),
    )

    with open(out_dir / "summary_by_condition_mean_std.json", "w") as f:
        json.dump(condition_rows, f, indent=2, default=str)
    with open(out_dir / "summary_by_dataset_condition_mean_std.json", "w") as f:
        json.dump(dataset_condition_rows, f, indent=2, default=str)

    _write_summary_csv(out_dir / "summary_by_condition_mean_std.csv", condition_rows)
    _write_summary_csv(
        out_dir / "summary_by_dataset_condition_mean_std.csv",
        dataset_condition_rows,
    )

    return condition_rows, dataset_condition_rows


def print_summary(
    condition_rows: list[dict],
    dataset_condition_rows: list[dict],
) -> None:
    """Print per-(generator, epsilon) mean±std and completion counts."""

    print("\n" + "=" * 120)
    print("EXP-PRIVACY Summary — Completion by Dataset × Condition")
    print("=" * 120)
    hdr0 = (
        f"{'Dataset':<18} {'Generator':<10} {'eps':>6}  "
        f"{'n':>5}  {'done?':>6}  {'ICL-Gap(PFN)':>17}  {'TSTR-XGB':>17}"
    )
    print(hdr0)
    print("-" * 120)
    for row in dataset_condition_rows:
        eps = row["epsilon"]
        eps_str = f"{eps:.1f}" if math.isfinite(eps) else "inf"
        print(
            f"{row['dataset']:<18} {row['generator']:<10} {eps_str:>6}  "
            f"{row['n_records']:>2}/{row['expected_n']:<2}  "
            f"{str(row['complete']):>6}  "
            f"{_fmt_mean_std(row['icl_gap_mean'], row['icl_gap_std']):>17}  "
            f"{_fmt_mean_std(row['fid_tstr_auc_mean'], row['fid_tstr_auc_std']):>17}"
        )

    print("\n" + "=" * 120)
    print("EXP-PRIVACY Summary — ICL Metrics (mean±std across datasets × samples)")
    print("=" * 120)
    hdr1 = (
        f"{'Generator':<10} {'eps':>6}  {'n':>5}  {'done?':>6}  "
        f"{'ICL-Gap(PFN)':>17}  {'ICL-Gap(ICL)':>17}  {'mean_JS':>17}  {'max_JS':>17}"
    )
    print(hdr1)
    print("-" * 120)
    for row in condition_rows:
        eps = row["epsilon"]
        eps_str = f"{eps:.1f}" if math.isfinite(eps) else "inf"
        print(
            f"{row['generator']:<10} {eps_str:>6}  "
            f"{row['n_records']:>2}/{row['expected_n']:<2}  {str(row['complete']):>6}  "
            f"{_fmt_mean_std(row['icl_gap_mean'], row['icl_gap_std']):>17}  "
            f"{_fmt_mean_std(row['icl_gap_tabicl_mean'], row['icl_gap_tabicl_std']):>17}  "
            f"{_fmt_mean_std(row['mean_js_mean'], row['mean_js_std']):>17}  "
            f"{_fmt_mean_std(row['max_js_mean'], row['max_js_std']):>17}"
        )

    print("\n" + "=" * 120)
    print("EXP-PRIVACY Summary — Fidelity Metrics (mean±std across datasets × samples)")
    print("=" * 120)
    hdr2 = (
        f"{'Generator':<10} {'eps':>6}  {'KS':>17}  {'JSD':>17}  "
        f"{'WD-1D':>17}  {'SWD':>17}  {'MMD':>17}  {'NFN':>17}"
    )
    print(hdr2)
    print("-" * 120)
    for row in condition_rows:
        eps = row["epsilon"]
        eps_str = f"{eps:.1f}" if math.isfinite(eps) else "inf"
        print(
            f"{row['generator']:<10} {eps_str:>6}  "
            f"{_fmt_mean_std(row['fid_ks_mean'], row['fid_ks_std']):>17}  "
            f"{_fmt_mean_std(row['fid_jsd_mean'], row['fid_jsd_std']):>17}  "
            f"{_fmt_mean_std(row['fid_wd_1d_mean'], row['fid_wd_1d_std']):>17}  "
            f"{_fmt_mean_std(row['fid_wd_sliced_mean'], row['fid_wd_sliced_std']):>17}  "
            f"{_fmt_mean_std(row['fid_mmd_mean'], row['fid_mmd_std']):>17}  "
            f"{_fmt_mean_std(row['fid_nfn_mean'], row['fid_nfn_std']):>17}"
        )

    print("\n" + "=" * 120)
    print("EXP-PRIVACY Summary — Utility & Coverage (mean±std across datasets × samples)")
    print("=" * 120)
    hdr3 = (
        f"{'Generator':<10} {'eps':>6}  {'TSTR-XGB':>17}  {'TSTR-MLP':>17}  "
        f"{'TSTR-RF':>17}  {'α-Prec':>17}  {'β-Rec':>17}"
    )
    print(hdr3)
    print("-" * 120)
    for row in condition_rows:
        eps = row["epsilon"]
        eps_str = f"{eps:.1f}" if math.isfinite(eps) else "inf"
        print(
            f"{row['generator']:<10} {eps_str:>6}  "
            f"{_fmt_mean_std(row['fid_tstr_auc_mean'], row['fid_tstr_auc_std']):>17}  "
            f"{_fmt_mean_std(row['fid_tstr_mlp_mean'], row['fid_tstr_mlp_std']):>17}  "
            f"{_fmt_mean_std(row['fid_tstr_rf_mean'], row['fid_tstr_rf_std']):>17}  "
            f"{_fmt_mean_std(row['fid_alpha_precision_mean'], row['fid_alpha_precision_std']):>17}  "
            f"{_fmt_mean_std(row['fid_beta_recall_mean'], row['fid_beta_recall_std']):>17}"
        )


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_tradeoff(results: list[dict], out_dir: Path) -> None:
    """Fig 7: three-panel eps-vs-metric plots for each DP generator."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib unavailable, skipping plots")
        return

    # Aggregate: mean over datasets × seeds per (gen, epsilon, metric)
    agg: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    all_keys = [rec_key for _, rec_key, _ in METRIC_DISPLAY]
    for r in results:
        k = (r["generator"], r["epsilon"])
        for rec_key in all_keys:
            v = r.get(rec_key, float("nan"))
            if math.isfinite(v):
                agg[k][rec_key].append(v)

    def get_curve(gen: str, rec_key: str) -> tuple[list[float], list[float]]:
        """Return (plot_x, plot_y) including CTGAN reference at x=50 (proxy for inf)."""
        eps_sorted = sorted([e for (g, e) in agg if g == gen and math.isfinite(e)])
        vals = [
            np.mean(agg[(gen, e)][rec_key]) if agg[(gen, e)].get(rec_key) else float("nan")
            for e in eps_sorted
        ]
        ref = agg[("ctgan", float("inf"))].get(rec_key, [])
        ref_val = float(np.mean(ref)) if ref else float("nan")
        return eps_sorted + [50.0], vals + [ref_val]

    # ── Panel definitions ─────────────────────────────────────────────────────
    panels: list[tuple[str, list[tuple[str, str, str, str, float, bool]]]] = [
        (
            "ICL Metrics",
            [
                ("ICL-Gap (TabPFN)", "icl_gap",         "#d62728", "-",  2.0, False),
                ("ICL-Gap (TabICL)", "icl_gap_tabicl",  "#e377c2", "--", 1.8, False),
                ("mean_JS",          "mean_js",          "#9467bd", "-",  1.5, False),
                ("max_JS",           "max_js",           "#8c564b", ":",  1.5, False),
            ],
        ),
        (
            "Fidelity",
            [
                ("KS",    "fid_ks",          "#1f77b4", "-",  1.5, False),
                ("JSD",   "fid_jsd",         "#17becf", "--", 1.2, False),
                ("WD-1D", "fid_wd_1d",       "#aec7e8", ":",  1.2, False),
                ("SWD",   "fid_wd_sliced",   "#0d5f8a", "-.", 1.2, False),
                ("MMD",   "fid_mmd",         "#6baed6", "--", 1.2, False),
                ("NFN",   "fid_nfn",         "#2171b5", ":",  1.2, False),
            ],
        ),
        (
            "Utility & Coverage",
            [
                ("TSTR-XGB",   "fid_tstr_auc",          "#ff7f0e", "-",  1.8, True),
                ("TSTR-MLP",   "fid_tstr_mlp",          "#ffbb78", "--", 1.4, True),
                ("TSTR-RF",    "fid_tstr_rf",           "#d62728", ":",  1.4, True),
                ("α-Precision","fid_alpha_precision",   "#2ca02c", "-",  1.4, True),
                ("β-Recall",   "fid_beta_recall",       "#98df8a", "--", 1.4, True),
            ],
        ),
    ]

    ICL_GAP_THRESHOLD = 0.05

    for gen_name in DP_GENERATORS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (panel_title, metric_styles) in zip(axes, panels):
            for label, rec_key, color, ls, lw, higher_better in metric_styles:
                xs, ys = get_curve(gen_name, rec_key)
                # For higher-is-better metrics, invert so all curves read "lower = worse"
                plot_ys = [
                    (1.0 - v) if (higher_better and math.isfinite(v)) else v
                    for v in ys
                ]
                ax.plot(xs, plot_ys, label=label,
                        color=color, ls=ls, lw=lw, marker="o", markersize=4)

            if panel_title == "ICL Metrics":
                ax.axhline(ICL_GAP_THRESHOLD, color="#d62728", lw=0.8, ls="--",
                           alpha=0.5, label="ICL-Gap threshold=0.05")

            ax.axvline(50.0, color="gray", lw=0.8, ls=":", alpha=0.5)
            ax.set_xscale("log")
            ax.set_xlim(0.08, 60)
            ax.set_xticks([0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
            ax.set_xticklabels(["0.1", "0.5", "1", "5", "10", "∞(CTGAN)"], fontsize=8)
            ax.set_xlabel("Privacy budget ε (log scale)", fontsize=9)
            ax.set_ylabel("Score (lower = worse quality)", fontsize=9)
            ax.set_title(f"{panel_title}", fontsize=10)
            ax.legend(fontsize=7, loc="best", ncol=1)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{gen_name.upper()} — Privacy-Utility Tradeoff (all 15 metrics)\n"
            f"mean over {len(DEFAULT_DATASETS)} datasets × {len(DEFAULT_SEEDS)} seeds; "
            "higher-is-better metrics inverted",
            fontsize=10,
        )
        fig.tight_layout()
        out_path = out_dir / f"privacy_tradeoff_{gen_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log.info(f"Saved → {out_path}")

    # ── Aggregated 2-generator comparison (DPGAN vs PATEGAN, ICL metrics only) ─
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    icl_styles = [
        ("ICL-Gap (TabPFN)", "icl_gap",        "#d62728", "-",  2.0, False),
        ("ICL-Gap (TabICL)", "icl_gap_tabicl", "#e377c2", "--", 1.8, False),
        ("mean_JS",          "mean_js",         "#9467bd", "-",  1.5, False),
        ("max_JS",           "max_js",          "#8c564b", ":",  1.5, False),
        ("TSTR-XGB",         "fid_tstr_auc",    "#ff7f0e", "-",  1.5, True),
        ("KS",               "fid_ks",          "#1f77b4", "--", 1.2, False),
        ("α-Precision",      "fid_alpha_precision", "#2ca02c", "-.", 1.2, True),
    ]
    for ax, gen_name in zip(axes2, DP_GENERATORS):
        for label, rec_key, color, ls, lw, higher in icl_styles:
            xs, ys = get_curve(gen_name, rec_key)
            plot_ys = [(1.0 - v) if (higher and math.isfinite(v)) else v for v in ys]
            ax.plot(xs, plot_ys, label=label,
                    color=color, ls=ls, lw=lw, marker="o", markersize=4)
        ax.axhline(ICL_GAP_THRESHOLD, color="#d62728", lw=0.8, ls="--",
                   alpha=0.5, label="ICL-Gap threshold=0.05")
        ax.axvline(50.0, color="gray", lw=0.8, ls=":", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlim(0.08, 60)
        ax.set_xticks([0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
        ax.set_xticklabels(["0.1", "0.5", "1", "5", "10", "∞(CTGAN)"], fontsize=8)
        ax.set_xlabel("Privacy budget ε (log scale)", fontsize=9)
        ax.set_ylabel("Score (lower = worse)", fontsize=9)
        ax.set_title(f"{gen_name.upper()}", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig2.suptitle(
        "EXP-PRIVACY: Key Metrics Overview — DPGAN vs PATEGAN\n"
        "(Fig 7 candidate: 7 representative metrics, mean over 4 datasets × 3 seeds)",
        fontsize=10,
    )
    fig2.tight_layout()
    out2 = out_dir / "privacy_tradeoff_key_metrics.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    log.info(f"Saved → {out2}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-PRIVACY: Privacy-utility tradeoff")
    p.add_argument("--datasets",   default=",".join(DEFAULT_DATASETS))
    p.add_argument("--seeds",      default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--epsilons",   default=",".join(map(str, EPSILON_VALUES)))
    p.add_argument("--device",     default="cuda")
    p.add_argument("--n_iter",     type=int, default=300)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--fit_seed",   type=int, default=0)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )
    args     = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",")]
    seeds    = [int(s)    for s in args.seeds.split(",")]
    epsilons = [float(e)  for e in args.epsilons.split(",")]

    log.info(f"Datasets  : {datasets}")
    log.info(f"Seeds     : {seeds}")
    log.info(f"Fit seed  : {args.fit_seed}")
    log.info(f"Epsilons  : {epsilons}")
    log.info(f"DP gens   : {DP_GENERATORS}")
    log.info(f"Ref gens  : {REF_GENERATORS}")
    log.info(f"Device    : {args.device}")
    log.info(f"Metrics   : {len(ALL_METRIC_KEYS)} total ({len(ALL_ICL_KEYS)} ICL + "
             f"{len(FIDELITY_METRIC_KEYS)} fidelity)")

    # Allow overriding module-level list so plots use correct epsilons
    global EPSILON_VALUES
    EPSILON_VALUES = epsilons

    results = run_experiment(
        datasets=datasets,
        seeds=seeds,
        epsilon_values=epsilons,
        device=args.device,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        fit_seed=args.fit_seed,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = RESULTS_BASE / "exp_privacy_v3" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results → {out_file}  ({len(results)} records)")

    # Remove partial checkpoint
    partial = RESULTS_BASE / "exp_privacy_v3" / "results_partial.json"
    if partial.exists():
        partial.unlink()

    condition_rows, dataset_condition_rows = write_summary_artifacts(
        results=results,
        out_dir=out_dir,
        datasets=datasets,
        seeds=seeds,
    )
    print_summary(condition_rows, dataset_condition_rows)
    plot_tradeoff(results, out_dir)

    log.info(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
