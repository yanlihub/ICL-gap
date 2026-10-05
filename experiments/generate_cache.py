"""
Generate all synthetic data for the benchmark cache.

Design
------
- 10 datasets × 8 non-DP generators
- For each (dataset, generator): fit ONCE (seed=0), sample 5 times (seed=0..4)
- Output: ./data/synthetic/{generator}/{dataset}_sample{i}.npz
- Idempotent: skips already-cached combinations
- Order: fast generators first, TabSyn last (smallest → largest dataset)

N/A combinations (skipped, no file created):
- SMOTE × regression datasets (california_housing, news, kin8nm)
- TabPFNGen × very large / high-dim datasets (adult, default, news)

Usage
-----
    python experiments/generate_cache.py                        # all
    python experiments/generate_cache.py --generators tabsyn    # one generator
    python experiments/generate_cache.py --datasets breast_cancer,diabetes
    python experiments/generate_cache.py --status               # print cache status
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synthetic_context.data.loader import DATASET_REGISTRY, DatasetLoader, get_task_type
from synthetic_context.data.syn_store import SYN_STORE_DIR, SyntheticDataStore
from synthetic_context.generators import get_generator
from synthetic_context.utils.seeds import set_all_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

BENCHMARK_10 = [
    # Classification (7)
    "breast_cancer", "diabetes", "credit-g",      # small, already have TabSyn ckpts
    "kin8nm",                                       # small regression
    "australian", "magic", "california_housing",   # medium
    "default",                                      # large clf
    "adult",                                        # large clf
    "news",                                         # large regression, high-dim
]

NON_DP_GENERATORS = [
    # Fast first
    "random", "marginal", "gmm", "smote",
    "ctgan", "tvae", "tabpfngen",
    # Slow last
    "tabsyn",
]

REGRESSION_DATASETS = {"california_housing", "news", "kin8nm"}

# Generator × dataset combinations that are N/A (no synthetic data produced)
NA_COMBINATIONS: set[tuple[str, str]] = set()
# SMOTE: regression uses k-NN interpolation (_generate_regression), preserves X-y joint structure.
# TabPFNGen: regression supported via TabPFNRegressor for label assignment.
# TabPFNGen: large datasets (adult, default) use max_fit_samples=2000 subsampling.
# TabSyn: regression supported via numerical target column.
# → No combinations are N/A; all generators can handle all datasets.

# TabSyn kwargs: stable_name enables checkpoint reuse across runs
TABSYN_KWARGS = {
    ds: {"stable_name": f"{ds}_{DATASET_REGISTRY.get(ds, ds)}", "seed": 0}
    for ds in BENCHMARK_10
}
# Datasets without an openml ID use the name itself
TABSYN_KWARGS["california_housing"] = {"stable_name": "california_housing_ca", "seed": 0}
TABSYN_KWARGS["news"] = {"stable_name": "news_uci", "seed": 0}

CTGAN_TVAE_KWARGS = {"n_iter": 300, "batch_size": 500, "seed": 0}

N_SAMPLES = 5  # number of synthetic datasets per (generator, dataset) combination


# ── Per-generator kwargs ────────────────────────────────────────────────────────

def _gen_kwargs(gen_name: str, dataset: str) -> dict:
    if gen_name == "tabsyn":
        return TABSYN_KWARGS[dataset]
    if gen_name in ("ctgan", "tvae"):
        return CTGAN_TVAE_KWARGS
    return {"seed": 0}


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_all(
    generators: list[str],
    datasets: list[str],
    store: SyntheticDataStore,
    data_dir: Path,
) -> None:
    loader = DatasetLoader(cache_dir=data_dir)

    total = sum(
        1 for g in generators for d in datasets
        if (g, d) not in NA_COMBINATIONS
    )
    done = 0

    for gen_name in generators:
        for dataset in datasets:
            if (gen_name, dataset) in NA_COMBINATIONS:
                log.info(f"SKIP  {gen_name:12s} × {dataset}  [N/A]")
                continue

            if store.all_exist(gen_name, dataset, N_SAMPLES):
                log.info(f"HIT   {gen_name:12s} × {dataset}  [all {N_SAMPLES} samples cached]")
                done += 1
                continue

            log.info(f"\n{'='*60}")
            log.info(f"GEN   {gen_name:12s} × {dataset}  ({done+1}/{total})")

            try:
                X_train, X_test, y_train, y_test = loader.load(dataset)
            except Exception as e:
                log.error(f"  Failed to load {dataset}: {e}")
                continue

            n_train = len(y_train)
            log.info(f"  n_train={n_train}  task={get_task_type(dataset)}")

            # ── Fit ONCE (seed=0) ──────────────────────────────────────────
            kwargs = _gen_kwargs(gen_name, dataset)
            t0 = time.perf_counter()
            try:
                set_all_seeds(0)
                generator = get_generator(gen_name, **kwargs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    generator.fit(X_train, y_train)
                fit_time = time.perf_counter() - t0
                log.info(f"  fit done in {fit_time/60:.1f} min")
            except Exception as e:
                log.error(f"  fit FAILED: {e}")
                continue

            # ── Sample N_SAMPLES times ─────────────────────────────────────
            for i in range(N_SAMPLES):
                if store.exists(gen_name, dataset, i):
                    log.info(f"  sample {i}: already cached, skip")
                    continue
                try:
                    set_all_seeds(i)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        X_syn, y_syn = generator.generate(n_train)
                    store.save(gen_name, dataset, i, X_syn, y_syn)
                except Exception as e:
                    log.error(f"  sample {i} FAILED: {e}")

            done += 1
            log.info(f"  [{done}/{total}] done")


def print_status(store: SyntheticDataStore) -> None:
    summary = store.summary()
    print(f"\n{'='*65}")
    print(f"  SyntheticDataStore status — {store.base_dir}")
    print(f"{'='*65}")
    print(f"{'Generator':<14} {'Dataset':<22} {'Samples':>7}")
    print("-" * 45)
    for gen in NON_DP_GENERATORS:
        ds_counts = summary.get(gen, {})
        for ds in BENCHMARK_10:
            if (gen, ds) in NA_COMBINATIONS:
                status = "N/A"
            else:
                n = ds_counts.get(ds, 0)
                status = f"{n}/{N_SAMPLES}" + (" ✓" if n == N_SAMPLES else " ✗")
            print(f"  {gen:<12}  {ds:<22}  {status:>7}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generators", default=",".join(NON_DP_GENERATORS),
                        help="Comma-separated list of generators")
    parser.add_argument("--datasets", default=",".join(BENCHMARK_10),
                        help="Comma-separated list of datasets")
    parser.add_argument("--store-dir", default=str(SYN_STORE_DIR))
    parser.add_argument("--data-dir",
                        default="./data")
    parser.add_argument("--status", action="store_true",
                        help="Print cache status and exit")
    args = parser.parse_args()

    store = SyntheticDataStore(args.store_dir)

    if args.status:
        print_status(store)
        sys.exit(0)

    generators = [g.strip() for g in args.generators.split(",")]
    datasets   = [d.strip() for d in args.datasets.split(",")]

    log.info(f"Generators : {generators}")
    log.info(f"Datasets   : {datasets}")
    log.info(f"Store      : {args.store_dir}")

    generate_all(generators, datasets, store, Path(args.data_dir))

    log.info("\nFinal status:")
    print_status(store)
