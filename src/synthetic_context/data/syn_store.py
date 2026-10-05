"""
Script: syn_store.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SYN_STORE_DIR = Path("./data/synthetic")


class SyntheticDataStore:
    def __init__(self, base_dir: Path | str = SYN_STORE_DIR):
        self.base_dir = Path(base_dir)

    def path(self, generator: str, dataset: str, sample_idx: int) -> Path:
        return self.base_dir / generator / f"{dataset}_sample{sample_idx}.npz"

    def exists(self, generator: str, dataset: str, sample_idx: int) -> bool:
        return self.path(generator, dataset, sample_idx).exists()

    def all_exist(self, generator: str, dataset: str, n_samples: int = 5) -> bool:
        return all(self.exists(generator, dataset, i) for i in range(n_samples))

    def save(
        self,
        generator: str,
        dataset: str,
        sample_idx: int,
        X_syn: np.ndarray,
        y_syn: np.ndarray,
    ) -> None:
        p = self.path(generator, dataset, sample_idx)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, X_syn=X_syn.astype(np.float32), y_syn=y_syn)
        log.info(f"[SynStore] saved {p.name}  shape={X_syn.shape}")

    def load(
        self, generator: str, dataset: str, sample_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        p = self.path(generator, dataset, sample_idx)
        data = np.load(p, allow_pickle=True)
        return data["X_syn"], data["y_syn"]

    def load_all(
        self, generator: str, dataset: str, n_samples: int = 5
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return [self.load(generator, dataset, i) for i in range(n_samples)]

    def summary(self) -> dict[str, dict[str, int]]:
        """Return {generator: {dataset: n_cached_samples}} for all cached files."""
        result: dict[str, dict[str, int]] = {}
        for gen_dir in sorted(self.base_dir.iterdir()):
            if not gen_dir.is_dir():
                continue
            result[gen_dir.name] = {}
            for npz in sorted(gen_dir.glob("*.npz")):
                # filename: {dataset}_sample{i}.npz
                parts = npz.stem.rsplit("_sample", 1)
                if len(parts) == 2:
                    ds = parts[0]
                    result[gen_dir.name][ds] = result[gen_dir.name].get(ds, 0) + 1
        return result
