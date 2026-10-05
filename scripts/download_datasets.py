"""Download the two non-OpenML regression datasets to ./data/raw/{name}.csv.

The other eight benchmark datasets are fetched from OpenML automatically by
DatasetLoader. Run from the repository root:

    python scripts/download_datasets.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

RAW_DIR = Path("./data/raw")
NEWS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"
NEWS_MEMBER = "OnlineNewsPopularity/OnlineNewsPopularity.csv"


def save(X: pd.DataFrame, y: pd.Series, name: str) -> None:
    path = RAW_DIR / f"{name}.csv"
    pd.concat([X, y.rename("target")], axis=1).to_csv(path, index=False)
    print(f"Saved {path}  shape={X.shape}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    housing = fetch_california_housing(as_frame=True)
    save(housing.data.reset_index(drop=True), housing.target.reset_index(drop=True), "california_housing")

    with urllib.request.urlopen(NEWS_URL, timeout=300) as resp:
        with zipfile.ZipFile(io.BytesIO(resp.read())) as zf, zf.open(NEWS_MEMBER) as f:
            df = pd.read_csv(f)
    df = df.dropna(subset=[" shares"]).reset_index(drop=True)  # column names keep UCI's leading space
    save(df.drop(columns=[" shares", "url", " timedelta"]), df[" shares"], "news")


if __name__ == "__main__":
    main()
