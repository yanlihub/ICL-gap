# ICL-Gap: Tabular Foundation Models as Zero-Shot Evaluators of Synthetic Data Quality

Official code for *Beyond TSTR: Tabular Foundation Models as Zero-Shot Evaluators of Synthetic Data Quality* (NeurIPS 2026 Workshop TAE).

ICL-Gap replaces the trainable downstream model of Train-on-Synthetic-Test-on-Real (TSTR) with a fixed pre-trained tabular foundation model (TabPFN v2 / TabICL v2): `ICL-Gap = AUC(TFM; real context) - AUC(TFM; synthetic context)`.

## Installation

Python 3.12 (tested); a GPU is recommended. `requirements.txt` pins the versions used for the paper.

```bash
git clone https://github.com/yanlihub/ICL-gap.git
cd ICL-gap
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export SCIPY_ARRAY_API=1   # needed by recent scikit-learn/scipy combinations

# Regression datasets not on OpenML (california_housing, news) -> ./data/raw/
python scripts/download_datasets.py

# Optional, only for the TabSyn generator: clone TabSyn into ./tabsyn
git clone https://github.com/amazon-science/tabsyn.git tabsyn
pip install icecream
# torch >= 2.7 removed ReduceLROnPlateau(verbose=...); drop the argument:
sed -i 's/, verbose=True//' tabsyn/tabsyn/main.py tabsyn/tabsyn/vae/main.py
```

With `tabpfn==6.4.1` the default TabPFN checkpoint is TabPFN-2.5, whose weights are gated on Hugging Face: accept the model license at https://huggingface.co/Prior-Labs and authenticate (`hf auth login`) before the first run.

The eight classification/regression datasets hosted on OpenML are downloaded automatically on first use and cached under `./data/`.

## Reproducing the paper

Run all scripts from the repository root. First generate the synthetic samples: each generator is fitted once per dataset and sampled five times (seeds 0-4), cached under `./data/synthetic/`:

```bash
python experiments/generate_cache.py                      # all generators x datasets
python experiments/generate_cache.py --generators smote --datasets australian   # subset
```

Each experiment script accepts `--help` and `--smoke_test` (minimal configuration for a quick check).

| Script | Paper |
|---|---|
| `experiments/run_tstr_imbalance.py` | Sec. 4 (TSTR rank inversion, SMOTE-rho ablation), Table 1, Fig. 1, App. A |
| `experiments/run_quality_spectrum.py` | Sec. 5 (Table 2, evaluator invariance), App. regression / invariance tables, metric-correlation heatmap |
| `experiments/run_corruption.py` | Sec. 6 (C1-C7 corruption experiments), corruption figure |
| `experiments/run_pda.py` | App. ICL-PDA |
| `experiments/run_privacy.py` | App. privacy experiments (PATE-GAN, DPGAN) |

Results are written to `./results/`.

## Datasets

| Key | Source | n | d |
|---|---|---|---|
| `breast_cancer` | OpenML 15 | 699 | 9 |
| `diabetes` | OpenML 37 | 768 | 8 |
| `credit-g` | OpenML 31 | 1,000 | 20 |
| `australian` | OpenML 40981 | 690 | 14 |
| `magic` | OpenML 1120 | 19,020 | 10 |
| `default` | OpenML 42477 | 30,000 | 23 |
| `adult` | OpenML 1590 | 48,842 | 14 |
| `kin8nm` | OpenML 189 | 8,192 | 8 |
| `california_housing` | scikit-learn | 20,640 | 8 |
| `news` | UCI Online News Popularity | 39,644 | 58 |

Note: earlier versions of this code referred to OpenML 40981 (Statlog Australian Credit Approval) under the key `shoppers`. It is the same dataset reported as *australian* in the paper; `shoppers` is kept as an alias.

## Repository structure

- `src/synthetic_context/data/`: OpenML loading, preprocessing (skrub `TableVectorizer`, ordinal encoding, fixed 80/20 split, `random_state=42`), synthetic-sample cache
- `src/synthetic_context/generators/`: Random, Marginal, GMM, SMOTE, CTGAN, TVAE, PATE-GAN, DPGAN (synthcity), TabSyn, TabPFNGen
- `src/synthetic_context/evaluators/icl.py`: ICL evaluation with TabPFN / TabICL
- `src/synthetic_context/utils/`: fidelity metrics, corruptions, seeding
- `experiments/`: `generate_cache.py` plus scripts reproducing the paper
