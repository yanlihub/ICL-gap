# ICL-Gap: Tabular Foundation Models as Zero-Shot Judges of Synthetic Data Quality

This repository contains the official implementation of the ICL-Gap evaluation framework for synthetic tabular data. 

Unlike traditional Train-Synthetic-Test-Real (TSTR) evaluation, ICL-Gap uses pre-trained Tabular Foundation Models to robustly rank synthetic data generators without requiring downstream model training.

## Installation

We recommend using a virtual environment. The code requires Python 3.9+ and PyTorch.

```bash
# Clone the repository
git clone https://github.com/your-username/icl-gap.git
cd icl-gap

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Repository Structure

- `src/synthetic_context/`: Core library code including the `ICLEvaluator`, data loaders, and metric implementations.
- `experiments/`: Scripts to reproduce the core results.

## Usage

To run the experiments, execute the Python scripts in the `experiments/` directory. For example, to run the corruption sensitivity analysis:

```bash
python experiments/run_corruption.py
```

### Running a specific experiment

Each script in the `experiments/` folder corresponds to a different analysis (e.g., class imbalance, privacy tradeoff, distributional corruptions). You can typically run them directly. Note that caching and data directories should be configured in the script or via command-line arguments depending on your local setup.
