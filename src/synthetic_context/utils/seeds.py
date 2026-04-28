import os
import random

import numpy as np
import torch

SEEDS = [0, 1, 2, 3, 4]


def set_all_seeds(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
