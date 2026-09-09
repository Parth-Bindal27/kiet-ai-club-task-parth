"""Seeding utilities for reproducibility."""
import random
import numpy as np
try:
    import torch
except ImportError:
    torch = None

def set_seed(seed: int) -> None:
    """Set seed for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
