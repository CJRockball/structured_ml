# utils_reproducibility.py
import random
import numpy as np
import torch
from config import global_cfg

def set_seed() -> None:
    s = global_cfg.seed
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if global_cfg.device == "cuda":
        torch.cuda.manual_seed_all(s)
    print(f"[seed] global seed set to {s} on {global_cfg.device}")