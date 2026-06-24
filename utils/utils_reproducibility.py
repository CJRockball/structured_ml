# utils_reproducibility.py
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
import torch

from config import (
    GlobalConfig, global_cfg,          
    FeatureConfig, features,
    CompetitionConfig, DataConfig, XGBConfig, CVConfig,
    comp_cfg, data_cfg, xgb_cfg, cv_cfg,
)


def set_seed() -> None:
    s = global_cfg.seed
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if global_cfg.device == "cuda":
        torch.cuda.manual_seed_all(s)
    print(f"[seed] global seed set to {s} on {global_cfg.device}")
    return
    
    
def get_folds(df_X:pd.DataFrame, df_y:pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(
        n_splits=cv_cfg.n_folds, 
        shuffle=True, 
        random_state=cv_cfg.seed)
    return list(skf.split(df_X, df_y))