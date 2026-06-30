"""
tune_xgb.py  —  Named parameter sweep for xgb1.py
Calls lgb1.py as a subprocess for each theme.
All logging and result output is handled by xgb1.py.

Usage:
    python tune_xgb.py
"""

import subprocess
import sys
from pathlib import Path

# ============================================================
# DATA  —  set once, applied to all themes
# ============================================================

TRAIN_FILE = "s6e6_star_train_subj_know_20260629.parquet" #None   # e.g. "s6e6_star_train_num_fe_20260611.parquet"
TEST_FILE  = "s6e6_star_test_subj_know_20260629.parquet" #None   # e.g. "s6e6_star_test_num_fe_20260611.parquet"
version = "ve4_no_te_target_weight"
model_name = "cat1"
# ============================================================
# THEMES
# ============================================================

THEMES = {
#     "base fe": {},
    # LGB_v1 — High num_leaves, Optuna-inspired (from s6e6 XGB+LGBM+CAT, best LGBM config)
    # CAT_v1 — Optuna-tuned: shallow depth, very low l2, high border_count, strong random_strength
# (from CatBoost AUC .96 notebook, Optuna best trial)
    "cat_params_v1": {
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1:average=Macro',
        'task_type': 'GPU',
        'auto_class_weights': 'Balanced',
        'learning_rate': 0.0644,
        'depth': 5,
        'l2_leaf_reg': 0.0121,
        'min_data_in_leaf': 52,
        'bagging_temperature': 0.0691,
        'random_strength': 1.6012,
        'border_count': 153,
        'iterations': 8000,
        'early_stopping_rounds': 100,
        'random_seed': 42,
        'verbose': 200,
    },

    # CAT_v2 — Deeper tree, stronger l2, long run (from s6e6 XGB+LGBM+CAT + GBM Ensemble)
    "cat_params_v2": {
        'loss_function': 'MultiClass',
        'eval_metric': 'TotalF1:average=Macro',
        'task_type': 'GPU',
        'auto_class_weights': 'Balanced',
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 5.0,
        'min_data_in_leaf': 20,
        'bagging_temperature': 0.5,
        'random_strength': 1.0,
        'border_count': 254,
        'iterations': 8000,
        'early_stopping_rounds': 150,
        'random_seed': 0,
        'verbose': 200,
    },
}



# ============================================================
# ARG MAP  —  param name → xgb1.py CLI flag
# ============================================================

ARG_MAP = {
    "learning_rate":        "--lr",
    "iterations":           "--iterations",
    "depth":                "--depth",
    "min_data_in_leaf":     "--min-data-in-leaf",
    "l2_leaf_reg":          "--l2-leaf-reg",
    "random_strength":      "--random-strength",
    "bagging_temperature":  "--bagging-temperature",
}


# ============================================================
# RUN
# ============================================================

XGB_SCRIPT = Path(__file__).parent / f"{model_name}.py"

for theme_name, params in THEMES.items():
    cmd = [sys.executable, str(XGB_SCRIPT),
           "--exp-name", f"{model_name}_tune_{theme_name}_{version}"]

    if TRAIN_FILE:
        cmd += ["--train-file", TRAIN_FILE]
    if TEST_FILE:
        cmd += ["--test-file", TEST_FILE]

    for param, flag in ARG_MAP.items():
        if param in params:
            cmd += [flag, str(params[param])]

    subprocess.run(cmd, check=True)