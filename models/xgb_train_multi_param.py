"""
tune_xgb.py  —  Named parameter sweep for xgb1.py
Calls xgb1.py as a subprocess for each theme.
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
model_name = 'xgb1'
# ============================================================
# THEMES
# ============================================================

# THEMES = {
#     "base fe": {},

#     "deep_slow": {
#         "max_depth":        8,
#         "min_child_weight": 5,
#         "learning_rate":    0.01,
#         "n_estimators":     4000,
#         "subsample":        0.8,
#         "colsample_bytree": 0.8,
#     },

#     "shallow_fast": {
#         "max_depth":        4,
#         "min_child_weight": 1,
#         "learning_rate":    0.1,
#         "n_estimators":     800,
#         "subsample":        0.9,
#         "colsample_bytree": 0.9,
#     },

#     "regularised": {
#         "max_depth":        6,
#         "min_child_weight": 10,
#         "learning_rate":    0.05,
#         "n_estimators":     2000,
#         "reg_alpha":        0.5,
#         "reg_lambda":       5.0,
#         "subsample":        0.75,
#         "colsample_bytree": 0.75,
#     },

#     "stochastic": {
#         "max_depth":        6,
#         "min_child_weight": 3,
#         "learning_rate":    0.05,
#         "n_estimators":     2000,
#         "subsample":        0.6,
#         "colsample_bytree": 0.6,
#         "reg_alpha":        0.1,
#         "reg_lambda":       1.0,
#     },

#     "balanced": {
#         "max_depth":        6,
#         "min_child_weight": 5,
#         "learning_rate":    0.03,
#         "n_estimators":     3000,
#         "subsample":        0.8,
#         "colsample_bytree": 0.8,
#         "reg_alpha":        0.1,
#         "reg_lambda":       2.0,
#     },
# }
# THEMES = {
#     "base_fe": {
#         # Mild regularization, fast to train, good for checking
#         "max_depth":         4,
#         "min_child_weight":  2,
#         "learning_rate":     0.08,
#         "n_estimators":      600,
#         "subsample":         0.9,
#         "colsample_bytree":  0.9,
#         "reg_alpha":         0.0,
#         "reg_lambda":        1.0,
#     },

#     "deep_slow": {
#         # Your original idea: deep + tiny LR, heavy L1/L2
#         "max_depth":         8,
#         "min_child_weight":  4,
#         "learning_rate":     0.01,
#         "n_estimators":      4000,
#         "subsample":         0.8,
#         "colsample_bytree":  0.8,
#         "reg_alpha":         2.0,
#         "reg_lambda":        4.0,
#     },

#     "wide_fast": {
#         # Shallower trees, bigger LR, more trees than default
#         # Good baseline for tabular Kaggle problems
#         "max_depth":         5,
#         "min_child_weight":  1,
#         "learning_rate":     0.15,
#         "n_estimators":      500,
#         "subsample":         0.9,
#         "colsample_bytree":  0.9,
#         "reg_alpha":         0.0,
#         "reg_lambda":        1.0,
#     },

#     "regularized": {
#         # Stronger shrinkage + L2, good if you see overfitting
#         "max_depth":         5,
#         "min_child_weight":  3,
#         "learning_rate":     0.05,
#         "n_estimators":      1200,
#         "subsample":         0.7,
#         "colsample_bytree":  0.7,
#         "reg_alpha":         0.5,
#         "reg_lambda":        3.0,
#     },

#     "balanced_focus": {
#         # Slightly deeper, moderate LR, a bit more child weight
#         # Often useful for imbalanced multi‑class with sample_weight
#         "max_depth":         6,
#         "min_child_weight":  4,
#         "learning_rate":     0.06,
#         "n_estimators":      1500,
#         "subsample":         0.8,
#         "colsample_bytree":  0.8,
#         "reg_alpha":         0.3,
#         "reg_lambda":        2.0,
#     },
# }

THEMES = {
    # XGB_v1 — Optuna-tuned, deep + aggressive regularization (from s6e6 XGB+LGBM+CAT, ~0.9665)
    "xgb_params_v1": {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'enable_categorical': True,
        'learning_rate': 0.0211,
        'max_depth': 10,
        'min_child_weight': 1,
        'max_delta_step': 2,
        'gamma': 0.6299,
        'reg_alpha': 5.1797,
        'reg_lambda': 5.6898,
        'subsample': 0.8825,
        'colsample_bytree': 0.5707,
        'seed': 42,
    },

    # XGB_v2 — Conservative depth, heavy L2, stable generalization (from GBM Ensemble & Tuning)
    "xgb_params_v2": {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'enable_categorical': True,
        'learning_rate': 0.015,
        'max_depth': 8,
        'min_child_weight': 5,
        'max_delta_step': 1,
        'gamma': 0.2,
        'reg_alpha': 0.5,
        'reg_lambda': 2.5,
        'subsample': 0.75,
        'colsample_bytree': 0.70,
        'colsample_bylevel': 0.80,
        'n_estimators': 5000,
        'early_stopping_rounds': 150,
        'seed': 0,
    },

    # XGB_v3 — Long training, low LR, deeper trees, moderate regs (from Stellar Class Prediction S6E6)
    "xgb_params_v3": {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'enable_categorical': True,
        'learning_rate': 0.008,
        'max_depth': 8,
        'min_child_weight': 3,
        'gamma': 1.1,
        'reg_alpha': 0.125,
        'reg_lambda': 0.13,
        'subsample': 0.85,
        'colsample_bytree': 0.80,
        'n_estimators': 7000,
        'early_stopping_rounds': 300,
        'seed': 1,
    },

    # XGB_v4 — Shallow + wide, high subsampling, different seed for diversity
    "xgb_params_v4": {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'enable_categorical': True,
        'learning_rate': 0.02,
        'max_depth': 6,
        'min_child_weight': 10,
        'max_delta_step': 1,
        'gamma': 0.05,
        'reg_alpha': 1.0,
        'reg_lambda': 3.0,
        'subsample': 0.90,
        'colsample_bytree': 0.65,
        'colsample_bylevel': 0.75,
        'n_estimators': 6000,
        'early_stopping_rounds': 200,
        'seed': 99,
    },

    # XGB_v5 — Moderate depth, balanced regs, higher column dropout (diversity variant)
    "xgb_params_v5": {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'enable_categorical': True,
        'learning_rate': 0.012,
        'max_depth': 9,
        'min_child_weight': 3,
        'max_delta_step': 2,
        'gamma': 0.35,
        'reg_alpha': 2.5,
        'reg_lambda': 4.0,
        'subsample': 0.80,
        'colsample_bytree': 0.55,
        'colsample_bylevel': 0.85,
        'colsample_bynode': 0.80,
        'n_estimators': 6500,
        'early_stopping_rounds': 200,
        'seed': 7,
    },
}



# ============================================================
# ARG MAP  —  param name → xgb1.py CLI flag
# ============================================================

ARG_MAP = {
    "learning_rate":      "--lr",
    "n_estimators":       "--n-estimators",
    "max_depth":          "--max-depth",
    "min_child_weight":   "--min-child-weight",
    "subsample":          "--subsample",
    "colsample_bytree":   "--colsample-bytree",
    "reg_alpha":          "--reg-alpha",
    "reg_lambda":         "--reg-lambda",
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