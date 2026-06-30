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
model_name = "lgb1"
# ============================================================
# THEMES
# ============================================================

THEMES = {
#     "base fe": {},
    # LGB_v1 — High num_leaves, Optuna-inspired (from s6e6 XGB+LGBM+CAT, best LGBM config)
    "lgb_params_v1": {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'device_type': 'gpu',
        'learning_rate': 0.025,
        'num_leaves': 191,
        'feature_fraction': 0.82,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'min_data_in_leaf': 50,
        'lambda_l1': 0.1,
        'lambda_l2': 0.5,
        'class_weight': 'balanced',
        'n_estimators': 5000,
        'early_stopping_rounds': 150,
        'seed': 42,
        'verbose': -1,
    },

    # LGB_v2 — Conservative num_leaves, high min_data_in_leaf, low LR (from Stacked Baseline)
    "lgb_params_v2": {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'device_type': 'gpu',
        'learning_rate': 0.04,
        'num_leaves': 127,
        'feature_fraction': 0.70,
        'bagging_fraction': 0.80,
        'bagging_freq': 1,
        'min_data_in_leaf': 300,
        'lambda_l1': 0.2,
        'lambda_l2': 1.0,
        'class_weight': 'balanced',
        'n_estimators': 4000,
        'early_stopping_rounds': 150,
        'seed': 0,
        'verbose': -1,
    },

    # LGB_v3 — Long training, very low LR, large tree (matches Stellar Class Prediction S6E6)
    "lgb_params_v3": {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'device_type': 'gpu',
        'learning_rate': 0.008,
        'num_leaves': 150,
        'feature_fraction': 0.80,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'min_data_in_leaf': 100,
        'min_split_gain': 1.1,
        'lambda_l1': 0.13,
        'lambda_l2': 0.14,
        'class_weight': 'balanced',
        'n_estimators': 7000,
        'early_stopping_rounds': 300,
        'seed': 1,
        'verbose': -1,
    },

    # LGB_v4 — DART boosting for regularization diversity
    "lgb_params_v4": {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'dart',
        'device_type': 'gpu',
        'learning_rate': 0.03,
        'num_leaves': 160,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.90,
        'bagging_freq': 1,
        'drop_rate': 0.10,
        'skip_drop': 0.50,
        'min_data_in_leaf': 80,
        'lambda_l1': 0.5,
        'lambda_l2': 2.0,
        'class_weight': 'balanced',
        'n_estimators': 3000,   # no early stopping with DART
        'seed': 99,
        'verbose': -1,
    },

    # LGB_v5 — Shallow max_depth cap, high feature dropout, strong L2 (diversity variant)
    "lgb_params_v5": {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'device_type': 'gpu',
        'learning_rate': 0.018,
        'num_leaves': 255,
        'max_depth': 8,
        'feature_fraction': 0.60,
        'bagging_fraction': 0.80,
        'bagging_freq': 1,
        'min_data_in_leaf': 150,
        'lambda_l1': 1.0,
        'lambda_l2': 3.5,
        'class_weight': 'balanced',
        'n_estimators': 6000,
        'early_stopping_rounds': 200,
        'seed': 7,
        'verbose': -1,
    },
}



# ============================================================
# ARG MAP  —  param name → xgb1.py CLI flag
# ============================================================

ARG_MAP = {
    "learning_rate":      "--lr",
    "n_estimators":       "--n-estimators",
    "num_leaves":         "--num-leaves",
    "max_depth":          "--max-depth",
    "min_child_samples":   "--min-child-samples",
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