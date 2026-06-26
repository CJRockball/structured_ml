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

TRAIN_FILE = None   # e.g. "s6e6_star_train_num_fe_20260611.parquet"
TEST_FILE  = None   # e.g. "s6e6_star_test_num_fe_20260611.parquet"

# ============================================================
# THEMES
# ============================================================

THEMES = {
    "base": {},

    "deep_slow": {
        "max_depth":        8,
        "min_child_weight": 5,
        "learning_rate":    0.01,
        "n_estimators":     4000,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
    },

    "shallow_fast": {
        "max_depth":        4,
        "min_child_weight": 1,
        "learning_rate":    0.1,
        "n_estimators":     800,
        "subsample":        0.9,
        "colsample_bytree": 0.9,
    },

    "regularised": {
        "max_depth":        6,
        "min_child_weight": 10,
        "learning_rate":    0.05,
        "n_estimators":     2000,
        "reg_alpha":        0.5,
        "reg_lambda":       5.0,
        "subsample":        0.75,
        "colsample_bytree": 0.75,
    },

    "stochastic": {
        "max_depth":        6,
        "min_child_weight": 3,
        "learning_rate":    0.05,
        "n_estimators":     2000,
        "subsample":        0.6,
        "colsample_bytree": 0.6,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
    },

    "balanced": {
        "max_depth":        6,
        "min_child_weight": 5,
        "learning_rate":    0.03,
        "n_estimators":     3000,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       2.0,
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

XGB_SCRIPT = Path(__file__).parent / "xgb1_copy.py"

for theme_name, params in THEMES.items():
    cmd = [sys.executable, str(XGB_SCRIPT),
           "--exp-name", f"tune_{theme_name}"]

    if TRAIN_FILE:
        cmd += ["--train-file", TRAIN_FILE]
    if TEST_FILE:
        cmd += ["--test-file", TEST_FILE]

    for param, flag in ARG_MAP.items():
        if param in params:
            cmd += [flag, str(params[param])]

    subprocess.run(cmd, check=True)