# ============================================================
# fe_v2.py  —  Feature Engineering Step 2
# Reads:  PROC_DIR/{comp}_train_base_{date}.parquet
#         PROC_DIR/{comp}_test_base_{date}.parquet
# Writes: PROC_DIR/{comp}_train_fe_{date}.parquet      (cat labels)
#         PROC_DIR/{comp}_train_num_fe_{date}.parquet  (int16 codes)
#         PROC_DIR/{comp}_test_fe_{date}.parquet
#         PROC_DIR/{comp}_test_num_fe_{date}.parquet
# ============================================================
#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import datetime

from utils.utils_basic import cat_encode, save_dataset
from utils.utils_features import save_dataset, print_lineage
from utils.utils_reproducibility import set_seed
from utils.utils_transform import (
    log1p_transform, 
    ratio_features, 
    clip_iqr, 
    polynomial_terms, 
    subtraction_features, 
    angle_converter)
from utils.utils_fe_cross import num_num_interact, cat_cat_interact, num_cat_interact

from config import (
    FeatureConfig, features,
    CompetitionConfig, comp_cfg,
    PROC_DIR, ART_DIR,
    feature_logger as log,
    pipeline_logger as pipe_log,
)

pipe_log.info("fe_v3 feature engineering stage started")
set_seed()

DATE_SLUG = datetime.date.today().strftime("%Y%m%d")
sname = 'subj_know'

# Set up transforms
CLIP_COLS = []  # e.g. ["bmi", "triglycerides", "alcohol_consumption_per_week"]
LOG_COLS = ['redshift'] # e.g. ["bmi", "triglycerides", "alcohol_consumption_per_week"]
RATIO_PAIRS = []  # e.g. [("triglycerides", "hdl_cholesterol"), ("bmi", "age")]
SUB_PAIRS = [('u','g'), ('g','r'), ('r','i'), ('i','z'),('u','r'), ('g','i')] # [('feature1', 'feature2')]
POLY_COLS    = [('u_minus_g'), ('g_minus_r'), ('r_minus_i'), ('i_minus_z')]   # e.g. ["bmi", "age"]
POLY_DEGREES = [2]  # squares only by default; add 3 if justified
# set up interactions
NUM_NUM_PAIRS = [('g_minus_r', 'r_minus_i'), ('u_minus_g','g_minus_r'), ('g_minus_r','r'), ('u_minus_g','g')]  # e.g. [("bmi", "age"), ("triglycerides", "bmi")]
NUM_CAT_PAIRS = [ ]  # e.g. [("bmi", "ethnicity"), ("age", "smoking_status")]
NUM_CAT_STATS = [] # e.g. ["mean", "std"]
CAT_CAT_PAIRS = [] # e.g. [("compound", "team_color")]
CAT_CAT_MAX_CARD = 200 # pairs above this threshold → high_card_cols (TE in CV)
# save
STAGES = ["load_base", f"fe_{sname}"]

# ============================================================
# 1. LOAD base parquets from basic.py
# ============================================================

def load_base(comp_cfg: CompetitionConfig, PROC_DIR: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    def latest(prefix: str) -> Path:
        files = sorted(PROC_DIR.glob(f"{prefix}*.parquet"))
        assert files, f"No parquet found for prefix '{prefix}' — run basic.py first"
        return files[-1]

    train_path = latest(f"{comp_cfg.name}_train_base_20260611")
    test_path  = latest(f"{comp_cfg.name}_test_base_20260611")
    df_train   = pd.read_parquet(train_path)
    df_test    = pd.read_parquet(test_path)

    log.info(f"Loaded train={df_train.shape}  test={df_test.shape}")
    log.info(f"  from {train_path.name} / {test_path.name}")
    return df_train, df_test

df_train, df_test = load_base(comp_cfg, PROC_DIR)


# ============================================================
# 2. FEATURE ENGINEERING BLOCKS
# Edit this section. Each block is independent — comment out to disable.
# Column names come from EDA output and domain knowledge.
# The DataFrame is the source of truth; Schema auto-discovers downstream.
# ============================================================

# ---- BLOCK A: Clip outliers before any interactions --------
# Run first — interactions on unclipped outliers amplify noise.
# Populate CLIP_COLS from eda numeric_distributions.png boxplots.


if CLIP_COLS:
    df_train, df_test = clip_iqr(
        df_train, df_test,
        cols=CLIP_COLS,
        lower_factor=3.0,
        upper_factor=3.0,
    )


# ---- BLOCK B: Log-transform right-skewed numerics ----------
# Populate from EDA: columns whose histogram shows heavy right tail.


if LOG_COLS:
    df_train, df_test = log1p_transform(
        df_train, df_test,
        cols=LOG_COLS,
        drop_original=False,  # keep original alongside log version
    )


# ---- BLOCK C: Ratio features --------------------------------
# Populate from domain knowledge (rates, proportions).
# Each tuple: (numerator_col, denominator_col)


if RATIO_PAIRS:
    df_train, df_test = ratio_features(
        df_train, df_test,
        pairs=RATIO_PAIRS,
        eps=1e-6,
    )


# ---- BLOCK C2: Ratio features --------------------------------
# Populate from domain knowledge (rates, proportions).
# Each tuple: (numerator_col, denominator_col)


if SUB_PAIRS:
    df_train, df_test = subtraction_features(
        df_train, df_test,
        pairs=SUB_PAIRS,
    )
    
    
# ---- BLOCK D: Polynomial terms ------------------------------
# Only add after confirming non-linearity in target_vs_numerics plots.


if POLY_COLS:
    df_train, df_test = polynomial_terms(
        df_train, df_test,
        cols=POLY_COLS,
        degrees=POLY_DEGREES,
    )

# ---- BLOCK E: Num × Num interactions ------------------------
# Use explicit pairs driven by correlation matrix + domain knowledge.
# auto_top_n is for rapid exploration only — prune before final run.


if NUM_NUM_PAIRS:
    df_train, df_test = num_num_interact(
        df_train, df_test,
        pairs=NUM_NUM_PAIRS,
        corr_threshold=0.85,  # skip near-collinear pairs
    )


# ---- BLOCK F: Num × Cat group aggregations ------------------
# Target-free: safe to compute on full train.
# Each tuple: (numeric_col, cat_col)


if NUM_CAT_PAIRS:
    df_train, df_test = num_cat_interact(
        df_train, df_test,
        pairs=NUM_CAT_PAIRS,
        stats=NUM_CAT_STATS,
        min_samples_leaf=10,
    )


# ---- BLOCK G: Cat × Cat compound features -------------------
# Low-cardinality pairs are ordinal-encoded here via utils_basic.cat_encode.
# High-cardinality pairs are returned as raw str for fold-aware TE in the CV loop.


if CAT_CAT_PAIRS:
    df_train, df_test, low_card_cols, high_card_cols = cat_cat_interact(
        df_train, df_test,
        pairs=CAT_CAT_PAIRS,
        max_combined_cardinality=CAT_CAT_MAX_CARD,
    )

    if low_card_cols:
        # Ordinal-encode low-cardinality combined cols using the same
        # cat_encode from utils_basic — consistent __UNKNOWN__ at index 0
        df_train, df_test = cat_encode(
            df_train, df_test,
            target=features.target,
            cols_to_encode=low_card_cols,
        )
        log.info(f"Cat×Cat low-card ordinal encoded: {low_card_cols}")

    if high_card_cols:
        # Leave as raw str — CV loop in model script applies fold-aware TE
        log.info(f"Cat×Cat high-card flagged for TE in CV loop: {high_card_cols}")
else:
    low_card_cols  = []
    high_card_cols = []


# ============================================================
# 2b. SPECIAL FUNCTIONS — Convert space angles
# ============================================================

df_train, df_test = angle_converter(
    df_train,
    df_test,
    ['alpha','delta'])

# ============================================================
# 3. AUDIT — confirm no nulls introduced, log column delta
# ============================================================

def audit_fe(df_post: pd.DataFrame, df_pre: pd.DataFrame, label: str) -> None:
    added   = sorted(set(df_post.columns) - set(df_pre.columns))
    nulls   = df_post.isna().sum()
    new_nulls = nulls[nulls > 0]

    log.info(f"[{label}] columns: {df_pre.shape[1]} → {df_post.shape[1]}  (+{len(added)} engineered)")
    if added:
        log.info(f"[{label}] added: {added}")
    if not new_nulls.empty:
        log.warning(f"[{label}] nulls introduced:\n{new_nulls}")
    else:
        log.info(f"[{label}] no nulls introduced")

audit_fe(df_train, pd.read_parquet(sorted(PROC_DIR.glob(
    f"{comp_cfg.name}_train_base_*.parquet"))[-1]), "train")
audit_fe(df_test, pd.read_parquet(sorted(PROC_DIR.glob(
    f"{comp_cfg.name}_test_base_*.parquet"))[-1]), "test")

# ============================================================
# 4. SAVE — mirrors basic.py output pattern
# ============================================================

if high_card_cols:
    STAGES.append("high_card_te_pending")  # signals to model script that TE is needed

save_dataset(
    df=df_train,
    path=PROC_DIR / f"{comp_cfg.name}_train_{sname}_{DATE_SLUG}.parquet",
    slug=f"{comp_cfg.name}_train_{sname}",
    target=[features.target],
    source_file=__file__,
    parent=f"{comp_cfg.name}_train_base_{DATE_SLUG}.parquet",
    stages=STAGES,
    notes=f"FE applied. high_card_te_cols={high_card_cols}",
)


save_dataset(
    df=df_test,
    path=PROC_DIR / f"{comp_cfg.name}_test_{sname}_{DATE_SLUG}.parquet",
    slug=f"{comp_cfg.name}_test_{sname}",
    target=[],
    source_file=__file__,
    parent=f"{comp_cfg.name}_test_base_{DATE_SLUG}.parquet",
    stages=STAGES,
    notes=f"FE applied. high_card_te_cols={high_card_cols}",
)


print_lineage(PROC_DIR / f"{comp_cfg.name}_train_{sname}_{DATE_SLUG}.parquet")

pipe_log.info(f"fe_{sname} stage complete — "
              f"train={df_train.shape}  test={df_test.shape}  "
              f"high_card_te={high_card_cols}")



