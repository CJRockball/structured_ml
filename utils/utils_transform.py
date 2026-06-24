"""
utils/utils_fe.py
-----------------
Generic, competition-agnostic feature transform utilities.
All functions:
  - Accept explicit column lists — no hardcoded names
  - Apply identically to train and test (caller passes both)
  - Return modified copies, never mutate in place
  - Cast outputs to float32 to stay consistent with basic.py dtype contract
  - Log what was added
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


# ============================================================
# LOG TRANSFORM
# ============================================================

def log1p_transform(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: list[str],
    clip_lower: float = 0.0,
    drop_original: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply log1p to right-skewed numeric columns.

    Use when: histogram shows heavy right tail (skew > 1.0 is a reasonable threshold).
    Clips to clip_lower before transform to handle negatives safely.

    Args:
        cols:          Columns to transform. Skips silently if col not in df.
        clip_lower:    Floor before log1p (default 0.0 — safe for counts/amounts).
        drop_original: If True, remove the source column after adding the log version.

    Output cols: {col}_log1p  (float32)
    """
    df1, df2 = df_train.copy(), df_test.copy()
    added = []

    for col in cols:
        if col not in df1.columns:
            log.warning(f"log1p_transform: '{col}' not in DataFrame — skipped")
            continue

        new_col = f"{col}_log1p"
        df1[new_col] = np.log1p(df1[col].clip(lower=clip_lower)).astype("float32")
        df2[new_col] = np.log1p(df2[col].clip(lower=clip_lower)).astype("float32")
        added.append(new_col)

        if drop_original:
            df1.drop(columns=[col], inplace=True)
            df2.drop(columns=[col], inplace=True)

    log.info(f"log1p_transform: added {added}")
    return df1, df2


# ============================================================
# RATIO FEATURES
# ============================================================

def ratio_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    pairs: list[tuple[str, str]],
    eps: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add ratio columns: numerator / (denominator + eps).

    Use when: domain knowledge suggests a rate or proportion matters more
    than the raw values (e.g. triglycerides/HDL, income/expenses).
    eps prevents division by zero without distorting large denominators.

    Args:
        pairs: List of (numerator_col, denominator_col) tuples.
        eps:   Small constant added to denominator (default 1e-6).

    Output cols: {numerator}_per_{denominator}  (float32)
    """
    df1, df2 = df_train.copy(), df_test.copy()
    added = []

    for num_col, den_col in pairs:
        missing = [c for c in (num_col, den_col) if c not in df1.columns]
        if missing:
            log.warning(f"ratio_features: {missing} not in DataFrame — pair skipped")
            continue

        new_col = f"{num_col}_per_{den_col}"
        df1[new_col] = (df1[num_col] / (df1[den_col] + eps)).astype("float32")
        df2[new_col] = (df2[num_col] / (df2[den_col] + eps)).astype("float32")
        added.append(new_col)

    log.info(f"ratio_features: added {added}")
    return df1, df2


# ============================================================
# CLIP / WINSORISE
# ============================================================

def clip_iqr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: list[str],
    lower_factor: float = 3.0,
    upper_factor: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Winsorise columns to [Q1 - k*IQR, Q3 + k*IQR] bounds.
    Bounds are computed on train only and applied to both train and test.

    Use when: boxplot shows extreme outliers that are likely noise, not signal.
    A factor of 3.0 is conservative (catches only severe outliers).
    Use 1.5 for aggressive clipping (standard Tukey fence).

    Args:
        cols:          Columns to clip. Skips silently if col not in df.
        lower_factor:  Multiplier below Q1 (default 3.0).
        upper_factor:  Multiplier above Q3 (default 3.0).

    Modifies columns in place within the copy (no new column names).
    """
    df1, df2 = df_train.copy(), df_test.copy()
    clipped = []

    for col in cols:
        if col not in df1.columns:
            log.warning(f"clip_iqr: '{col}' not in DataFrame — skipped")
            continue

        q1 = df1[col].quantile(0.25)
        q3 = df1[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - lower_factor * iqr
        hi = q3 + upper_factor * iqr

        n_clipped_train = ((df1[col] < lo) | (df1[col] > hi)).sum()
        n_clipped_test  = ((df2[col] < lo) | (df2[col] > hi)).sum()

        df1[col] = df1[col].clip(lower=lo, upper=hi).astype(df1[col].dtype)
        df2[col] = df2[col].clip(lower=lo, upper=hi).astype(df2[col].dtype)
        clipped.append(col)

        log.info(
            f"clip_iqr: '{col}'  bounds=[{lo:.3f}, {hi:.3f}]  "
            f"clipped train={n_clipped_train}  test={n_clipped_test}"
        )

    log.info(f"clip_iqr: processed {clipped}")
    return df1, df2


# ============================================================
# POLYNOMIAL TERMS
# ============================================================

def polynomial_terms(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: list[str],
    degrees: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add degree-n polynomial terms for numeric columns.

    Use when: target_vs_numerics plot shows clear non-linear relationship
    that a linear model (or embedding layer) cannot capture directly.
    Confirm with partial dependence plots before adding blindly.

    Args:
        cols:    Columns to expand. Skips silently if col not in df.
        degrees: List of integer powers to add (default [2]).
                 e.g. degrees=[2, 3] adds col_sq and col_cube.

    Output cols: {col}_sq (degree 2), {col}_cube (degree 3),
                 {col}_pow{n} for n >= 4  (float32)
    """
    if degrees is None:
        degrees = [2]

    DEGREE_NAMES = {2: "sq", 3: "cube"}

    df1, df2 = df_train.copy(), df_test.copy()
    added = []

    for col in cols:
        if col not in df1.columns:
            log.warning(f"polynomial_terms: '{col}' not in DataFrame — skipped")
            continue

        for deg in degrees:
            suffix = DEGREE_NAMES.get(deg, f"pow{deg}")
            new_col = f"{col}_{suffix}"
            df1[new_col] = (df1[col] ** deg).astype("float32")
            df2[new_col] = (df2[col] ** deg).astype("float32")
            added.append(new_col)

    log.info(f"polynomial_terms: added {added}")
    return df1, df2