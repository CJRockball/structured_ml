"""
utils/utils_fe_cross.py
-----------------------
Cross-feature utilities: num×num, cat×cat, num×cat interactions.
All functions follow the same contract as utils_fe.py:
  - Caller passes both train and test explicitly — no leakage
  - Returns modified copies, never mutates in place
  - All numeric outputs cast to float32
  - Categorical outputs remain str (object) so cat_encode in basic.py
    or a downstream ordinal encoder handles them correctly
  - Logs every column added
"""

import numpy as np
import pandas as pd
import logging
from itertools import combinations

log = logging.getLogger(__name__)


# ============================================================
# NUM × NUM  — multiply two numeric columns
# ============================================================

def num_num_interact(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
    auto_top_n: int | None = None,
    corr_threshold: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multiply pairs of numeric columns to create interaction terms.

    Two modes:
      - Explicit:  pass pairs=[("bmi", "age"), ("bmi", "triglycerides")]
      - Auto:      pass auto_top_n=N to cross all combinations of the
                   top-N numeric columns by variance. Use with caution —
                   N=10 produces 45 new columns. Filter by corr_threshold
                   to suppress pairs that are already highly correlated
                   (collinear pairs add noise, not signal).

    corr_threshold: skip a pair if abs(Pearson r) between the two columns
                    exceeds this value (default 0.0 = no filtering).
                    Recommended: 0.85 to suppress near-duplicate interactions.

    Output cols: {col_a}_x_{col_b}  (float32)
    """
    df1, df2 = df_train.copy(), df_test.copy()
    added = []

    if pairs is None and auto_top_n is not None:
        num_cols = df1.select_dtypes("number").columns.tolist()
        # Rank by variance on train, take top N
        variances = df1[num_cols].var().sort_values(ascending=False)
        top_cols = variances.head(auto_top_n).index.tolist()
        pairs = list(combinations(top_cols, 2))
        log.info(f"num_num_interact auto mode: top {auto_top_n} cols → {len(pairs)} pairs")

    if not pairs:
        log.warning("num_num_interact: no pairs provided and auto_top_n not set — nothing done")
        return df1, df2

    for col_a, col_b in pairs:
        missing = [c for c in (col_a, col_b) if c not in df1.columns]
        if missing:
            log.warning(f"num_num_interact: {missing} not in DataFrame — pair skipped")
            continue

        if corr_threshold > 0.0:
            r = df1[[col_a, col_b]].corr().iloc[0, 1]
            if abs(r) > corr_threshold:
                log.info(
                    f"num_num_interact: '{col_a}' × '{col_b}' skipped "
                    f"(r={r:.3f} > threshold={corr_threshold})"
                )
                continue

        new_col = f"{col_a}_x_{col_b}"
        df1[new_col] = (df1[col_a] * df1[col_b]).astype("float32")
        df2[new_col] = (df2[col_a] * df2[col_b]).astype("float32")
        added.append(new_col)

    log.info(f"num_num_interact: added {len(added)} columns: {added}")
    return df1, df2


# ============================================================
# CAT × CAT  — concatenate two categorical columns into a compound label
# ============================================================

def cat_cat_interact(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
    auto_all: bool = False,
    max_combined_cardinality: int = 200,
    low_card_threshold: int | None = None,
    sep: str = "__",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Create compound categorical features by concatenating two categorical columns.

    Returns:
        df_train_out, df_test_out, low_card_cols, high_card_cols

    Logic:
    - Candidate pair generation comes from explicit `pairs` or `auto_all=True`
    - Pair creation is skipped if nunique(A) * nunique(B) exceeds
      `max_combined_cardinality`
    - Created columns are split into low/high-card buckets using the observed
      train-set cardinality of the new compound column
    - If `low_card_threshold` is None, it defaults to `max_combined_cardinality`

    Notes:
    - No target is used
    - Copies are returned; inputs are not mutated
    - Output categorical columns remain object dtype for downstream encoders
    """
    df1, df2 = df_train.copy(), df_test.copy()
    low_card_cols: list[str] = []
    high_card_cols: list[str] = []
    added: list[str] = []

    if low_card_threshold is None:
        low_card_threshold = max_combined_cardinality

    if pairs is None:
        if auto_all:
            cat_cols = df1.select_dtypes(include=["object", "category"]).columns.tolist()
            pairs = list(combinations(cat_cols, 2))
            log.info(
                f"cat_cat_interact auto mode: {len(cat_cols)} cat cols -> "
                f"{len(pairs)} candidate pairs"
            )
        else:
            pairs = []

    if not pairs:
        log.warning("cat_cat_interact: no pairs provided and auto_all not set — nothing done")
        return df1, df2, low_card_cols, high_card_cols

    for col_a, col_b in pairs:
        missing = [c for c in (col_a, col_b) if c not in df1.columns]
        if missing:
            log.warning(f"cat_cat_interact: {missing} not in DataFrame — pair skipped")
            continue

        est_combined_card = df1[col_a].nunique(dropna=False) * df1[col_b].nunique(dropna=False)
        if est_combined_card > max_combined_cardinality:
            log.info(
                f"cat_cat_interact: '{col_a}' x '{col_b}' skipped "
                f"(estimated cardinality {est_combined_card} > {max_combined_cardinality})"
            )
            continue

        new_col = f"{col_a}{sep}{col_b}"
        df1[new_col] = df1[col_a].astype(str) + sep + df1[col_b].astype(str)
        df2[new_col] = df2[col_a].astype(str) + sep + df2[col_b].astype(str)
        added.append(new_col)

        observed_card = df1[new_col].nunique(dropna=False)
        if observed_card <= low_card_threshold:
            low_card_cols.append(new_col)
        else:
            high_card_cols.append(new_col)

    log.info(f"cat_cat_interact: added {len(added)} columns: {added}")
    log.info(
        f"cat_cat_interact: low_card_cols={low_card_cols}, "
        f"high_card_cols={high_card_cols}"
    )

    return df1, df2, low_card_cols, high_card_cols

# ============================================================
# NUM × CAT  — per-category mean / std of a numeric column
# ============================================================

def num_cat_interact(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    pairs: list[tuple[str, str]],
    stats: list[str] | None = None,
    min_samples_leaf: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group a numeric column by a categorical column and add per-group
    aggregation statistics as new numeric features.

    This is a form of target-free aggregation encoding — it captures
    'what is the typical BMI for this ethnicity group?' without leaking
    the target. It is safe to compute on the full train set because no
    target is involved.

    Stats available: "mean", "std", "median", "min", "max"
    Default: ["mean", "std"]

    Groups with fewer than min_samples_leaf samples fall back to the
    global statistic (train-computed). This prevents sparse group
    estimates from destabilising the model on rare categories.

    Bounds are fitted on train only and applied to test — no leakage.

    Args:
        pairs:             List of (numeric_col, cat_col) tuples.
        stats:             Which aggregations to add (default ["mean", "std"]).
        min_samples_leaf:  Minimum group size before falling back to global stat.

    Output cols: {num_col}_by_{cat_col}_{stat}  (float32)
    """
    if stats is None:
        stats = ["mean", "std"]

    valid_stats = {"mean", "std", "median", "min", "max"}
    bad = set(stats) - valid_stats
    if bad:
        raise ValueError(f"num_cat_interact: unsupported stats {bad}. Choose from {valid_stats}")

    df1, df2 = df_train.copy(), df_test.copy()
    added = []

    for num_col, cat_col in pairs:
        missing = [c for c in (num_col, cat_col) if c not in df1.columns]
        if missing:
            log.warning(f"num_cat_interact: {missing} not in DataFrame — pair skipped")
            continue

        # Compute group maps on train only
        group_counts = df1.groupby(cat_col)[num_col].count()

        for stat in stats:
            agg_fn = getattr(df1.groupby(cat_col)[num_col], stat)
            group_map = agg_fn()

            # Global fallback for small / unseen groups
            global_val = getattr(df1[num_col], stat)()

            # Apply min_samples_leaf guard: replace small-group estimates with global
            small_groups = group_counts[group_counts < min_samples_leaf].index
            group_map[small_groups] = global_val

            new_col = f"{num_col}_by_{cat_col}_{stat}"

            df1[new_col] = (
                df1[cat_col].astype(str).map(group_map).fillna(global_val).astype("float32")
            )
            df2[new_col] = (
                df2[cat_col].astype(str).map(group_map).fillna(global_val).astype("float32")
            )
            added.append(new_col)

    log.info(f"num_cat_interact: added {len(added)} columns: {added}")
    return df1, df2