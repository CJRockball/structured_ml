#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import logging
import datetime

from utils.utils_features import save_dataset, print_lineage

from utils.utils_reproducibility import set_seed

from config import (     
    FeatureConfig, features,
    CompetitionConfig, comp_cfg,
    RAW_DIR, PROC_DIR, MODEL_DIR, SUB_DIR, ART_DIR,
    feature_logger as log,
    pipeline_logger as pipe_log,
)

pipe_log.info("data basic processing stage started")
set_seed()


#%%


def load_data(features:FeatureConfig, RAW_DIR:Path)-> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data, set types and structure the output dataframes
    """
    try:
        df_train = pd.read_csv(RAW_DIR / 'train.csv').drop(columns=['id'])
        df_test = pd.read_csv(RAW_DIR / 'test.csv').drop(columns=['id'])

    except Exception as e:
        log.error(f'Failed to load data: {e}')
        raise
     
    # Set column order
    df_train = df_train[features.nums + features.cats + [features.target]]
    df_test = df_test[features.nums + features.cats]
    
    assert df_train.shape[1] == (df_test.shape[1] + 1), 'Train and test frames are different size'
    assert df_train.columns.to_list() == df_test.columns.to_list() + [features.target], 'Train and test have different columns'

    log.info('Load train and test successfull')
    
    return df_train, df_test

df_train, df_test = load_data(features, RAW_DIR)

#%%

def target_encode(
    df_train: pd.DataFrame,
    features: FeatureConfig,
    encode_target: bool = False
) -> pd.DataFrame :
    """
    Clean and encode the target column on the training set only.
    Test set is never passed here — target is absent by definition.

    Args:
        df_train:       Training DataFrame with target column present
        target:         List containing target column name
        target_type:    dtype to cast target to after cleaning
        encode_target:  If True, ordinal encode a string target

    Returns:
        Cleaned DataFrame with target encoded
    """
    col = features.target
    df = df_train.copy()
    assert col in df.columns, f'Target column {col} not found in DataFrame'
    
    n_missing = df[col].isna().sum()
    
    if n_missing  > 0:
        log.info(f"Target column {col} has {n_missing} missing values")

        df = df.dropna(subset=[col])
        log.warning(f'Dropped {n_missing} rows with missing target.'
                       f'Remaining rows: {len(df)}')
    else: 
        log.info(f'Target column {col} has no missing values')
        
    assert df[col].isna().sum() == 0, 'Target still has missing values after encoding'


    # Categorical encoding
    if encode_target:
        freq_order = (
            df[col].astype(str)
            .value_counts()
            .index.tolist()
        )
        df[col] = pd.Categorical(
            df[col].astype(str),
            categories=freq_order,
            ordered=False
        )
        log.info(f'Target {col} encoded as category'
                    f'Classes: {list(df[col].cat.categories)}')

    else:
        df[col] = df[col].astype(features.target_type)
        log.info(f'Target {col} is cast to {features.target_type}')

    return df

df_train = target_encode(df_train, features, encode_target=True)

#%%

def missing_encode(
    df_train:pd.DataFrame,
    df_test:pd.DataFrame,
    features:FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fill missing values, then cast to intended dtypes.
    Uses explicit cats/nums lists because some categoricals are
    integer-typed in the raw CSV and cannot be reliably auto-detected.
    Order is critical: fill NaN before astype(str) to avoid 'nan' strings.
    Cats:  NaN → 'missing', then cast to str (object)
    Nums:  NaN → column mean + has_missing flag, then cast to float32
    """
    df1 = df_train.copy()
    df_y = df_train[[features.target]]
    df1 = df1.drop(columns=[features.target])
    df2 = df_test.copy()
 
    # --- Categoricals: fill NaN first, then cast to str ---
    for cat in features.cats:
        df1[cat] = df1[cat].fillna('missing').astype(str)
        df2[cat] = df2[cat].fillna('missing').astype(str)

    # --- Numerics: flag + fill + cast ---
    for num in features.nums:
        train_na = df1[num].isna().sum()
        test_na  = df2[num].isna().sum()

        if train_na > 0 or test_na > 0:
            df1[f'{num}_has_missing'] = df1[num].isna().astype('int8').astype('str')
            df2[f'{num}_has_missing'] = df2[num].isna().astype('int8').astype('str')
            
            mean_val = df1[num].mean()
            df1[num] = df1[num].fillna(mean_val)
            df2[num] = df2[num].fillna(mean_val)

        df1[num] = df1[num].astype('float32')
        df2[num] = df2[num].astype('float32')
 

    # Add back the label
    df1[features.target] = df_y[features.target]

    assert df1.isnull().sum().sum() == 0, 'Training file still has missing values'
    assert df2.isnull().sum().sum() == 0, 'Test file still has missing values'
    assert df1.shape[1] == (df2.shape[1] + 1), 'Train and test frames are different size'
    assert df1.columns.to_list() == df2.columns.to_list() + [features.target], 'Train and test have different columns'

    log.info('missing data encoded')
    return df1, df2 

df_train, df_test = missing_encode(df_train, df_test, features)

# %%

def cat_encode(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ordinal-encode categorical columns using train categories.
    Unseen test categories are mapped to code 0 (an explicit 'unknown' category).
    Returned columns are pandas Categorical with codes starting at 0.
    """
    df1 = df_train.copy()
    df_y = df_train[[target]]
    df1 = df1.drop(columns=[target])
    df2 = df_test.copy()   
    cats = df1.select_dtypes('str').columns.tolist()
    
    for col in cats:
        # 1) Get train categories in a fixed order
        train_cats = df1[col].astype("category").cat.categories

        # 2) Build the final category labels, with unknown at index 0
        final_categories = pd.Index(
            ["__UNKNOWN__"] + train_cats.tolist()
        )

        # 3) Train codes: base codes (0..n-1) then +1
        base_train_codes = pd.Categorical(df1[col], categories=train_cats).codes
        train_codes = base_train_codes + 1  # now 1..n; 0 is reserved for unknown

        # 4) Test codes: unseen -> -1, then +1 => 0
        base_test_codes = pd.Categorical(df2[col], categories=train_cats).codes
        test_codes = base_test_codes + 1
        test_codes = np.where(base_test_codes == -1, 0, test_codes)

        # 5) Convert back to Categorical with explicit categories (0..n)
        df1[col] = pd.Categorical.from_codes(
            codes=train_codes,
            categories=final_categories,
            ordered=False,
        )
        
        df2[col] = pd.Categorical.from_codes(
            codes=test_codes,
            categories=final_categories,
            ordered=False,
        )

    # Add back the label
    df1[target] = df_y[target]

    assert df1.shape[1] == (df2.shape[1] + 1), 'Train and test frames are different size'
    assert df1.columns.to_list() == df2.columns.to_list() + [target], 'Train and test have different columns'
    
    log.info('Categorical features encoded')
    return df1, df2

df_train, df_test = cat_encode(df_train, df_test, features.target)

#%%

def save_base(
        df_train:pd.DataFrame, 
        df_test:pd.DataFrame, 
        target:str):
    """
    This function makes a copy of df_train and df_test with categorical features
    with numeric labels
    """
    cats = [cat for cat in df_train.select_dtypes('category').columns if cat != target]
    
    df_train_num = df_train.copy()
    df_test_num = df_test.copy()
    
    df_train_num[cats] = df_train_num[cats].apply(lambda s: s.cat.codes.astype("int16"))
    df_train_num[target] = df_train_num[[target]].apply(lambda s: s.cat.codes.astype("int16"))
    df_test_num[cats] = df_test_num[cats].apply(lambda s: s.cat.codes.astype("int16"))

    log.info(f'Converted categorical from name labels to numeric labels')
    return df_train_num, df_test_num


df_train_num, df_test_num = save_base(df_train, df_test, features.target)
        
#%%

date_slug = datetime.date.today().strftime("%Y%m%d")

#
save_dataset(
    df=df_train,
    path=PROC_DIR / f"{comp_cfg.name}_train_base_{date_slug}.parquet",
    slug=f"{comp_cfg.name}_train_base",
    target=[features.target],
    source_file=__file__,       # always correct, works on laptop and in git
    parent="train.csv",
    stages=["load", "missing_encode", "cat_encode"],
    notes="Baseline encoding, no feature engineering",
)


# Save train numeric variant (derived from the base parquet)
save_dataset(
    df=df_train_num,
    path=PROC_DIR / f"{comp_cfg.name}_train_num_base_{date_slug}.parquet",
    slug=f"{comp_cfg.name}_train_num_base",
    target=[features.target],
    source_file=__file__,       # always correct, works on laptop and in git
    parent=f"{comp_cfg.name}_train_base_{date_slug}.parquet",   # points to its parent
    stages=["cat_to_codes"],
    notes="Cat columns converted to int16 codes for GBDT numeric input",
)

# Save test base (derived from raw)
save_dataset(
    df=df_test,
    path=PROC_DIR / f"{comp_cfg.name}_test_base_{date_slug}.parquet",
    slug=f"{comp_cfg.name}_test_base",
    target=[features.target],
    source_file=__file__,       # always correct, works on laptop and in git
    parent="test.csv",           # raw file it came from
    stages=["load", "missing_encode", "ord_encode"],
    notes="Baseline encoding, no feature engineering",
)

# Save numeric variant (derived from the base parquet)
save_dataset(
    df=df_test_num,
    path=PROC_DIR / f"{comp_cfg.name}_test_num_base_{date_slug}.parquet",
    slug=f"{comp_cfg.name}_test_num_base",
    target=[features.target],
    source_file=__file__,       # always correct, works on laptop and in git
    parent=f"{comp_cfg.name}_test_base_{date_slug}.parquet",   # points to its parent
    stages=["cat_to_codes"],
    notes="Cat columns converted to int16 codes for GBDT numeric input",
)

# %%

print_lineage(PROC_DIR / f"{comp_cfg.name}_train_base_{date_slug}.parquet")
print_lineage(PROC_DIR / f"{comp_cfg.name}_train_num_base_{date_slug}.parquet")

# %%



