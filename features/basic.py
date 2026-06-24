#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import logging
import datetime

from utils.utils_basic import (
    save_dataset, 
    print_lineage, 
    target_encode, 
    missing_encode, 
    cat_encode)
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


# %%

if __name__ == "__main__":
    
    df_train, df_test = load_data(features, RAW_DIR)   
    df_train = target_encode(df_train, features, encode_target=True)
    df_train, df_test = missing_encode(df_train, df_test, features)
    df_train, df_test = cat_encode(df_train, df_test, features.target)
    df_train_num, df_test_num = save_base(df_train, df_test, features.target)
 
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
    
 

    print_lineage(PROC_DIR / f"{comp_cfg.name}_train_base_{date_slug}.parquet")
    print_lineage(PROC_DIR / f"{comp_cfg.name}_train_num_base_{date_slug}.parquet")




