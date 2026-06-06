#%%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import logging
import shap 

from category_encoders import TargetEncoder

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import auc, log_loss, roc_curve, \
    roc_auc_score, root_mean_squared_error, balanced_accuracy_score

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import datetime
import json

from utils.utils_fit_models import get_device
from utils.utils_reproducibility import set_seed

# # Create log directory
# DIR = Path(__file__).parent.parent
# LOGDIR = DIR / 'logs'
# LOGDIR.mkdir(exist_ok=True)
# LOGFILE = LOGDIR / 'ml.log'

# # Set up logging, for file and console
# logging.basicConfig(
#     level=logging.INFO, 
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers = [
#         logging.FileHandler(LOGFILE),
#         logging.StreamHandler()
#     ]
# )

# logger = logging.getLogger(__name__)
# logger.info(f"📝 Logging to {LOGFILE}")

# RANDOM_STATE = 1337
# date_slug = datetime.date.today().strftime("%Y%m%d")
# BNAME  = 'star_base'
# TRAIN  = f'{BNAME}_train_20260601'
# TEST   = f'{BNAME}_test_20260601'
# TARGET = ['class']
# KFOLD  = 5
# FNAME  = 'xgb_base_test'
# SUB_ROWS = 247435


# XGB_PARAMS = dict(tree_method='hist',
#                     n_estimators=2000, 
#                     objective='multi:softprob', 
#                     eval_metric='mlogloss', 
#                     num_class=3,
#                     enable_categorical=True, 
#                     n_jobs=4,
#                     random_state=RANDOM_STATE,    
                    
#                     #learning_rate=0.1,
#                     max_bin=1024,
#                     # min_child_weight=3,
#                     #subsample=0.8,
#                     #colsample_bytree=0.5,
#                     #colsample_bylevel=0.5,
#                     # gamma=0.1,
#                     reg_alpha=2,
#                     reg_lambda=0.3,

#                     max_depth = 6,
#                     device=get_device())

from config import (
    global_cfg,          # ← already available, no extra import needed
    features,
    data_cfg, xgb_cfg, cv_cfg,
    PROC_DIR, MODEL_DIR, SUB_DIR,
    model_logger as log,
    pipeline_logger as pipe_log,
)

pipe_log.info("[models.xgb] stage started")

set_seed()

#%%

def import_data(train:str, test:str, target:list[str]) \
        -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Function to import train and test data.
    """
    # Load data
    df_train = pd.read_parquet(data_cfg.train_path(PROC_DIR))
    df_test  = pd.read_parquet(data_cfg.test_path(PROC_DIR))
    
    # Get feature types
    cats = [c for c in df_train.select_dtypes('category').columns if c != target[0]]
    nums = [c for c in df_train.select_dtypes('float').columns if c != target[0]]
    
    log.info(f"Success. Imported {train} and {test}")
    return df_train, df_test, cats, nums

df_train, df_test, cats, nums = import_data(TRAIN, TEST, TARGET)

# %%

def train_model(df_train, df_test, KFOLD, cats, XGB_PARAMS, rounds=100, target_enc=True):
    df_y = df_train[TARGET[0]].copy().cat.codes
    df_X = df_train.drop(columns=TARGET).copy()
    df_X_test = df_test.copy()
    
    for col in cats:
        df_X[col] = df_X[col].cat.codes.astype('int16')
        df_X_test[col] = df_X_test[col].cat.codes.astype('int16')
    
    skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE )

    n_classes = df_y.nunique()
    oof = np.zeros((len(df_train),n_classes))
    preds = np.zeros((len(df_test),n_classes))
    fold_metrics = []
    fold_loglosses = []
    for i,(train_index, valid_index) in enumerate(skf.split(df_X, df_y)):
        Xtrain = df_X.iloc[train_index]
        ytrain = df_y.iloc[train_index]
        Xvalid = df_X.iloc[valid_index]
        yvalid = df_y.iloc[valid_index]
        Xtest = df_X_test.copy()
        
        if target_enc:
            enc = TargetEncoder(cols=cats, min_samples_leaf=20, smoothing=10)
            enc.fit(Xtrain, ytrain)
        
            Xtrain = enc.transform(Xtrain)
            Xvalid = enc.transform(Xvalid)
            Xtest = enc.transform(Xtest)

        # XGB
        # Early stopping call back, use to get best model back
        es = xgb.callback.EarlyStopping(
        rounds=rounds,
        min_delta=1e-3,
        save_best=True,
        maximize=False,
        data_name="validation_0",
        metric_name="mlogloss",)

        model = XGBClassifier(**XGB_PARAMS, early_stopping_rounds=rounds, callbacks=[es])
                
        model = model.fit(Xtrain, ytrain, 
                        eval_set=[(Xvalid, yvalid)],
                        verbose=rounds)   
        
        ypred_proba = model.predict_proba(Xvalid)
        y_pred = model.predict(Xvalid)
        
        
        fold_logloss = log_loss(yvalid, ypred_proba)
        fold_metric = balanced_accuracy_score(yvalid, y_pred)
        oof[valid_index] = ypred_proba  # Save as multi col with percentages

        # Save
        fold_loglosses.append(fold_logloss)
        fold_metrics.append(fold_metric)
        logger.info(f'Fold {i+1}, Log loss: {fold_logloss:.5f}, metric: {fold_metric:.5f}')

        preds += model.predict_proba(Xtest) / KFOLD
        
    logger.info(f"Overall Score, logloss: {np.mean(fold_loglosses):.5f}, metric: {np.mean(fold_metrics):.5f}")
    return model, Xtrain, Xvalid, oof, preds, fold_metrics, fold_loglosses


model, Xtrain, Xvalid, oof, preds, fold_metrics, fold_loglosses = train_model(df_train, df_test, KFOLD, cats, XGB_PARAMS, rounds=100)

# %%
def model_importance(model, Xtrain):
    # Get feature importance scores
    importance_scores = model.get_booster().get_score(importance_type='total_gain')
    df_imp = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['Importance'])
    df_imp.index = Xtrain.columns

    df_imp.plot(kind='barh')
    return

model_importance(model, Xtrain)

# %% Shap With target encoding i.e. all numerical data

def model_shap(model, Xtrain, Xvalid, random_state):
    X_shap_train = Xtrain.sample(n=5000, random_state=random_state)
    X_shap_valid = Xvalid.sample(n=10000, random_state=random_state)
    # SHAP analysis for XGBoost (global + local + interactions)

    # 0) Sample for speed + stability in plots
    # For 600k rows, SHAP plots are clearer with ~2k–20k points
    X_bg = X_shap_train.sample(n=len(X_shap_train), random_state=random_state)
    X_explain = X_shap_valid.sample(n=len(X_shap_valid), random_state=random_state)

    # 1) Preferred modern API (creates shap.Explanation; works well for tree models)
    explainer = shap.Explainer(model, X_bg)  # background = reference distribution
    sv = explainer(X_explain)

    # 2) Global importance (bar) + distribution (beeswarm)
    shap.plots.bar(sv, max_display=30)        # mean(|SHAP|) global ranking
    shap.plots.beeswarm(sv, max_display=30)   # shows direction + nonlinearity
    return

model_shap(model, Xtrain, Xvalid, RANDOM_STATE)

#%% Save preds and oof

def save_files(
        df_train: pd.DataFrame,
        oof:np.ndarray, # Multi col with probabilitities
        preds:np.ndarray, 
        fname:str, 
        target:list[str],
        fold_metrics:list[float],
        fold_loglosses:list[float],
        XGB_PARAMS:dict,
        sub_rows:int,
        label_flag:bool=False,
        ):
    """
    Save all files, oof, preds and submissions
    """

    # Save utility files
    # Save full numpy arrays with probabilities.
    np.save(DIR / f'saved/{fname}_oof.npy', oof)
    np.save(DIR / f'saved/{fname}_preds.npy', preds)
    # Save sidecar
    meta = {
    'fname':    fname,
    'train':    TRAIN,
    'test':     TEST,
    'created':  datetime.date.today().isoformat(),
    'cv_auc':   float(np.mean(fold_metrics)),
    'cv_logloss': float(np.mean(fold_loglosses)),
    'params':   XGB_PARAMS,
    }
    with open(DIR / f'saved/{fname}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Save submissions file
    col = target[0]
    categories = df_train[col].cat.categories   
    df_sub = pd.read_csv(DIR / 'data/raw/sample_submission.csv')
    preds_ordinal = np.argmax(preds, axis=1)
    df_sub[col] = pd.Categorical.from_codes(
        codes = preds_ordinal,
        categories = categories
    )
        
    if label_flag:
        df_sub[col] = df_sub[col].astype(str)
        logger.info(f'Predictions decoded to labels: {list(categories)}')
    else:
        df_sub[col] = df_sub[col].cat.codes.astype('int8')
        logger.info(f'Predictions kept as integer codes')
        
        
    df_sub.to_csv(DIR / f'submissions/{fname}.csv', index=False)

    df_check = pd.read_csv(DIR / f'submissions/{fname}.csv')
    assert df_check.shape[1] == 2, 'the saved submission file has the wrong number of columns'
    assert df_check.shape[0] == sub_rows, 'the saved file has the wrong number of rows'
    return 


save_files(df_train, oof, preds, FNAME, TARGET, fold_metrics, fold_loglosses, XGB_PARAMS, SUB_ROWS, label_flag=True)

# %%
