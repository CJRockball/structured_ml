import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import logging
import shap 

from category_encoders import TargetEncoder

from sklearn.metrics import auc, log_loss, roc_curve, \
    roc_auc_score, root_mean_squared_error, balanced_accuracy_score

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import datetime
import json
import argparse

from utils.utils_reproducibility import set_seed, get_folds
from utils.utils_fit_models import plot_shap_bar, plot_shap_beeswarm, plot_xgb_importance
from utils.utils_basic import log_run

from config import (
    GlobalConfig, global_cfg,          
    FeatureConfig, features,
    CompetitionConfig, DataConfig, XGBConfig, CVConfig,
    comp_cfg, data_cfg, xgb_cfg, cv_cfg,
    RAW_DIR, PROC_DIR, MODEL_DIR, SUB_DIR, ART_DIR,
    model_logger as log,
    pipeline_logger as pipe_log,
)
from config.env_cfg import make_exp_logger
import os
import joblib

pipe_log.info("[models.xgb] stage started")
set_seed()


def parse_args(
    data_cfg: DataConfig,
    xgb_cfg: XGBConfig
) -> tuple[DataConfig, XGBConfig]:

    parser = argparse.ArgumentParser(description="XGBoost training script")

    # ── DataConfig overrides ──────────────────────────────────
    parser.add_argument("--train-file",  type=str,   default=None)
    parser.add_argument("--test-file",   type=str,   default=None)
    parser.add_argument("--exp-name",    type=str,   default=None)
    parser.add_argument("--exp-notes",   type=str,   default=None)

    # ── XGBConfig overrides ───────────────────────────────────
    parser.add_argument("--lr",           type=float, default=None, dest="learning_rate")
    parser.add_argument("--n-estimators", type=int,   default=None)
    parser.add_argument("--max-depth",    type=int,   default=None)
    parser.add_argument("--min-child-weight", type=int, default=None)
    parser.add_argument("--subsample",    type=float, default=None)
    parser.add_argument("--colsample-bytree", type=float, default=None)
    parser.add_argument("--reg-alpha",    type=float, default=None)
    parser.add_argument("--reg-lambda",   type=float, default=None)

    args = parser.parse_args()
    all_args = {k: v for k, v in vars(args).items() if v is not None}

    # Split overrides by config ownership
    data_keys = {"train_file", "test_file", "exp_name", "exp_notes"}
    xgb_keys  = {
        "learning_rate", "n_estimators", "max_depth", "min_child_weight",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"
    }

    # argparse uses underscores internally after dest= or hyphen→underscore conversion
    data_overrides = {k: v for k, v in all_args.items() if k in data_keys}
    xgb_overrides  = {k: v for k, v in all_args.items() if k in xgb_keys}

    if data_overrides:
        log.info(f"[cli] data overrides: {data_overrides}")
        data_cfg = data_cfg.model_copy(update=data_overrides)

    if xgb_overrides:
        log.info(f"[cli] xgb overrides:  {xgb_overrides}")
        xgb_cfg = xgb_cfg.model_copy(update=xgb_overrides)  # Pydantic re-validates

    return data_cfg, xgb_cfg

# Replace the module-level data_cfg usage with the parsed version
data_cfg, xgb_cfg = parse_args(data_cfg, xgb_cfg)         # ← add this line
# Experiment notes
exp_dir  = ART_DIR / data_cfg.exp_name
# assert not exp_dir.exists(), (
#         f"Experiment folder already exists: {exp_dir}\n"
#         f"Rename exp_name in DataConfig or delete the folder to rerun."
#     )
exp_dir.mkdir(parents=True, exist_ok=True)
exp_log  = make_exp_logger(exp_dir)   # live from this point forward

exp_log.info(f"Experiment: {data_cfg.exp_name}")
exp_log.info(f"Notes: {data_cfg.exp_notes}")





def import_data(cfg:DataConfig, proc_dir:Path, target:str) \
        -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """
    Function to import train and test data.
    """
    
    train_path = cfg.train_path(proc_dir)
    test_path  = cfg.test_path(proc_dir)

    df_train = pd.read_parquet(train_path)
    df_test  = pd.read_parquet(test_path)

    # Get feature types
    cats = [c for c in df_train.select_dtypes('category').columns if c != target]
    nums = [c for c in df_train.select_dtypes('float').columns if c != target]
    
    log.info(f"[load] {cfg.train_file}  shape={df_train.shape}")
    log.info(f"[load] {cfg.test_file}   shape={df_test.shape}")
    return df_train, df_test, cats, nums



def train_model(
                df_train:pd.DataFrame, 
                df_test:pd.DataFrame, 
                cats, 
                target:str,
                cv_cfg:CVConfig,
                xgb_cfg:XGBConfig, 
                target_enc:bool=True):
    
    df_y = df_train[target].copy().cat.codes
    df_X = df_train.drop(columns=[target]).copy()
    df_X_test = df_test.copy()
    
    for col in cats:
        df_X[col] = df_X[col].cat.codes.astype('int16')
        df_X_test[col] = df_X_test[col].cat.codes.astype('int16')
    
    df_cv_split = get_folds(df_X, df_y)
    
    
    n_classes = df_y.nunique()
    oof = np.zeros((len(df_train),n_classes))
    preds = np.zeros((len(df_test),n_classes))
    fold_metrics = []
    fold_loglosses = []
    models = []
    for i,(train_index, valid_index) in enumerate(df_cv_split): # skf.split(df_X, df_y)):
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
        rounds=xgb_cfg.early_stopping_rounds,
        min_delta=1e-3,
        save_best=True,
        maximize=False,
        data_name="validation_0",
        metric_name=xgb_cfg.eval_metric,)

        model = XGBClassifier(**xgb_cfg.to_dict(), callbacks=[es])
                
        model = model.fit(Xtrain, ytrain, 
                        eval_set=[(Xvalid, yvalid)],
                        verbose=100) #xgb_cfg.early_stopping_rounds)   
        
        models.append(model)
        ypred_proba = model.predict_proba(Xvalid)
        y_pred = model.predict(Xvalid)
        
        
        fold_logloss = log_loss(yvalid, ypred_proba)
        fold_metric = balanced_accuracy_score(yvalid, y_pred)
        oof[valid_index] = ypred_proba  # Save as multi col with percentages

        # Save
        fold_loglosses.append(fold_logloss)
        fold_metrics.append(fold_metric)
        log.info(f'Fold {i+1}, Log loss: {fold_logloss:.5f}, metric: {fold_metric:.5f}')

        preds += model.predict_proba(Xtest) / cv_cfg.n_folds
    
       
    log.info(f"Overall Score, logloss: {np.mean(fold_loglosses):.5f}, metric: {np.mean(fold_metrics):.5f}")
    return models, Xtrain, Xvalid, oof, preds, fold_metrics, fold_loglosses



def model_importance(model, Xtrain):
    # # Get feature importance scores
    feature_list = Xtrain.columns
    plot_xgb_importance(model, feature_list, exp_dir, max_display=30)
    return



def model_shap(model, Xtrain, Xvalid, exp_dir, random_state):
    X_bg      = Xtrain.sample(n=200,  random_state=random_state)
    X_explain = Xvalid.sample(n=1000, random_state=random_state)

    explainer = shap.TreeExplainer(
        model,
        data                 = X_bg,
        feature_perturbation = "interventional",
    )

    sv_raw = np.array(explainer.shap_values(X_explain))
    print(f"[shap] raw shape: {sv_raw.shape}")

    # Normalise to (n_samples, n_features) regardless of output format
    if sv_raw.ndim == 3:
        # (n_samples, n_features, n_classes) → mean abs across classes
        sv_2d = np.mean(np.abs(sv_raw), axis=2)
    elif sv_raw.ndim == 2:
        # (n_samples, n_features) — regression or binary
        sv_2d = sv_raw
    else:
        raise ValueError(f"Unexpected SHAP output shape: {sv_raw.shape}")

    print(f"[shap] plot shape: {sv_2d.shape}")  # always (n_samples, n_features)
    
    base = float(np.mean(explainer.expected_value))

    sv = shap.Explanation(
        values        = sv_2d,
        base_values   = np.full(len(X_explain), base),  # ← (n_samples,) not scalar
        data          = X_explain.values,
        feature_names = X_explain.columns.tolist(),
    )

    # Save — per-class columns if multiclass
    if sv_raw.ndim == 3:
        frames = [
            pd.DataFrame(sv_raw[:, :, i], columns=[f"{c}_class{i}" for c in X_explain.columns])
            for i in range(sv_raw.shape[2])
        ]
        pd.concat(frames, axis=1).to_parquet(exp_dir / "shap_values.parquet")
    else:
        pd.DataFrame(sv_2d, columns=X_explain.columns).to_parquet(exp_dir / "shap_values.parquet")

    print(f"[shap] saved → {exp_dir / 'shap_values.parquet'}")

    plot_shap_bar(sv,      exp_dir)
    plot_shap_beeswarm(sv, exp_dir)
    return


def save_files(
        df_train: pd.DataFrame,
        oof:np.ndarray, # Multi col with probabilitities
        preds:np.ndarray, 
        fold_metrics:list[float],
        fold_loglosses:list[float],
        comp_cfg:CompetitionConfig, 
        data_cfg:DataConfig, 
        features:FeatureConfig, 
        xgb_cfg:XGBConfig, 
        cv_cfg:CVConfig, 
        global_cfg:GlobalConfig,
        label_flag:bool=False,
        ):
    """
    Save all files, oof, preds and submissions
    """

    # ── Config snapshot — everything needed to reproduce ──────────────────────
    snapshot = {
        "comp":     comp_cfg.model_dump(),
        "data":     data_cfg.model_dump(),
        "features": features.model_dump(),
        "xgb":      xgb_cfg.model_dump(),
        "cv":       cv_cfg.model_dump(),
        "global":   global_cfg.model_dump(),
    }
    (exp_dir / "config.json").write_text(json.dumps(snapshot, indent=2))
    log.info(f"[config] saved → {exp_dir / 'config.json'}")

    
    # Save utility files
    # Save full numpy arrays with probabilities.
    np.save(exp_dir / f'{data_cfg.exp_name}_oof.npy', oof)
    np.save(exp_dir / f'{data_cfg.exp_name}_preds.npy', preds)
    # Save sidecar
    meta = {
    'fname':    data_cfg.exp_name,
    'train':    data_cfg.train_file,
    'test':     data_cfg.test_file,
    'created':  datetime.datetime.now().strftime("%Y%m%d_%H%M"), # "20260604_1042"
    'cv_auc':   float(np.mean(fold_metrics)),
    'cv_logloss': float(np.mean(fold_loglosses)),
    'params':   xgb_cfg.to_dict(),
    }
    with open(exp_dir / f'{data_cfg.exp_name}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)


    # Save submissions file
    col = features.target
    categories = df_train[col].cat.categories   
    df_sub = pd.read_csv(comp_cfg.sample_sub_path(RAW_DIR))
    preds_ordinal = np.argmax(preds, axis=1)
    df_sub[col] = pd.Categorical.from_codes(
        codes = preds_ordinal,
        categories = categories
    )
        
    if label_flag:
        df_sub[col] = df_sub[col].astype(str)
        log.info(f'Predictions decoded to labels: {list(categories)}')
    else:
        df_sub[col] = df_sub[col].cat.codes.astype('int8')
        log.info(f'Predictions kept as integer codes')
        
        
    df_sub.to_csv(SUB_DIR / f'{data_cfg.exp_name}.csv', index=False)
    df_sub.to_csv(exp_dir / f'{data_cfg.exp_name}.csv', index=False)

    df_check = pd.read_csv(SUB_DIR / f'{data_cfg.exp_name}.csv')
    assert df_check.shape[1] == 2, 'the saved submission file has the wrong number of columns'
    assert df_check.shape[0] == comp_cfg.n_test, 'the saved file has the wrong number of rows'
    
        
    # build the record — all values already in scope
    run_record = {
        "date":          datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exp_name":      data_cfg.exp_name,
        "exp_notes":     data_cfg.exp_notes,
        "train_file":    data_cfg.train_file,
        "n_folds":       cv_cfg.n_folds,
        "mean_cv_metric":   round(float(np.mean(fold_metrics)), 5),
        "std_cv_metric":    round(float(np.std(fold_metrics)), 5),
        "mean_logloss":  round(float(np.mean(fold_loglosses)), 5),
        #"oof_auc":       round(float(roc_auc_score(df_train[features.target], oof)), 5),
        "n_features":    df_train.shape[1],
        "xgb_params":    json.dumps(xgb_cfg.to_dict()),   # json already imported
    }

    log_run(ART_DIR, run_record)
    pipe_log.info(f"Run logged to {ART_DIR / 'run_log.csv'}")
        
    return 


def main():
    """
    Executes all the functions
    """
    
    df_train, df_test, cats, nums = import_data(data_cfg, PROC_DIR, features.target)
    models, Xtrain, Xvalid, oof, preds, fold_metrics, fold_loglosses = \
             train_model(df_train, df_test, cats, features.target, cv_cfg, xgb_cfg)
    model_importance(models[-1], Xtrain)
    #model_shap(models[-1], Xtrain, Xvalid, exp_dir, xgb_cfg.seed)
    save_files(df_train, oof, preds, fold_metrics, fold_loglosses,
           comp_cfg, data_cfg, features, xgb_cfg, cv_cfg, global_cfg,
           label_flag=True)
    return

if __name__ == "__main__":
    main()
