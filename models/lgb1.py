import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
import shap
import datetime
import json
import argparse

from category_encoders import TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import log_loss, balanced_accuracy_score
import lightgbm as lgb
from lightgbm import LGBMClassifier

from utils.utils_reproducibility import set_seed, get_folds
from utils.utils_fit_models import plot_shap_bar, plot_shap_beeswarm
from utils.utils_basic import log_run

from config import (
    GlobalConfig, global_cfg,
    FeatureConfig, features,
    CompetitionConfig, DataConfig, LGBConfig, CVConfig,
    comp_cfg, data_cfg, lgb_cfg, cv_cfg,
    RAW_DIR, PROC_DIR, MODEL_DIR, SUB_DIR, ART_DIR,
    model_logger as log,
    pipeline_logger as pipe_log,
)
from config.env_cfg import make_exp_logger

pipe_log.info("[models.lgb] stage started")
set_seed()


# ── CLI arg parsing ───────────────────────────────────────────────────────────

def parse_args(
    data_cfg: DataConfig,
    lgb_cfg: LGBConfig
) -> tuple[DataConfig, LGBConfig]:

    parser = argparse.ArgumentParser(description="LightGBM training script")

    # DataConfig overrides
    parser.add_argument("--train-file",  type=str,   default=None)
    parser.add_argument("--test-file",   type=str,   default=None)
    parser.add_argument("--exp-name",    type=str,   default=None)
    parser.add_argument("--exp-notes",   type=str,   default=None)

    # LGBConfig overrides — mirrors LGBConfig fields
    parser.add_argument("--lr",               type=float, default=None, dest="learning_rate")
    parser.add_argument("--n-estimators",     type=int,   default=None)
    parser.add_argument("--num-leaves",       type=int,   default=None)
    parser.add_argument("--max-depth",        type=int,   default=None)
    parser.add_argument("--min-child-samples",type=int,   default=None)
    parser.add_argument("--subsample",        type=float, default=None)
    parser.add_argument("--colsample-bytree", type=float, default=None)
    parser.add_argument("--reg-alpha",        type=float, default=None)
    parser.add_argument("--reg-lambda",       type=float, default=None)

    args = parser.parse_args()
    all_args = {k: v for k, v in vars(args).items() if v is not None}

    data_keys = {"train_file", "test_file", "exp_name", "exp_notes"}
    lgb_keys  = {
        "learning_rate", "n_estimators", "num_leaves", "max_depth",
        "min_child_samples", "subsample", "colsample_bytree",
        "reg_alpha", "reg_lambda",
    }

    data_overrides = {k: v for k, v in all_args.items() if k in data_keys}
    lgb_overrides  = {k: v for k, v in all_args.items() if k in lgb_keys}

    if data_overrides:
        log.info(f"[cli] data overrides: {data_overrides}")
        data_cfg = data_cfg.model_copy(update=data_overrides)
    if lgb_overrides:
        log.info(f"[cli] lgb overrides:  {lgb_overrides}")
        lgb_cfg = lgb_cfg.model_copy(update=lgb_overrides)

    return data_cfg, lgb_cfg


data_cfg, lgb_cfg = parse_args(data_cfg, lgb_cfg)
exp_dir  = ART_DIR / data_cfg.exp_name
exp_dir.mkdir(parents=True, exist_ok=True)
exp_log  = make_exp_logger(exp_dir)
exp_log.info(f"Experiment: {data_cfg.exp_name}")
exp_log.info(f"Notes: {data_cfg.exp_notes}")


# ── Data import (identical to xgb1.py) ───────────────────────────────────────

def import_data(cfg: DataConfig, proc_dir: Path, target: str) \
        -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_path = cfg.train_path(proc_dir)
    test_path  = cfg.test_path(proc_dir)
    df_train = pd.read_parquet(train_path)
    df_test  = pd.read_parquet(test_path)
    cats = [c for c in df_train.select_dtypes('category').columns if c != target]
    nums = [c for c in df_train.select_dtypes('float').columns    if c != target]
    log.info(f"[load] {cfg.train_file}  shape={df_train.shape}")
    log.info(f"[load] {cfg.test_file}   shape={df_test.shape}")
    return df_train, df_test, cats, nums


# ── Training loop ─────────────────────────────────────────────────────────────

def train_model(
        df_train:   pd.DataFrame,
        df_test:    pd.DataFrame,
        cats:       list[str],
        target:     str,
        cv_cfg:     CVConfig,
        lgb_cfg:    LGBConfig,
        target_enc: bool = False):

    df_y = df_train[target].copy().cat.codes
    df_X = df_train.drop(columns=[target]).copy()
    df_X_test = df_test.copy()

    # Encode cats as int16 (LGB sklearn API accepts integer-coded cats)
    for col in cats:
        df_X[col]      = df_X[col].cat.codes.astype('int16')
        df_X_test[col] = df_X_test[col].cat.codes.astype('int16')

    df_cv_split = get_folds(df_X, df_y)

    n_classes = df_y.nunique()
    oof   = np.zeros((len(df_train), n_classes))
    preds = np.zeros((len(df_test),  n_classes))
    fold_metrics   = []
    fold_loglosses = []
    models = []

    for i, (train_index, valid_index) in enumerate(df_cv_split):
        Xtrain = df_X.iloc[train_index]
        ytrain = df_y.iloc[train_index]
        Xvalid = df_X.iloc[valid_index]
        yvalid = df_y.iloc[valid_index]
        Xtest  = df_X_test.copy()

        if target_enc:
            enc = TargetEncoder(cols=cats, min_samples_leaf=20, smoothing=10)
            enc.fit(Xtrain, ytrain)
            Xtrain = enc.transform(Xtrain)
            Xvalid = enc.transform(Xvalid)
            Xtest  = enc.transform(Xtest)

        sample_weights = compute_sample_weight(class_weight="balanced", y=ytrain)

        # LightGBM sklearn API — callbacks replace the XGBoost EarlyStopping object
        callbacks = [lgb.early_stopping(lgb_cfg.early_stopping_rounds, verbose=False),
                     lgb.log_evaluation(period=100)]

        model = LGBMClassifier(
            objective         = lgb_cfg.objective,
            metric            = lgb_cfg.metric,
            num_class         = n_classes,           # overrides config default
            num_leaves        = lgb_cfg.num_leaves,
            max_depth         = lgb_cfg.max_depth,
            learning_rate     = lgb_cfg.learning_rate,
            n_estimators      = lgb_cfg.n_estimators,
            min_child_samples = lgb_cfg.min_child_samples,
            subsample         = lgb_cfg.subsample,
            subsample_freq    = lgb_cfg.subsample_freq,
            colsample_bytree  = lgb_cfg.colsample_bytree,
            reg_alpha         = lgb_cfg.reg_alpha,
            reg_lambda        = lgb_cfg.reg_lambda,
            random_state      = lgb_cfg.seed,
            n_jobs            = -1,
            verbosity         = lgb_cfg.verbosity,
        )

        model.fit(
            Xtrain, ytrain,
            sample_weight    = sample_weights,
            eval_set         = [(Xvalid, yvalid)],
            callbacks        = callbacks,
        )

        models.append(model)
        ypred_proba = model.predict_proba(Xvalid)
        y_pred      = model.predict(Xvalid)

        fold_logloss = log_loss(yvalid, ypred_proba)
        fold_metric  = balanced_accuracy_score(yvalid, y_pred)
        oof[valid_index] = ypred_proba

        fold_loglosses.append(fold_logloss)
        fold_metrics.append(fold_metric)
        log.info(f'Fold {i+1}, Log loss: {fold_logloss:.5f}, metric: {fold_metric:.5f}')

        preds += model.predict_proba(Xtest) / cv_cfg.n_folds

    log.info(f"Overall Score, logloss: {np.mean(fold_loglosses):.5f}, "
             f"metric: {np.mean(fold_metrics):.5f}")
    return models, Xtrain, Xvalid, oof, preds, fold_metrics, fold_loglosses


# ── Feature importance (LGB native) ──────────────────────────────────────────

def model_importance(model: LGBMClassifier, Xtrain: pd.DataFrame):
    import matplotlib.pyplot as plt
    lgb.plot_importance(model.booster_, max_num_features=30,
                        importance_type="gain", figsize=(10, 8))
    plt.tight_layout()
    plt.savefig(exp_dir / "feature_importance.png", dpi=150)
    plt.close()
    # Save as parquet for notebook reuse
    feat_imp = pd.DataFrame({
        "feature":   model.booster_.feature_name(),
        "importance": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    feat_imp.to_parquet(exp_dir / "feature_importance.parquet", index=False)


# ── SHAP (identical to xgb1.py — TreeExplainer works for LGB too) ────────────

def model_shap(model, Xtrain, Xvalid, exp_dir, random_state):
    X_bg      = Xtrain.sample(n=200,  random_state=random_state)
    X_explain = Xvalid.sample(n=1000, random_state=random_state)

    explainer = shap.TreeExplainer(model, data=X_bg,
                                   feature_perturbation="interventional")
    sv_raw = np.array(explainer.shap_values(X_explain))
    print(f"[shap] raw shape: {sv_raw.shape}")

    if sv_raw.ndim == 3:
        sv_2d = np.mean(np.abs(sv_raw), axis=2)
    elif sv_raw.ndim == 2:
        sv_2d = sv_raw
    else:
        raise ValueError(f"Unexpected SHAP output shape: {sv_raw.shape}")

    base = float(np.mean(explainer.expected_value))
    sv = shap.Explanation(values=sv_2d,
                          base_values=np.full(len(X_explain), base),
                          data=X_explain.values,
                          feature_names=X_explain.columns.tolist())

    if sv_raw.ndim == 3:
        frames = [pd.DataFrame(sv_raw[:, :, i],
                               columns=[f"{c}_class{i}" for c in X_explain.columns])
                  for i in range(sv_raw.shape[2])]
        pd.concat(frames, axis=1).to_parquet(exp_dir / "shap_values.parquet")
    else:
        pd.DataFrame(sv_2d, columns=X_explain.columns).to_parquet(
            exp_dir / "shap_values.parquet")

    plot_shap_bar(sv, exp_dir)
    plot_shap_beeswarm(sv, exp_dir)


# ── Save files (one key change: "lgb" replaces "xgb" in snapshot) ────────────

def save_files(df_train, oof, preds, fold_metrics, fold_loglosses,
               comp_cfg, data_cfg, features, lgb_cfg, cv_cfg, global_cfg,
               label_flag=False):

    snapshot = {
        "comp":     comp_cfg.model_dump(),
        "data":     data_cfg.model_dump(),
        "features": features.model_dump(),
        "lgb":      lgb_cfg.model_dump(),          # ← "lgb" not "xgb"
        "cv":       cv_cfg.model_dump(),
        "global":   global_cfg.model_dump(),
    }
    (exp_dir / "config.json").write_text(json.dumps(snapshot, indent=2))
    log.info(f"[config] saved → {exp_dir / 'config.json'}")

    np.save(exp_dir / f'{data_cfg.exp_name}_oof.npy',   oof)
    np.save(exp_dir / f'{data_cfg.exp_name}_preds.npy', preds)

    meta = {
        "fname":      data_cfg.exp_name,
        "train":      data_cfg.train_file,
        "test":       data_cfg.test_file,
        "created":    datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        "cv_auc":     float(np.mean(fold_metrics)),
        "cv_logloss": float(np.mean(fold_loglosses)),
        "params":     lgb_cfg.model_dump(),
    }
    with open(exp_dir / f'{data_cfg.exp_name}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    col        = features.target
    categories = df_train[col].cat.categories
    df_sub     = pd.read_csv(comp_cfg.sample_sub_path(RAW_DIR))
    preds_ordinal = np.argmax(preds, axis=1)
    df_sub[col] = pd.Categorical.from_codes(codes=preds_ordinal, categories=categories)

    if label_flag:
        df_sub[col] = df_sub[col].astype(str)
    else:
        df_sub[col] = df_sub[col].cat.codes.astype('int8')

    df_sub.to_csv(SUB_DIR / f'{data_cfg.exp_name}.csv', index=False)
    df_sub.to_csv(exp_dir  / f'{data_cfg.exp_name}.csv', index=False)

    df_check = pd.read_csv(SUB_DIR / f'{data_cfg.exp_name}.csv')
    assert df_check.shape[1] == 2
    assert df_check.shape[0] == comp_cfg.n_test

    run_record = {
        "date":            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exp_name":        data_cfg.exp_name,
        "exp_notes":       data_cfg.exp_notes,
        "train_file":      data_cfg.train_file,
        "n_folds":         cv_cfg.n_folds,
        "mean_cv_metric":  round(float(np.mean(fold_metrics)),   5),
        "std_cv_metric":   round(float(np.std(fold_metrics)),    5),
        "mean_logloss":    round(float(np.mean(fold_loglosses)), 5),
        "n_features":      df_train.shape[1],
        "lgb_params":      json.dumps(lgb_cfg.model_dump()),
    }
    log_run(ART_DIR, run_record)
    pipe_log.info(f"Run logged to {ART_DIR / 'run_log.csv'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df_train, df_test, cats, nums = import_data(data_cfg, PROC_DIR, features.target)
    models, Xtrain, Xvalid, oof, preds, fold_metrics, fold_loglosses = \
        train_model(df_train, df_test, cats, features.target, cv_cfg, lgb_cfg)
    model_importance(models[-1], Xtrain)
    # model_shap(models[-1], Xtrain, Xvalid, exp_dir, lgb_cfg.seed)
    save_files(df_train, oof, preds, fold_metrics, fold_loglosses,
               comp_cfg, data_cfg, features, lgb_cfg, cv_cfg, global_cfg,
               label_flag=True)


if __name__ == "__main__":
    main()