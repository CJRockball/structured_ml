import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import datetime
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from utils.utils_basic import log_run
from config import (
    global_cfg, features, comp_cfg,
    RAW_DIR, PROC_DIR, SUB_DIR, ART_DIR,
    pipeline_logger as pipe_log,
)
from config.env_cfg import make_exp_logger
from ensemble import (
    load_artifacts,
    load_labels,
    renorm,
    compute_oof_metrics,
)


pipe_log.info("[run_meta] started")


# =============================================================================
#  USER CONFIG — edit this section only
# =============================================================================

TRAIN_FILE = None  # global default — auto-detected from first model's meta.json

# Each key becomes the artifact folder + submission CSV name.
# Each value:
#   models      : list of exp_name strings whose OOF/pred .npy files to stack
#   meta_learner: "logreg" | "hgbc"
#   n_folds     : inner CV folds for meta-learner OOF (default 5)
#   notes       : free-text saved to config.json
#   train_file  : override PROC_DIR parquet for ground-truth labels
#   label_flag  : True → decode predictions to class label strings

META_RUNS = {

    # ── Logistic Regression stacker on all 12 base models ─────────────────
    "meta_logreg_all12": dict(
        models= ['tune_xgb_params_v1_ve4_no_te_target_weight',
                 'tune_xgb_params_v2_ve4_no_te_target_weight', 
                 'tune_xgb_params_v3_ve4_no_te_target_weight', 
                 'tune_xgb_params_v4_ve4_no_te_target_weight', 
                 'tune_xgb_params_v5_ve4_no_te_target_weight', 
                 'lgb1_tune_lgb_params_v1_ve4_no_te_target_weight', 
                 'lgb1_tune_lgb_params_v2_ve4_no_te_target_weight', 
                 'lgb1_tune_lgb_params_v3_ve4_no_te_target_weight', 
                 'lgb1_tune_lgb_params_v4_ve4_no_te_target_weight', 
                 'lgb1_tune_lgb_params_v5_ve4_no_te_target_weight', 
                 'cat1_tune_cat_params_v1_ve4_no_te_target_weight', 
                 'cat1_tune_cat_params_v2_ve4_no_te_target_weight'],
        meta_learner = "logreg",
        n_folds      = 5,
        notes        = "GPU LogReg stacker on all 12 base model OOFs",
        train_file   = TRAIN_FILE,
        label_flag   = True,
    ),

    # ── HGBC stacker on all 12 base models ────────────────────────────────
    "meta_hgbc_all12": dict(
        models=["tune_xgb_params_v1_ve4_no_te_target_weight", 
                 "tune_xgb_params_v2_ve4_no_te_target_weight", 
                 "tune_xgb_params_v3_ve4_no_te_target_weight",
                 "tune_xgb_params_v4_ve4_no_te_target_weight", 
                 "tune_xgb_params_v5_ve4_no_te_target_weight", 
                 "lgb1_tune_lgb_params_v1_ve4_no_te_target_weight", 
                 "lgb1_tune_lgb_params_v2_ve4_no_te_target_weight",
                 "lgb1_tune_lgb_params_v3_ve4_no_te_target_weight", 
                 "lgb1_tune_lgb_params_v4_ve4_no_te_target_weight", 
                 "lgb1_tune_lgb_params_v5_ve4_no_te_target_weight", 
                 "cat1_tune_cat_params_v1_ve4_no_te_target_weight", 
                 "cat1_tune_cat_params_v2_ve4_no_te_target_weight",],
        meta_learner = "hgbc",
        n_folds      = 5,
        notes        = "HGBC stacker on all 12 base model OOFs",
        train_file   = TRAIN_FILE,
        label_flag   = True,
    ),

    # ── Logistic Regression stacker: GBDTs only (no cat) ──────────────────
    "meta_logreg_xgb_lgb": dict(
        models=["tune_xgb_params_v1_ve4_no_te_target_weight",
                 "tune_xgb_params_v2_ve4_no_te_target_weight", 
                 "tune_xgb_params_v3_ve4_no_te_target_weight", 
                 "tune_xgb_params_v4_ve4_no_te_target_weight", 
                 "tune_xgb_params_v5_ve4_no_te_target_weight", 
                 "lgb1_tune_lgb_params_v1_ve4_no_te_target_weight", 
                 "lgb1_tune_lgb_params_v2_ve4_no_te_target_weight", 
                 "lgb1_tune_lgb_params_v3_ve4_no_te_target_weight", 
                 "lgb1_tune_lgb_params_v4_ve4_no_te_target_weight", 
                 "lgb1_tune_lgb_params_v5_ve4_no_te_target_weight", 
                 "cat1_tune_cat_params_v1_ve4_no_te_target_weight", 
                 "cat1_tune_cat_params_v2_ve4_no_te_target_weight",],
        meta_learner = "logreg",
        n_folds      = 5,
        notes        = "LogReg stacker on XGB + LGB only (10 models)",
        train_file   = TRAIN_FILE,
        label_flag   = True,
    ),

    # ── HGBC stacker: XGB-only for diagnostics ────────────────────────────
    # "meta_hgbc_xgb_only": dict(
    #     models      = ["xgb_v1", "xgb_v2", "xgb_v3", "xgb_v4", "xgb_v5"],
    #     meta_learner = "hgbc",
    #     n_folds     = 5,
    #     notes       = "HGBC stacker on 5x XGB only",
    #     train_file  = TRAIN_FILE,
    #     label_flag  = True,
    # ),
}


# =============================================================================
#  END USER CONFIG — nothing below needs editing
# =============================================================================


# ── Meta-learner constructors ─────────────────────────────────────────────────

def make_meta_learner(kind: str):
    """Return an unfitted sklearn-compatible meta-learner."""
    if kind == "logreg":
        return LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
            n_jobs=-1,
        )
    elif kind == "hgbc":
        return HistGradientBoostingClassifier(
            max_leaf_nodes=135,
            min_samples_leaf=600,
            l2_regularization=40.0,
            learning_rate=0.05,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            class_weight="balanced",
            random_state=global_cfg.seed,
        )
    else:
        raise ValueError(f"Unknown meta_learner: '{kind}'. Use 'logreg' or 'hgbc'.")


# ── Stacking helpers ──────────────────────────────────────────────────────────

def build_meta_features(
    oof_list: list[np.ndarray],
    pred_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate per-model OOF and test probability arrays into a single
    meta-feature matrix.

    Each OOF array  : (n_train, n_classes)
    Each pred array : (n_test,  n_classes)

    Returns
    -------
    X_meta_train : (n_train, n_models * n_classes)
    X_meta_test  : (n_test,  n_models * n_classes)
    """
    X_meta_train = np.hstack(oof_list)   # (n_train, n_models * n_classes)
    X_meta_test  = np.hstack(pred_list)  # (n_test,  n_models * n_classes)
    return X_meta_train, X_meta_test


def fit_meta_learner(
    X_meta: np.ndarray,
    y: np.ndarray,
    pred_meta: np.ndarray,
    meta_learner_kind: str,
    n_folds: int,
    log: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """
    Fit the meta-learner using nested StratifiedKFold so the meta OOF
    predictions are themselves out-of-fold (no leakage).

    Returns
    -------
    meta_oof   : (n_train, n_classes)  — OOF probabilities of meta-learner
    meta_preds : (n_classes,)          — averaged test probabilities
    fold_lls   : list of per-fold log-loss values
    fold_baccs : list of per-fold balanced accuracy values
    """
    skf        = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    n_classes  = pred_meta.shape[1]
    meta_oof   = np.zeros((len(y), n_classes), dtype=np.float32)
    test_preds = np.zeros((pred_meta.shape[0], n_classes), dtype=np.float64)
    fold_lls   = []
    fold_baccs = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_meta, y)):
        X_tr, X_val = X_meta[tr_idx], X_meta[val_idx]
        y_tr, y_val = y[tr_idx],      y[val_idx]

        clf = make_meta_learner(meta_learner_kind)
        clf.fit(X_tr, y_tr)

        val_proba = clf.predict_proba(X_val)
        meta_oof[val_idx] = val_proba.astype(np.float32)

        test_preds += clf.predict_proba(pred_meta)

        ll   = log_loss(y_val, val_proba)
        bacc = balanced_accuracy_score(y_val, np.argmax(val_proba, axis=1))
        fold_lls.append(ll)
        fold_baccs.append(bacc)
        log.info(
            f"  fold {fold_idx + 1}/{n_folds}  "
            f"logloss={ll:.5f}  bacc={bacc:.5f}"
        )

    test_preds /= n_folds
    return meta_oof, test_preds, fold_lls, fold_baccs


# ── Main per-run function ─────────────────────────────────────────────────────

def run_single_meta(
    run_name: str,
    cfg: dict,
    log: logging.Logger,
) -> dict:
    """
    Run one meta-learner stacking config.
    Returns a summary dict for the results table.
    """
    models      = cfg["models"]
    kind        = cfg["meta_learner"]
    n_folds     = cfg.get("n_folds", 5)
    notes       = cfg.get("notes", "")
    train_file  = cfg.get("train_file", None)
    label_flag  = cfg.get("label_flag", True)

    exp_dir = ART_DIR / run_name
    if exp_dir.exists():
        log.warning(f"[skip] {run_name} — folder already exists, skipping.")
        return {
            "name": run_name, "status": "skipped",
            "logloss": None, "bacc": None,
        }

    exp_dir.mkdir(parents=True)
    exp_log = make_exp_logger(exp_dir)
    exp_log.info(f"Meta run    : {run_name}")
    exp_log.info(f"Learner     : {kind}")
    exp_log.info(f"Base models : {models}")
    exp_log.info(f"Inner folds : {n_folds}")
    exp_log.info(f"Notes       : {notes}")

    # ── Load base model artifacts ──────────────────────────────────────────
    oofs, preds, metas = load_artifacts(models, ART_DIR, exp_log)
    y, categories      = load_labels(train_file, metas, PROC_DIR, features.target, exp_log)

    oof_list  = [oofs[n]  for n in models]
    pred_list = [preds[n] for n in models]

    # ── Build meta-feature matrices ────────────────────────────────────────
    X_meta_train, X_meta_test = build_meta_features(oof_list, pred_list)
    exp_log.info(
        f"Meta-feature matrix  train={X_meta_train.shape}  "
        f"test={X_meta_test.shape}"
    )

    # ── Fit meta-learner (nested CV) ───────────────────────────────────────
    meta_oof, meta_preds, fold_lls, fold_baccs = fit_meta_learner(
        X_meta        = X_meta_train,
        y             = y,
        pred_meta     = X_meta_test,
        meta_learner_kind = kind,
        n_folds       = n_folds,
        log           = exp_log,
    )

    meta_oof   = renorm(meta_oof)
    meta_preds = renorm(meta_preds)

    mean_ll   = float(np.mean(fold_lls))
    std_ll    = float(np.std(fold_lls))
    mean_bacc = float(np.mean(fold_baccs))
    std_bacc  = float(np.std(fold_baccs))

    exp_log.info(
        f"[meta CV]  logloss={mean_ll:.5f} ± {std_ll:.5f}  "
        f"bacc={mean_bacc:.5f} ± {std_bacc:.5f}"
    )

    # Final OOF metrics (full dataset)
    ll_full, bacc_full = compute_oof_metrics(meta_oof, y, exp_log, label=run_name)

    # ── Save config snapshot ───────────────────────────────────────────────
    snapshot = {
        "run_name":       run_name,
        "run_notes":      notes,
        "meta_learner":   kind,
        "n_folds":        n_folds,
        "base_models":    models,
        "created":        datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        "global":         global_cfg.model_dump(),
        "source_metas":   {name: metas[name] for name in models},
        "fold_logloss":   fold_lls,
        "fold_bacc":      fold_baccs,
    }
    (exp_dir / "config.json").write_text(json.dumps(snapshot, indent=2))

    # ── Arrays ────────────────────────────────────────────────────────────
    np.save(exp_dir / f"{run_name}_oof.npy",   meta_oof)
    np.save(exp_dir / f"{run_name}_preds.npy", meta_preds)

    # ── Meta sidecar ──────────────────────────────────────────────────────
    meta_json = {
        "fname":      run_name,
        "meta_learner": kind,
        "base_models": models,
        "created":    datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        "cv_auc":     float(bacc_full),
        "cv_logloss": float(ll_full),
        "mean_fold_bacc": mean_bacc,
        "std_fold_bacc":  std_bacc,
        "mean_fold_ll":   mean_ll,
        "std_fold_ll":    std_ll,
    }
    (exp_dir / f"{run_name}_meta.json").write_text(json.dumps(meta_json, indent=2))

    # ── Submission CSV ─────────────────────────────────────────────────────
    target   = features.target
    df_sub   = pd.read_csv(comp_cfg.sample_sub_path(RAW_DIR))
    pred_ord = np.argmax(meta_preds, axis=1)
    df_sub[target] = pd.Categorical.from_codes(codes=pred_ord, categories=categories)