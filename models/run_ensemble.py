import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import datetime
import logging

from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import log_loss, balanced_accuracy_score

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
    blend_equal,
    blend_rank,
    blend_optimised,
    compute_oof_metrics,
    renorm,
)

pipe_log.info("[run_ensemble] started")

# =============================================================================
#  USER CONFIG — edit this section only
# =============================================================================

# Each key is the ensemble name (becomes the artifact folder + submission CSV).
# Each value is a dict with:
#   models        : list of exp_name strings (must exist under ART_DIR)
#   strategy      : "equal_avg" | "rank_avg" | "opt_weights"
#   notes         : free-text saved to config.json
#   train_file    : parquet in PROC_DIR for ground-truth labels
#                   set to None to auto-detect from the first model's meta.json
#   label_flag    : True → decode to class label strings (default True)

TRAIN_FILE = None  # global default — overridden per-run if needed

ENSEMBLES = {
    # ── Baseline: equal-weight average of all three GBDTs ─────────────────
    "ensemble_equal_v3": dict(
        models      = ["xgb1_tune_base fe_Target_weighting_no_te", 
                       "lgb1_tune_base fe_Target_weighting_no_te_fe_ve3"],
        strategy    = "equal_avg",
        notes       = "Equal-weight average of ve3",
        train_file  = TRAIN_FILE,
        label_flag  = True,
    ),

    # ── Optimised weights on OOF logloss ──────────────────────────────────
    "ensemble_opt_v3": dict(
        models      = ["xgb1_tune_base fe_Target_weighting_no_te", 
                       "lgb1_tune_base fe_Target_weighting_no_te_fe_ve3"],
        strategy    = "opt_weights",
        notes       = "Nelder-Mead weights, ve3",
        train_file  = TRAIN_FILE,
        label_flag  = True,
    ),

    # ── Two-model pair: XGB + CAT only ────────────────────────────────────
    # "ensemble_xgb_cat": dict(
    #     models      = ["xgb1_base", "cat1_base"],
    #     strategy    = "opt_weights",
    #     notes       = "XGB + CAT, optimised weights",
    #     train_file  = TRAIN_FILE,
    #     label_flag  = True,
    # ),

    # ── FE variant models ─────────────────────────────────────────────────
    # "ensemble_fe_v1": dict(
    #     models      = ["xgb1_fe_v1", "lgb1_fe_v1", "cat1_fe_v1"],
    #     strategy    = "opt_weights",
    #     notes       = "All three models trained on fe_v1 parquet",
    #     train_file  = None,
    #     label_flag  = True,
    # ),

    # ── Cross-FE ensemble: mix base and FE runs ───────────────────────────
    # "ensemble_mixed_fe": dict(
    #     models      = ["xgb1_base", "xgb1_fe_v1", "lgb1_fe_v1"],
    #     strategy    = "rank_avg",
    #     notes       = "Rank avg across base and FE parquet models",
    #     train_file  = None,
    #     label_flag  = True,
    # ),
}

# =============================================================================
#  END USER CONFIG — nothing below needs editing
# =============================================================================


def run_single(
    ensemble_name: str,
    cfg: dict,
    log: logging.Logger,
) -> dict:
    """Run one ensemble config. Returns a summary dict for the results table."""
    models     = cfg["models"]
    strategy   = cfg["strategy"]
    notes      = cfg.get("notes", "")
    train_file = cfg.get("train_file", None)
    label_flag = cfg.get("label_flag", True)

    exp_dir = ART_DIR / ensemble_name
    if exp_dir.exists():
        log.warning(f"[skip] {ensemble_name} — folder already exists, skipping.")
        return {"name": ensemble_name, "status": "skipped", "logloss": None, "bacc": None}

    exp_dir.mkdir(parents=True)
    exp_log = make_exp_logger(exp_dir)
    exp_log.info(f"Ensemble : {ensemble_name}")
    exp_log.info(f"Strategy : {strategy}")
    exp_log.info(f"Models   : {models}")
    exp_log.info(f"Notes    : {notes}")

    # Load
    oofs, preds, metas = load_artifacts(models, ART_DIR, exp_log)
    y, categories      = load_labels(train_file, metas, PROC_DIR, features.target, exp_log)

    oof_list  = [oofs[n]  for n in models]
    pred_list = [preds[n] for n in models]

    # Blend
    weights = None
    if strategy == "equal_avg":
        oof_blended   = blend_equal(oof_list)
        preds_blended = blend_equal(pred_list)

    elif strategy == "rank_avg":
        oof_blended   = blend_rank(oof_list)
        preds_blended = blend_rank(pred_list)

    elif strategy == "opt_weights":
        oof_blended, preds_blended, weights = blend_optimised(
            oof_list, pred_list, y, exp_log
        )
        for name, w in zip(models, weights):
            exp_log.info(f"  {name:<30} weight={w:.4f}")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    oof_blended   = renorm(oof_blended)
    preds_blended = renorm(preds_blended)
    ll, bacc = compute_oof_metrics(oof_blended, y, exp_log, label=ensemble_name)

    # Save config snapshot
    snapshot = {
        "ensemble_name":  ensemble_name,
        "ensemble_notes": notes,
        "strategy":       strategy,
        "models":         models,
        "weights":        weights.tolist() if weights is not None else None,
        "created":        datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        "global":         global_cfg.model_dump(),
        "source_metas":   {name: metas[name] for name in models},
    }
    (exp_dir / "config.json").write_text(json.dumps(snapshot, indent=2))

    # Arrays
    np.save(exp_dir / f"{ensemble_name}_oof.npy",   oof_blended)
    np.save(exp_dir / f"{ensemble_name}_preds.npy", preds_blended)

    # Meta sidecar
    meta = {
        "fname":      ensemble_name,
        "strategy":   strategy,
        "models":     models,
        "created":    datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        "cv_auc":     float(bacc),
        "cv_logloss": float(ll),
        "weights":    weights.tolist() if weights is not None else None,
    }
    (exp_dir / f"{ensemble_name}_meta.json").write_text(json.dumps(meta, indent=2))

    # Submission CSV
    target   = features.target
    df_sub   = pd.read_csv(comp_cfg.sample_sub_path(RAW_DIR))
    pred_ord = np.argmax(preds_blended, axis=1)
    df_sub[target] = pd.Categorical.from_codes(codes=pred_ord, categories=categories)

    if label_flag:
        df_sub[target] = df_sub[target].astype(str)
    else:
        df_sub[target] = df_sub[target].cat.codes.astype("int8")

    df_sub.to_csv(SUB_DIR / f"{ensemble_name}.csv", index=False)
    df_sub.to_csv(exp_dir / f"{ensemble_name}.csv", index=False)

    # Integrity checks
    df_check = pd.read_csv(SUB_DIR / f"{ensemble_name}.csv")
    assert df_check.shape[1] == 2,               f"Bad column count: {df_check.shape[1]}"
    assert df_check.shape[0] == comp_cfg.n_test, f"Bad row count: {df_check.shape[0]}"

    exp_log.info(f"[save] {SUB_DIR / ensemble_name}.csv  ({df_check.shape[0]} rows)")

    # run_log
    run_record = {
        "date":              datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exp_name":          ensemble_name,
        "exp_notes":         notes,
        "train_file":        "ensemble",
        "n_folds":           "—",
        "mean_cv_metric":    round(float(bacc), 5),
        "std_cv_metric":     0.0,
        "mean_logloss":      round(float(ll), 5),
        "n_features":        "—",
        "ensemble_models":   json.dumps(models),
        "ensemble_strategy": strategy,
    }
    log_run(ART_DIR, run_record)

    return {
        "name":     ensemble_name,
        "status":   "ok",
        "strategy": strategy,
        "models":   " + ".join(models),
        "logloss":  round(ll, 5),
        "bacc":     round(bacc, 5),
        "notes":    notes,
    }


def main() -> None:
    
    results = []
    for ensemble_name, cfg in ENSEMBLES.items():
        pipe_log.info(f"[run_ensemble] starting {ensemble_name}")
        try:
            row = run_single(ensemble_name, cfg, pipe_log)
        except Exception as exc:
            pipe_log.error(f"[run_ensemble] {ensemble_name} FAILED: {exc}")
            row = {"name": ensemble_name, "status": "FAILED", "logloss": None, "bacc": None}
        results.append(row)

    # Summary table
    df_results = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("ENSEMBLE SUMMARY")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)

    out_path = ART_DIR / "ensemble_results.csv"
    df_results.to_csv(out_path, index=False)
    pipe_log.info(f"[run_ensemble] summary → {out_path}")


if __name__ == "__main__":
    main()
