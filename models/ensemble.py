import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
import datetime
import argparse
import logging

from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import log_loss, balanced_accuracy_score

from utils.utils_basic import log_run
from config import (
    GlobalConfig, global_cfg,
    FeatureConfig, features,
    CompetitionConfig, comp_cfg,
    RAW_DIR, PROC_DIR, SUB_DIR, ART_DIR,
    pipeline_logger as pipe_log,
)
from config.env_cfg import make_exp_logger

pipe_log.info("[ensemble] stage started")


# ── Config ────────────────────────────────────────────────────────────────────

STRATEGY_CHOICES = ["equal_avg", "rank_avg", "opt_weights"]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble: combine OOF/pred .npy artifacts into a submission."
    )
    parser.add_argument(
        "--exp-names",
        nargs="+",
        required=True,
        help="Space-separated list of experiment folder names under ART_DIR. "
             "E.g. --exp-names xgb1_base lgb1_base cat1_base",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default="opt_weights",
        help="Blending strategy (default: opt_weights).",
    )
    parser.add_argument(
        "--ensemble-name",
        type=str,
        default="ensemble_v1",
        dest="ensemble_name",
        help="Name for the output experiment folder and submission CSV.",
    )
    parser.add_argument(
        "--ensemble-notes",
        type=str,
        default="",
        dest="ensemble_notes",
        help="Free-text notes saved to config.json.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Train parquet filename in PROC_DIR (needed to load ground-truth labels). "
             "Auto-detected from first experiment's meta.json if omitted.",
    )
    parser.add_argument(
        "--label-flag",
        action="store_true",
        default=True,
        help="Decode predictions to class label strings (default: True).",
    )
    return parser.parse_args()


# ── Artifact discovery ────────────────────────────────────────────────────────

def load_artifacts(
    exp_names: list[str],
    art_dir: Path,
    log: logging.Logger,
) -> tuple[dict, dict, dict]:
    """
    Returns:
        oofs   : {exp_name: np.ndarray (n_train, n_classes)}
        preds  : {exp_name: np.ndarray (n_test,  n_classes)}
        metas  : {exp_name: dict}
    """
    oofs, preds, metas = {}, {}, {}
    for name in exp_names:
        exp_dir = art_dir / name
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")

        oof_files  = sorted(exp_dir.glob("*_oof.npy"))
        pred_files = sorted(exp_dir.glob("*_preds.npy"))
        meta_files = sorted(exp_dir.glob("*_meta.json"))

        if not oof_files:
            raise FileNotFoundError(f"No *_oof.npy in {exp_dir}")
        if not pred_files:
            raise FileNotFoundError(f"No *_preds.npy in {exp_dir}")
        if not meta_files:
            raise FileNotFoundError(f"No *_meta.json in {exp_dir}")

        oofs[name]  = np.load(oof_files[0])
        preds[name] = np.load(pred_files[0])
        metas[name] = json.loads(meta_files[0].read_text())
        log.info(f"[load] {name:<30} oof={oofs[name].shape} preds={preds[name].shape}")

    # Shape consistency
    ref_oof_shape  = next(iter(oofs.values())).shape
    ref_pred_shape = next(iter(preds.values())).shape
    for name in exp_names:
        assert oofs[name].shape  == ref_oof_shape,  f"{name}: OOF shape {oofs[name].shape} != {ref_oof_shape}"
        assert preds[name].shape == ref_pred_shape,  f"{name}: pred shape {preds[name].shape} != {ref_pred_shape}"

    log.info(f"[load] all shapes consistent ✓  oof={ref_oof_shape}  preds={ref_pred_shape}")
    return oofs, preds, metas


# ── Load ground-truth labels ──────────────────────────────────────────────────

def load_labels(
    train_file: str | None,
    metas: dict,
    proc_dir: Path,
    target: str,
    log: logging.Logger,
) -> tuple[np.ndarray, list]:
    if train_file is None:
        train_file = next(iter(metas.values())).get("train")
        if not train_file:
            raise ValueError("Cannot auto-detect train_file from meta.json. Pass --train-file explicitly.")
        log.info(f"[labels] auto-detected train_file: {train_file}")

    df = pd.read_parquet(proc_dir / train_file)
    y  = df[target].cat.codes.values
    categories = df[target].cat.categories.tolist()
    log.info(f"[labels] {len(y)} rows, classes={categories}")
    return y, categories

# ── Normalisation utility ─────────────────────────────────────────────────────

def renorm(arr: np.ndarray) -> np.ndarray:
    """Row-wise L1 normalise so every row sums to exactly 1.0.
    Fixes float32/float64 accumulation drift after weighted sums."""
    row_sums = arr.sum(axis=1, keepdims=True)
    return arr / row_sums

# ── Blending strategies ───────────────────────────────────────────────────────

def blend_equal(arrays: list[np.ndarray]) -> np.ndarray:
    return renorm(np.mean(arrays, axis=0))


def blend_rank(arrays: list[np.ndarray]) -> np.ndarray:
    n = arrays[0].shape[0]
    ranked = []
    for arr in arrays:
        m = arr.copy()
        for c in range(m.shape[1]):
            m[:, c] = rankdata(m[:, c]) / n
        ranked.append(m)
    blended = np.mean(ranked, axis=0)
    # Renormalise rows to sum to 1
    blended = blended / blended.sum(axis=1, keepdims=True)
    return blended


def blend_optimised(
    oofs: list[np.ndarray],
    preds: list[np.ndarray],
    y: np.ndarray,
    log: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimise blending weights on OOF log-loss via Nelder-Mead.
    Weights are softmax-parameterised → always positive, sum to 1.
    Returns (blended_oof, blended_preds, weights).
    """
    n = len(oofs)

    def neg_ll(w_raw):
        w = np.exp(w_raw) / np.exp(w_raw).sum()
        blended = renorm(np.tensordot(w, np.stack(oofs), axes=([0], [0])))
        return log_loss(y, blended)

    result = minimize(
        neg_ll,
        x0=np.zeros(n),
        method="Nelder-Mead",
        options={"maxiter": 10_000, "xatol": 1e-7, "fatol": 1e-7},
    )
    w_opt = np.exp(result.x) / np.exp(result.x).sum()
    log.info(f"[opt_weights] converged={result.success}  logloss={result.fun:.6f}")
    for name, w in zip([f"model_{i}" for i in range(n)], w_opt):
        log.info(f"  weight {name}: {w:.4f}")

    blended_oof   = renorm(np.tensordot(w_opt, np.stack(oofs),  axes=([0], [0])))
    blended_preds = renorm(np.tensordot(w_opt, np.stack(preds), axes=([0], [0])))
    return blended_oof, blended_preds, w_opt


# ── OOF metrics ───────────────────────────────────────────────────────────────

def compute_oof_metrics(
    oof: np.ndarray,
    y: np.ndarray,
    log: logging.Logger,
    label: str = "",
) -> tuple[float, float]:
    ll   = log_loss(y, oof)
    bacc = balanced_accuracy_score(y, np.argmax(oof, axis=1))
    log.info(f"[oof_metrics] {label:<30} logloss={ll:.5f}  balanced_acc={bacc:.5f}")
    return ll, bacc


# ── Save ──────────────────────────────────────────────────────────────────────

def save_ensemble(
    ensemble_name: str,
    ensemble_notes: str,
    strategy: str,
    exp_names: list[str],
    weights: np.ndarray | None,
    oof_blended: np.ndarray,
    preds_blended: np.ndarray,
    y: np.ndarray,
    categories: list,
    metas: dict,
    label_flag: bool,
    log: logging.Logger,
) -> None:
    exp_dir = ART_DIR / ensemble_name
    if exp_dir.exists():
        raise FileExistsError(
            f"Ensemble folder already exists: {exp_dir}\n"
            f"Rename --ensemble-name or delete the folder first."
        )
    exp_dir.mkdir(parents=True)

    ll, bacc = compute_oof_metrics(oof_blended, y, log, label=ensemble_name)

    # ── Config snapshot ───────────────────────────────────────────────────────
    snapshot = {
        "ensemble_name":  ensemble_name,
        "ensemble_notes": ensemble_notes,
        "strategy":       strategy,
        "models":         exp_names,
        "weights":        weights.tolist() if weights is not None else None,
        "created":        datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        "global":         global_cfg.model_dump(),
        "source_metas":   {name: metas[name] for name in exp_names},
    }
    (exp_dir / "config.json").write_text(json.dumps(snapshot, indent=2))
    log.info(f"[save] config → {exp_dir / 'config.json'}")

    # ── OOF / pred arrays ─────────────────────────────────────────────────────
    np.save(exp_dir / f"{ensemble_name}_oof.npy",   oof_blended)
    np.save(exp_dir / f"{ensemble_name}_preds.npy", preds_blended)

    # ── Meta sidecar ──────────────────────────────────────────────────────────
    meta = {
        "fname":      ensemble_name,
        "strategy":   strategy,
        "models":     exp_names,
        "created":    datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        "cv_auc":     bacc,
        "cv_logloss": ll,
        "weights":    weights.tolist() if weights is not None else None,
    }
    (exp_dir / f"{ensemble_name}_meta.json").write_text(json.dumps(meta, indent=2))

    # ── Submission CSV ────────────────────────────────────────────────────────
    target      = features.target
    df_sub      = pd.read_csv(comp_cfg.sample_sub_path(RAW_DIR))
    pred_ord    = np.argmax(preds_blended, axis=1)
    df_sub[target] = pd.Categorical.from_codes(codes=pred_ord, categories=categories)

    if label_flag:
        df_sub[target] = df_sub[target].astype(str)
    else:
        df_sub[target] = df_sub[target].cat.codes.astype("int8")

    df_sub.to_csv(SUB_DIR  / f"{ensemble_name}.csv", index=False)
    df_sub.to_csv(exp_dir  / f"{ensemble_name}.csv", index=False)

    # Integrity checks (mirror xgb1.py / cat1.py)
    df_check = pd.read_csv(SUB_DIR / f"{ensemble_name}.csv")
    assert df_check.shape[1] == 2,                f"Submission has {df_check.shape[1]} columns, expected 2"
    assert df_check.shape[0] == comp_cfg.n_test,  f"Submission has {df_check.shape[0]} rows, expected {comp_cfg.n_test}"

    log.info(f"[save] submission → {SUB_DIR / ensemble_name}.csv")

    # ── run_log.csv ───────────────────────────────────────────────────────────
    run_record = {
        "date":             datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exp_name":         ensemble_name,
        "exp_notes":        ensemble_notes,
        "train_file":       "ensemble",
        "n_folds":          "—",
        "mean_cv_metric":   round(bacc, 5),
        "std_cv_metric":    0.0,
        "mean_logloss":     round(ll, 5),
        "n_features":       "—",
        "ensemble_models":  json.dumps(exp_names),
        "ensemble_strategy": strategy,
    }
    log_run(ART_DIR, run_record)
    pipe_log.info(f"[ensemble] run logged → {ART_DIR / 'run_log.csv'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    exp_dir = ART_DIR / args.ensemble_name
    # Create dir early so we can attach the logger; guard against collision
    if exp_dir.exists():
        raise FileExistsError(
            f"Output folder already exists: {exp_dir}\n"
            "Use --ensemble-name to choose a different name."
        )
    exp_dir.mkdir(parents=True)
    log = make_exp_logger(exp_dir)
    log.info(f"Ensemble: {args.ensemble_name}")
    log.info(f"Strategy: {args.strategy}")
    log.info(f"Models:   {args.exp_names}")

    # 1. Load artifacts
    oofs, preds, metas = load_artifacts(args.exp_names, ART_DIR, log)

    # 2. Ground-truth labels
    y, categories = load_labels(
        args.train_file, metas, PROC_DIR, features.target, log
    )

    # Log individual model metrics for reference
    log.info("[oof_metrics] individual model baselines:")
    for name in args.exp_names:
        compute_oof_metrics(oofs[name], y, log, label=name)

    oof_list  = [oofs[n]  for n in args.exp_names]
    pred_list = [preds[n] for n in args.exp_names]

    # 3. Blend
    weights = None
    if args.strategy == "equal_avg":
        oof_blended   = blend_equal(oof_list)
        preds_blended = blend_equal(pred_list)

    elif args.strategy == "rank_avg":
        oof_blended   = blend_rank(oof_list)
        preds_blended = blend_rank(pred_list)

    elif args.strategy == "opt_weights":
        oof_blended, preds_blended, weights = blend_optimised(
            oof_list, pred_list, y, log
        )
        # Attach human-readable weight map to log
        for name, w in zip(args.exp_names, weights):
            log.info(f"  {name:<30} weight={w:.4f}")

    # 4. Save everything
    # exp_dir already created — save_ensemble skips mkdir but still checks existence
    # We created it above for the logger; pass a flag or just recreate with exist_ok
    # Re-use the same dir by temporarily removing the collision guard in save_ensemble
    # (simplest: just call the save logic directly without the mkdir guard)
    ll, bacc = compute_oof_metrics(oof_blended, y, log, label=f"{args.ensemble_name} ({args.strategy})")

    snapshot = {
        "ensemble_name":  args.ensemble_name,
        "ensemble_notes": args.ensemble_notes,
        "strategy":       args.strategy,
        "models":         args.exp_names,
        "weights":        weights.tolist() if weights is not None else None,
        "created":        datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        "global":         global_cfg.model_dump(),
        "source_metas":   {name: metas[name] for name in args.exp_names},
    }
    (exp_dir / "config.json").write_text(json.dumps(snapshot, indent=2))

    np.save(exp_dir / f"{args.ensemble_name}_oof.npy",   oof_blended)
    np.save(exp_dir / f"{args.ensemble_name}_preds.npy", preds_blended)

    meta = {
        "fname":      args.ensemble_name,
        "strategy":   args.strategy,
        "models":     args.exp_names,
        "created":    datetime.datetime.now().strftime("%Y%m%d_%H%M"),
        "cv_auc":     float(bacc),
        "cv_logloss": float(ll),
        "weights":    weights.tolist() if weights is not None else None,
    }
    (exp_dir / f"{args.ensemble_name}_meta.json").write_text(json.dumps(meta, indent=2))

    target   = features.target
    df_sub   = pd.read_csv(comp_cfg.sample_sub_path(RAW_DIR))
    pred_ord = np.argmax(preds_blended, axis=1)
    df_sub[target] = pd.Categorical.from_codes(codes=pred_ord, categories=categories)

    if args.label_flag:
        df_sub[target] = df_sub[target].astype(str)
    else:
        df_sub[target] = df_sub[target].cat.codes.astype("int8")

    df_sub.to_csv(SUB_DIR / f"{args.ensemble_name}.csv", index=False)
    df_sub.to_csv(exp_dir / f"{args.ensemble_name}.csv", index=False)

    df_check = pd.read_csv(SUB_DIR / f"{args.ensemble_name}.csv")
    assert df_check.shape[1] == 2,               f"Bad column count: {df_check.shape[1]}"
    assert df_check.shape[0] == comp_cfg.n_test, f"Bad row count: {df_check.shape[0]}"

    log.info(f"[save] submission → {SUB_DIR / args.ensemble_name}.csv  ({df_check.shape[0]} rows)")

    run_record = {
        "date":              datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "exp_name":          args.ensemble_name,
        "exp_notes":         args.ensemble_notes,
        "train_file":        "ensemble",
        "n_folds":           "—",
        "mean_cv_metric":    round(float(bacc), 5),
        "std_cv_metric":     0.0,
        "mean_logloss":      round(float(ll), 5),
        "n_features":        "—",
        "ensemble_models":   json.dumps(args.exp_names),
        "ensemble_strategy": args.strategy,
    }
    log_run(ART_DIR, run_record)
    pipe_log.info(f"[ensemble] done → {args.ensemble_name}")


if __name__ == "__main__":
    main()
