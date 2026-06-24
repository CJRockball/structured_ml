import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

import subprocess



def get_device():
    try:
        subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL)
        return 'cuda'
    except Exception:
        return 'cpu'
    
def plot_shap_bar(sv: shap.Explanation, exp_dir: Path, max_display: int = 30) -> None:
    shap.plots.bar(sv, max_display=max_display, show=False)
    plt.gcf().set_size_inches(12, max_display * 0.4 + 2)   # scale height to n features
    plt.savefig(exp_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] saved bar plot → {exp_dir / 'shap_bar.png'}")


def plot_shap_beeswarm(sv: shap.Explanation, exp_dir: Path, max_display: int = 30) -> None:
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    plt.gcf().set_size_inches(12, max_display * 0.4 + 2)   # scale height to n features
    plt.savefig(exp_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] saved beeswarm → {exp_dir / 'shap_beeswarm.png'}")


def plot_xgb_importance(
    model,
    feat_cols: list[str],
    exp_dir:   Path,
    max_display: int = 30,
) -> pd.DataFrame:

    # ── Build DataFrame with real feature names ───────────────────────────────
    scores = model.get_booster().get_score(importance_type="total_gain")

    if not scores:
        print("[importance] no scores returned")
        return pd.DataFrame()

    # get_score keys are f0, f1... — map back to real names via position
    df_imp = (
        pd.DataFrame({
            "feature":    feat_cols,
            "importance": [scores.get(f"f{i}", 0.0) for i in range(len(feat_cols))]
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Save full importance table — useful for later analysis
    df_imp.to_parquet(exp_dir / "feature_importance.parquet")
    print(f"[importance] table saved → {exp_dir / 'feature_importance.parquet'}")

    # ── Plot top N ────────────────────────────────────────────────────────────
    df_plot = df_imp.head(max_display).iloc[::-1]   # reverse for barh (top = highest)

    fig, ax = plt.subplots(figsize=(10, max_display * 0.4 + 2))
    ax.barh(df_plot["feature"], df_plot["importance"], color="steelblue")
    ax.set_xlabel("Total Gain")
    ax.set_title("XGBoost Feature Importance (total_gain)")
    plt.tight_layout()
    plt.savefig(exp_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[importance] plot saved → {exp_dir / 'feature_importance.png'}")

    return df_imp