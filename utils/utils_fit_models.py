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


def plot_xgb_importance(model, exp_dir: Path, max_display: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(12, max_display * 0.4 + 2))  # pre-create is fine here
    xgb.plot_importance(model, ax=ax, max_num_features=max_display, importance_type="total_gain")
    plt.tight_layout()
    plt.savefig(exp_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[importance] saved → {exp_dir / 'feature_importance.png'}")