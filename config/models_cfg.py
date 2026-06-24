from pathlib import Path
from pydantic import BaseModel, field_validator, model_validator
from .global_cfg import global_cfg


class CompetitionConfig(BaseModel):
    """
    Competition-level file layout.
    Raw paths are competition-specific, not environment-specific.
    """
    name:            str = "s6e6_star"
    raw_train_file:  str = "train.csv"
    raw_test_file:   str = "test.csv"
    sample_sub_file: str = "sample_submission.csv"
    n_test:          int = 247435
    
    model_config = {"frozen": True}

    @field_validator("raw_train_file", "raw_test_file", "sample_sub_file")
    @classmethod
    def must_be_csv(cls, v):
        assert v.endswith(".csv"), f"Expected .csv file, got: {v}"
        return v

    def train_path(self, raw_dir: Path) -> Path:
        return raw_dir / self.raw_train_file

    def test_path(self, raw_dir: Path) -> Path:
        return raw_dir / self.raw_test_file

    def sample_sub_path(self, raw_dir: Path) -> Path:
        return raw_dir / self.sample_sub_file
    

class DataConfig(BaseModel):
    """Which processed files this model experiment consumes."""
    train_file: str = "s6e6_star_train_base_20260611.parquet"
    test_file:  str = "s6e6_star_test_base_20260611.parquet"
    
    exp_name:    str = "xgb1_base"          # ← you set this per experiment
    exp_notes:   str = "baseline xgb run"   # ← optional human description

    model_config = {"frozen": True}

    @field_validator("train_file", "test_file")
    @classmethod
    def must_be_parquet(cls, v):
        assert v.endswith(".parquet"), f"Expected .parquet file, got: {v}"
        return v

    def train_path(self, proc_dir: Path) -> Path:
        return proc_dir / self.train_file

    def test_path(self, proc_dir: Path) -> Path:
        return proc_dir / self.test_file

class XGBConfig(BaseModel):
    # ── Inherited from global ─────────────────────────────────────────────────
    seed:    int  = global_cfg.seed
    device:  str  = global_cfg.device      # "cpu" or "cuda"

    # ── Tree structure ────────────────────────────────────────────────────────
    objective:        str = 'multi:softprob'
    tree_method:      str = "hist"
    max_depth:        int = 6
    min_child_weight: int = 1

    # ── Regularization ────────────────────────────────────────────────────────
    learning_rate:         float = 0.05
    subsample:             float = 1.0
    colsample_bytree:      float = 1.0
    colsample_bylevel:     float = 1.0
    reg_alpha:             float = 2.0    # L1
    reg_lambda:            float = 0.3   # L2

    # ── Boosting rounds ───────────────────────────────────────────────────────
    n_estimators:          int = 2000
    early_stopping_rounds: int = 100
    eval_metric:           str = 'mlogloss'

    # ── Verbosity ─────────────────────────────────────────────────────────────
    # verbosity: int = 100 if global_cfg.verbose else 0

    model_config = {"frozen": True}

    @field_validator("learning_rate", "subsample", "colsample_bytree", "colsample_bylevel")
    @classmethod
    def must_be_unit_interval(cls, v):
        assert 0 < v <= 1, f"Must be in (0, 1], got {v}"
        return v

    @field_validator("max_depth", "n_estimators", "early_stopping_rounds")
    @classmethod
    def must_be_positive(cls, v):
        assert v > 0, f"Must be positive, got {v}"
        return v

    @field_validator("device")
    @classmethod
    def valid_device(cls, v):
        assert v in ("cpu", "cuda"), f"device must be 'cpu' or 'cuda', got {v}"
        return v

    @model_validator(mode="after")
    def gpu_needs_hist(self):
        """XGBoost requires tree_method='hist' when using GPU."""
        if self.device == "cuda" and self.tree_method != "hist":
            raise ValueError("tree_method must be 'hist' when device='cuda'")
        return self

    def to_dict(self) -> dict:
        """Plain dict for xgb.XGBRegressor(**...) — excludes Pydantic internals."""
        return self.model_dump()


class CVConfig(BaseModel):
    n_folds:  int  = 5
    seed:     int  = global_cfg.seed    # same seed as model
    shuffle:  bool = True

    model_config = {"frozen": True}

    @field_validator("n_folds")
    @classmethod
    def min_folds(cls, v):
        assert v >= 2, f"n_folds must be >= 2, got {v}"
        return v


# ── Instantiate once ──────────────────────────────────────────────────────────
comp_cfg = CompetitionConfig()
data_cfg = DataConfig()
xgb_cfg = XGBConfig()
cv_cfg  = CVConfig()