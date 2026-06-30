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

# config/models_cfg.py  ── add after XGBConfig ─────────────────────────────


class LGBConfig(BaseModel):
    # Inherited from global
    seed:             int   = global_cfg.seed
    device:           str   = global_cfg.device          # "cpu" or "gpu"

    # Objective & eval
    objective:        str   = "multiclass"
    metric:           str   = "multi_logloss"
    num_class:        int   = 3                          # update for your target
    verbosity:        int   = 1 if global_cfg.verbose else -1

    # Tree structure
    num_leaves:       int   = 31
    max_depth:        int   = -1                         # -1 = unlimited (LGB default)
    min_child_samples: int  = 20                         # LGB name for min_child_weight
    min_child_weight: float = 1e-3

    # Boosting
    learning_rate:    float = 0.05
    n_estimators:     int   = 1000
    early_stopping_rounds: int = 50

    # Sampling
    subsample:        float = 0.8
    subsample_freq:   int   = 1                          # must be > 0 for subsample to work
    colsample_bytree: float = 0.8                        # alias: feature_fraction

    # Regularisation
    reg_alpha:        float = 0.0                        # L1
    reg_lambda:       float = 1.0                        # L2

    # Categoricals
    cat_smooth:       float = 10.0

    model_config = {"frozen": True}

    @field_validator("learning_rate", "subsample", "colsample_bytree")
    @classmethod
    def must_be_unit_interval(cls, v):
        assert 0 < v <= 1, f"Must be in (0, 1], got {v}"
        return v

    @field_validator("device")
    @classmethod
    def valid_device(cls, v):
        assert v in ("cpu", "gpu"), f"device must be 'cpu' or 'gpu', got {v}"
        return v

    def to_dict(self) -> dict:
        """LightGBM-compatible param dict (model.fit uses separate n_estimators)."""
        d = self.model_dump()
        # LGB uses 'feature_fraction' not 'colsample_bytree' in its native API;
        # lightgbm sklearn API accepts colsample_bytree directly — keep as-is.
        # 'n_estimators' and 'early_stopping_rounds' are sklearn-API args, not params.
        return d


# config/models_cfg.py  ── add after LGBConfig ───────────────────────────────

class CatConfig(BaseModel):
    # Inherited from global
    seed:               int   = global_cfg.seed
    # CatBoost uses "GPU" (uppercase), not "gpu"
    task_type:          str   = "GPU" if global_cfg.device == "gpu" else "CPU"

    # Objective & eval
    loss_function:      str   = "MultiClass"
    eval_metric:        str   = "MultiClass"        # log-loss for multiclass
    classes_count:      int   = 3                   # update for your target

    # Boosting rounds
    iterations:         int   = 4000               # CatBoost name for n_estimators
    learning_rate:      float = 0.05
    early_stopping_rounds: int = 50

    # Tree structure
    depth:              int   = 6                  # CatBoost name for max_depth (max 16)
    min_data_in_leaf:   int   = 20

    # Regularisation
    l2_leaf_reg:        float = 3.0               # L2 only; no L1 in CatBoost
    random_strength:    float = 1.0               # adds noise to splits (anti-overfit)
    bagging_temperature: float = 1.0              # Bayesian bootstrap intensity

    # Sampling (CatBoost Bayesian bootstrap — subsample only used with Bernoulli)
    bootstrap_type:     str   = "Bayesian"        # "Bayesian" | "Bernoulli" | "MVS"
    subsample:          float = 0.8               # only active when bootstrap_type="Bernoulli"

    # Verbosity
    verbose:            int   = 100               # print every N iterations (0 = silent)

    model_config = {"frozen": True}

    @field_validator("learning_rate")
    @classmethod
    def must_be_unit_interval(cls, v):
        assert 0 < v <= 1, f"Must be in (0, 1], got {v}"
        return v

    @field_validator("depth")
    @classmethod
    def max_depth_16(cls, v):
        assert 1 <= v <= 16, f"CatBoost depth must be 1–16, got {v}"
        return v

    @field_validator("task_type")
    @classmethod
    def valid_task_type(cls, v):
        assert v in ("CPU", "GPU"), f"task_type must be 'CPU' or 'GPU', got {v}"
        return v

    def to_dict(self) -> dict:
        return self.model_dump()



# ── Instantiate once ──────────────────────────────────────────────────────────
comp_cfg = CompetitionConfig()
data_cfg = DataConfig()
xgb_cfg = XGBConfig()
lgb_cfg = LGBConfig()
cat_cfg = CatConfig()
cv_cfg  = CVConfig()