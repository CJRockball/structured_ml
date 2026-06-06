# config_global.py
import torch
from pydantic import BaseModel, field_validator

class GlobalConfig(BaseModel):
    seed:          int  = 1337
    n_jobs:        int  = -1        # -1 = use all cores (sklearn, joblib)
    use_gpu:       bool = True
    log_level:     str  = "INFO"
    verbose:       bool = True

    model_config = {"frozen": True}


    @property
    def device(self) -> str:
        """Resolved at runtime — doesn't assume GPU is available."""
        if self.use_gpu and torch.cuda.is_available():
            return "cuda"
        return "cpu"


global_cfg = GlobalConfig()