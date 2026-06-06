from pathlib import Path
from dotenv import load_dotenv
from .global_cfg import global_cfg
import logging
import os

# ── This is the only line that matters for your problem ───────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")   # also anchor .env to project root

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR   = PROJECT_ROOT / Path(os.getenv("DATA_RAW_DIR",  "data/raw"))
PROC_DIR  = PROJECT_ROOT / Path(os.getenv("DATA_PROC_DIR", "data/processed"))
MODEL_DIR = PROJECT_ROOT / Path(os.getenv("MODEL_DIR",     "models"))
SUB_DIR   = PROJECT_ROOT / Path(os.getenv("SUB_DIR",       "submissions"))
LOG_DIR   = PROJECT_ROOT / Path(os.getenv("LOG_DIR",       "logs"))
ART_DIR   = PROJECT_ROOT / Path(os.getenv("ARTIFACTS_DIR", "artifacts"))


for _d in (PROC_DIR, MODEL_DIR, SUB_DIR, LOG_DIR, ART_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Shared format ─────────────────────────────────────────────────────────────
_fmt   = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)-12s  %(message)s",
    datefmt="%H:%M:%S",
)
_level = getattr(logging, global_cfg.log_level)


def _make_logger(name: str, logfile: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(_level)

    ch = logging.StreamHandler()          # console
    ch.setFormatter(_fmt)

    fh = logging.FileHandler(LOG_DIR / logfile)   # file
    fh.setFormatter(_fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False    # prevents double-logging to root logger
    return logger

def make_exp_logger(exp_dir: Path) -> logging.Logger:
    """
    Creates a per-experiment logger that writes to exp_dir/run.log.
    Called once per run after exp_dir is known.
    """
    logger = logging.getLogger(f"exp.{exp_dir.name}")
    logger.setLevel(_level)

    fh = logging.FileHandler(exp_dir / "run.log")
    fh.setFormatter(_fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(_fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger

# ── The 3 loggers — import these, never call logging.getLogger() elsewhere ────
pipeline_logger = _make_logger("pipeline", "pipeline.log")
feature_logger  = _make_logger("features", "features.log")
model_logger    = _make_logger("models",   "models.log")