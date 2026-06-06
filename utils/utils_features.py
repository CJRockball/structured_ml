# utils/sidecar.py

import json
import datetime
import subprocess
import pandas as pd
from pathlib import Path
import hashlib


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def get_git_message() -> str:
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "no-git"


def get_file_hash(filepath: str | Path, length: int = 8) -> str:
    """Hash the contents of a source file. 
    Equivalent to a git hash for non-git workflows."""
    try:
        content = Path(filepath).read_bytes()
        return hashlib.sha256(content).hexdigest()[:length]
    except Exception:
        return "no-file"


def get_data_hash(path: Path, length: int = 8) -> str:
    """Hash the contents of the saved parquet file."""
    try:
        content = Path(path).read_bytes()
        return hashlib.sha256(content).hexdigest()[:length]
    except Exception:
        return "no-data-hash"

def get_code_version(filepath: str | Path) -> dict:
    """
    Returns code provenance that works in both git and non-git contexts.
    Git hash takes priority if available, falls back to file hash.
    """
    git_hash = get_git_hash()
    git_msg  = get_git_message()
    
    if git_hash != "no-git":
        return {
            "source":      "git",
            "hash":        git_hash,
            "message":     git_msg,
            "file_hash":   get_file_hash(filepath),  # belt and suspenders
        }
    else:
        return {
            "source":      "file",
            "hash":        get_file_hash(filepath),
            "message":     "no-git",
            "file_hash":   get_file_hash(filepath),
        }

# utils.py
def save_dataset(
    df: pd.DataFrame,
    path: Path | str,
    slug: str,
    target: list[str],
    source_file: str | Path | None = None,
    parent: str | None = None,
    stages: list[str] | None = None,
    notes: str = "",
) -> Path:
    """
    Save a DataFrame to parquet and write a companion .meta.json sidecar.

    Args:
        df:          DataFrame to save
        path:        Full output path for the parquet file
        slug:        Short human-readable name for this dataset version
        target:      List of target column names — excluded from feature groups
        source_file: Path to the script that produced this file (__file__)
        parent:      Filename of the parquet this was derived from (None if from raw)
        stages:      List of pipeline stage names applied to produce this file
        notes:       Any free-text note you want to attach

    Returns:
        Path to the saved parquet file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save parquet
    df.to_parquet(path, index=False)

    # Code provenance — git if available, file hash otherwise
    code_version = get_code_version(source_file) if source_file else {
        "source":    "unknown",
        "hash":      "unknown",
        "message":   "no source file provided",
        "file_hash": "unknown",
    }
    
    # Data hash — unique per output file
    data_hash = get_data_hash(path)    

    # Schema introspection
    # Resolve target first so it is excluded from feature groups
    target_cols = [c for c in target if c in df.columns]
    cat_cols    = df.select_dtypes("category").columns.tolist()
    str_cols    = df.select_dtypes("object").columns.tolist()
    num_cols    = [c for c in df.select_dtypes("floating").columns
                   if c not in target_cols]
    int_cols    = [c for c in df.select_dtypes("integer").columns
                   if c not in target_cols]

    # Safety net — catch any column with an unexpected dtype
    classified   = cat_cols + str_cols + num_cols + int_cols + target_cols
    unclassified = [c for c in df.columns if c not in classified]
    assert len(unclassified) == 0, (
        f"save_dataset: unclassified columns with unexpected dtypes: "
        f"{ {c: str(df[c].dtype) for c in unclassified} }"
    )

    # Per-column stats for numeric cols — useful for drift detection later
    num_stats = {}
    for col in num_cols:
        num_stats[col] = {
            "mean":   round(float(df[col].mean()), 6),
            "std":    round(float(df[col].std()),  6),
            "min":    round(float(df[col].min()),  6),
            "max":    round(float(df[col].max()),  6),
            "n_null": int(df[col].isna().sum()),
        }

    # Cardinality and null count for categorical cols
    cat_stats = {}
    for col in cat_cols + str_cols:
        cat_stats[col] = {
            "n_unique": int(df[col].nunique()),
            "n_null":   int(df[col].isna().sum()),
        }

    # Cardinality for integer-encoded cols (int16 codes etc.)
    int_stats = {}
    for col in int_cols:
        int_stats[col] = {
            "n_unique": int(df[col].nunique()),
            "min":      int(df[col].min()),
            "max":      int(df[col].max()),
            "n_null":   int(df[col].isna().sum()),
        }

    meta = {
        "slug":         slug,
        "created":      datetime.datetime.now().isoformat(timespec="seconds"),
        "code_version": code_version,
        "data_hash":    data_hash,       # from get_data_hash — parquet hash
        "parent":       parent,
        "stages":       stages or [],
        "notes":        notes,
        "rows":         len(df),
        "columns":      df.columns.tolist(),
        "cat_cols":     cat_cols,
        "str_cols":     str_cols,
        "num_cols":     num_cols,
        "int_cols":     int_cols,
        "target_cols":  target_cols,
        "dtypes":       {c: str(t) for c, t in df.dtypes.items()},
        "num_stats":    num_stats,
        "cat_stats":    cat_stats,
        "int_stats":    int_stats,
    }

    meta_path = path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return path

def load_meta(path: Path | str) -> dict:
    """Load the sidecar meta for a given parquet path."""
    return json.loads(Path(path).with_suffix(".meta.json").read_text())


def print_lineage(path: Path | str, indent: int = 0) -> None:
    """Recursively print the full lineage chain for a dataset."""
    try:
        meta = load_meta(path)
    except FileNotFoundError:
        print(" " * (indent * 2) + f"[no meta found for {Path(path).name}]")
        return

    prefix  = "  " * indent
    arrow   = "→ " if indent == 0 else "↳ "

    # Code provenance — handles both git and file-hash contexts
    cv      = meta.get("code_version", {})
    source  = cv.get("source", "unknown")
    hash_   = cv.get("hash", "unknown")
    message = cv.get("message", "")

    if source == "git":
        version_str = f"git:{hash_} — {message}"
    elif source == "file":
        version_str = f"file-hash:{hash_}"
    else:
        version_str = "unknown provenance"

    # Column summary — distinguish str (pre-encode) from category (post-encode)
    cat_cols = meta.get("cat_cols", [])
    str_cols = meta.get("str_cols", [])
    num_cols = meta.get("num_cols", [])
    int_cols    = meta.get("int_cols", [])
    target   = meta.get("target_cols", [])
    

    cat_summary = f"category({len(cat_cols)})" if cat_cols else ""
    str_summary = f"str({len(str_cols)})"      if str_cols else ""
    num_summary = f"floats({len(num_cols)})"
    int_summary = f"ints({len(int_cols)})" if int_cols else ""
    col_summary = ", ".join(filter(None, [cat_summary, str_summary, 
                                        int_summary, num_summary]))

    print(f"{prefix}{arrow}{meta['slug']}")
    print(f"{prefix}  file:     {Path(path).name}")
    print(f"{prefix}  date:     {meta['created']}")
    print(f"{prefix}  source:   {version_str}")
    print(f"{prefix}  data:     {meta['data_hash']}")
    print(f"{prefix}  rows:     {meta['rows']}")
    print(f"{prefix}  cols:     {col_summary} | target: {target}")
    print(f"{prefix}  stages:   {meta['stages']}")

    if meta.get("notes"):
        print(f"{prefix}  notes:    {meta['notes']}")

    if meta.get("parent"):
        parent_path = Path(path).parent / meta["parent"]
        print(f"{prefix}  {'─' * 40}")
        print_lineage(parent_path, indent + 1)
