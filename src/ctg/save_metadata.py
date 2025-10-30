from pathlib import Path
import json
import shutil
import subprocess
from datetime import datetime


def save_run_metadata(config_path: Path, dir_prefix: str, extra: dict | None = None):
    """Save reproducibility metadata (config file, commit hash, timestamp, etc.)."""
    date_time = datetime.now().isoformat(timespec="seconds")
    results_dir = Path("results") / (dir_prefix + "_" + date_time)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Copy the YAML file for full reproducibility
    config_path_funs = config_path.parent / (config_path.stem + "_functions.py")
    config_copy = results_dir / "config_used.yaml"
    config_copy_funs = results_dir / "config_used_functions.py"
    try:
        # copy configurations .yaml file
        shutil.copy(config_path, config_copy)
        # Also copy the functions file with name config_path+"_functions.py"
        if config_path_funs.exists():
            shutil.copy(config_path_funs, config_copy_funs)
    except Exception as e:
        print(f"Warning: could not copy config files: {e}")

    # Gather git info
    def git_info(cmd):
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except Exception:
            return "unknown"

    commit = git_info(["git", "rev-parse", "--short", "HEAD"])
    branch = git_info(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(subprocess.getoutput("git status --porcelain"))

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_original": str(config_path),
        "config_copy": str(config_copy),
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }
    if extra:
        meta.update(extra)

    with open(results_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
