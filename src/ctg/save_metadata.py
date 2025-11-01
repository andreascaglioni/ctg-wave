from pathlib import Path
import json
import subprocess
from datetime import datetime
import yaml
from typing import Union


def save_run_metadata(
    config_source: Union[Path, str], results_dir: Path, extra: dict | None = None
):
    """Save reproducibility metadata for a run.

    Creates a timestamped results directory and saves configuration files and git metadata
    for full reproducibility of the experiment/run.

    Args:
        config_source: Either a Path to the configuration YAML file or a string containing
            the YAML configuration content.
        results_dir: Directory where the results are saved. It is created in the project root.
        extra: Optional dictionary of additional metadata to include in meta.json.

    Returns:
        None

    Raises:
        None: Exceptions during config file copying are caught and logged as warnings.

    Note:
        - Creates directory: results/{dir_prefix}_{timestamp}/
        - Copies config to results_dir/config_used.yaml
        - Copies companion {config_path}_functions.py if it exists
        - Saves git info (commit, branch, dirty status) to results_dir/meta.json
        - Git information defaults to "unknown" if git commands fail
    """

    results_dir.mkdir(parents=True, exist_ok=True)

    # Get YAML text from source (either read from Path or use string directly)
    if isinstance(config_source, Path):
        text_config = config_source.read_text(encoding="utf-8")
    else:
        text_config = config_source

    # Copy the YAML file for full reproducibility
    config_copy_path = results_dir / "config_used.yaml"
    try:
        config_copy_path.write_text(text_config)
    except Exception as e:
        print(f"Warning: could not copy config file: {e}")

    # TODO Copy the file with Callables used in yaml config file

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
        "config_copy": str(config_copy_path),
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }
    if extra:
        meta.update(extra)

    with open(results_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def get_config_functions_file_path(config_text):
    # Get name of file (py) containing callabels uised by config yaml (input path)

    # 1. Get "."-separated file name from config_path
    config_data_dict = yaml.safe_load(config_text)
    boundary_D_callable = config_data_dict.get("physics").get("boundary_D")
    config_path_funs = boundary_D_callable.split(":")[0]
    # 2. Convert dots to slashes to create a path
    return Path(config_path_funs.replace(".", "/"))
