from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import typer
import numpy as np
import logging
from importlib.resources import files

from ctg.ctg_solver import CTGSolver
from ctg.brownian_motion import param_LC_W
from ctg.config import AppConfig, load_config
from ctg.save_metadata import save_run_metadata

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@app.command()
def run(
    config_path: Path = typer.Option(
        None, help="Path to YAML data file. Use a default file if none provided"
    )
):
    """Run CTG wave equation solver with configuration from YAML data file. If no data file is given, use default."""

    # Initialize configuration class
    text_yaml = get_yaml_text(config_path)
    cfg: AppConfig = load_config(text_yaml)

    # Set the logging verbosity from config
    set_log_verbosity(config_path, cfg)

    # Sample the Brownian motion
    np.random.seed(cfg.numerics.seed)
    y = np.random.standard_normal(100)

    def W_t(tt):
        return param_LC_W(y, tt, T=cfg.physics.end_time)[0]

    # Compute the wave
    ctg_solver = CTGSolver(cfg.numerics)
    sol_slabs, time_slabs, space_time_fe, total_n_dofs = ctg_solver.run(cfg.physics, W_t)

    # Log partial results
    logger.info("Computation completed successfully")
    logger.info(f"Number of time slabs: {len(time_slabs)}")
    logger.info(f"Spatial DOFs per time slab: {space_time_fe.space_fe.n_dofs}")
    logger.info(f"Temporal DOFs per time slab: {space_time_fe.time_fe.n_dofs}")
    logger.info(f"Space-time DOFs per slab: {space_time_fe.n_dofs}")
    logger.info(f"Total DOFs (all slabs): {total_n_dofs}")

    # Save results
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(cfg.post.dir_save + "_" + date_time)

    save_run_metadata(
        text_yaml,
        results_dir,
        extra={
            "n_dofs_space": space_time_fe.space_fe.n_dofs,
            "n_dofs_1_t_slab": space_time_fe.time_fe.n_dofs,
            "n_time_slabs": len(time_slabs),
            "n_dofs_xt_slab": space_time_fe.n_dofs,
            "total_n_dofs": total_n_dofs,
        },
    )

    # Save relevant solution data
    np.savez(results_dir / "solution_data.npz", sol_slabs=sol_slabs, time_slabs=time_slabs)

    # Plot solution at final time
    xt_fe = space_time_fe
    U_dofs = sol_slabs[-1][xt_fe.n_dofs : xt_fe.n_dofs + xt_fe.space_fe.n_dofs]
    V_dofs = sol_slabs[-1][xt_fe.n_dofs + xt_fe.space_fe.n_dofs :]
    plt.figure()
    plt.title("DOFs solutiojn (U and V) final time")
    plt.plot(space_time_fe.space_fe.dofs, U_dofs, ".-", label="U")
    plt.plot(space_time_fe.space_fe.dofs, V_dofs, ".-", label="V")
    plt.legend()
    plt.savefig(results_dir / "solution_T.png")

    logger.info(f"Solution data saved to {results_dir}")

    return 0


def get_yaml_text(config_path):
    # Load either user config or default
    if config_path is None:
        text_yaml = (files("ctg.default_data") / "data_swe.yaml").read_text(encoding="utf-8")
        logger.info("Using packaged default configuration.")
    else:
        try:
            text_yaml = config_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.info(f"Could not read configurations: {e}")
            raise typer.Exit(code=1)
    return text_yaml


def set_log_verbosity(config, cfg):
    v = cfg.numerics.verbose
    logging.basicConfig(level=logging.INFO if v else logging.WARNING, format="%(message)s")

    logger.info(f"Configuration loaded from YAML data file: {config}")
    for key, value in cfg.__dict__.items():
        logger.info(f"{key}:")
        for subkey, subvalue in value.__dict__.items():
            logger.info(f"    {subkey}: {subvalue}")


if __name__ == "__main__":
    app()
