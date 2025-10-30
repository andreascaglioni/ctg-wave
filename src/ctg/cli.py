from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import typer
import numpy as np
import logging

from ctg.ctg_solver import CTGSolver
from ctg.brownian_motion import param_LC_W
from ctg.config import AppConfig, load_config
from ctg.save_metadata import save_run_metadata

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@app.command()
def run(
    data_file: Path = typer.Argument(Path("data/data_swe.yaml"), help="Path to YAML data file")
):
    """Run CTG wave equation solver with configuration from YAML data file. If no data file is given, use `data/data_swe.yaml`."""

    cfg: AppConfig = load_config(data_file)
    v = cfg.numerics.verbose

    # Configure logging level based on verbose setting
    logging.basicConfig(level=logging.INFO if v else logging.WARNING, format="%(message)s")

    logger.info(f"Configuration loaded from YAML data file: {data_file}")
    for key, value in cfg.__dict__.items():
        logger.info(f"{key}:")
        for subkey, subvalue in value.__dict__.items():
            logger.info(f"    {subkey}: {subvalue}")

    np.random.seed(cfg.numerics.seed)
    y = np.random.standard_normal(100)

    def W_t(tt):
        return param_LC_W(y, tt, T=cfg.physics.end_time)[0]

    ctg_solver = CTGSolver(cfg.numerics)
    sol_slabs, time_slabs, space_time_fe, total_n_dofs = ctg_solver.run(cfg.physics, W_t)

    logger.info(f"Time slabs (print max 5): {time_slabs[:min(len(time_slabs), 5)]}")
    logger.info(f"Solution computed on {len(sol_slabs)} time slabs")
    logger.info(f"Total degrees of freedom: {total_n_dofs}")

    # Figure result at final time
    n_scalar = space_time_fe.n_dofs
    n_x = space_time_fe.space_fe.n_dofs
    U_dofs = sol_slabs[-1][n_scalar : n_scalar + n_x]
    V_dofs = sol_slabs[-1][n_scalar + n_x :]
    plt.figure()
    plt.title("DOFs solutiojn (U and V) final time")
    plt.plot(space_time_fe.space_fe.dofs, U_dofs, ".-", label="U")
    plt.plot(space_time_fe.space_fe.dofs, V_dofs, ".-", label="V")
    plt.legend()

    # Save and log results
    date_time = datetime.now().isoformat(timespec="seconds")
    results_dir = Path("results/run_ctg_" + date_time)
    save_run_metadata(
        config_path=data_file, dir_prefix="run_ctg", extra={"total_n_dofs": total_n_dofs}
    )

    # Save solution data (do not save space_time_fe object because relative only to last slab)
    np.savez(results_dir / "solution_data.npz", sol_slabs=sol_slabs, time_slabs=time_slabs)

    plt.savefig(results_dir / "solution_T.png")

    logger.info(f"Solution data saved to {results_dir}")

    plt.show()


if __name__ == "__main__":
    app()
