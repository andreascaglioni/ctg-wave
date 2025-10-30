from pathlib import Path
from matplotlib import pyplot as plt
import typer
import numpy as np
import logging

from ctg.ctg_solver import CTGSolver
from ctg.brownian_motion import param_LC_W
from ctg.config import AppConfig, load_config

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@app.command()
def run(
    data_file: Path = typer.Argument(
        Path("data/data_swe.yaml"), help="Path to YAML data file (e.g., `data/data_swe.yaml`)"
    )
):
    """Run CTG wave equation solver with configuration from YAML data file. If no data file is given, use `data/data_swe.yaml`."""

    cfg: AppConfig = load_config(data_file)
    v = cfg.numerics.verbose

    # Configure logging level based on verbose setting
    logging.basicConfig(level=logging.INFO if v else logging.WARNING, format="%(message)s")

    logger.info(f"Configuration loaded from YAML data file: {data_file}")
    if v:
        for key, value in cfg.__dict__.items():
            print(f"{key}:")
            for subkey, subvalue in value.__dict__.items():
                print(f"    {subkey}: {subvalue}")

    np.random.seed(cfg.numerics.seed)
    y = np.random.standard_normal(100)

    def W_t(tt):
        return param_LC_W(y, tt, T=cfg.physics.end_time)[0]

    ctg_solver = CTGSolver(cfg.numerics)
    sol_slabs, time_slabs, space_time_fe, total_n_dofs = ctg_solver.run(cfg.physics, W_t)

    logger.info(f"Time slabs (print max 5): {time_slabs[:min(len(time_slabs), 5)]}")
    logger.info(f"Solution computed on {len(sol_slabs)} time slabs")
    logger.info(f"Total degrees of freedom: {total_n_dofs}")
    if v:
        n_scalar = space_time_fe.n_dofs
        n_x = space_time_fe.space_fe.n_dofs
        U_dofs = sol_slabs[-1][n_scalar : n_scalar + n_x]
        V_dofs = sol_slabs[-1][n_scalar + n_x :]

        plt.figure()
        plt.title("DOFs solutiojn (U and V) final time")
        plt.plot(space_time_fe.space_fe.dofs, U_dofs, ".-", label="U")
        plt.plot(space_time_fe.space_fe.dofs, V_dofs, ".-", label="V")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    app()
