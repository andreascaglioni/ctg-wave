"""Generate the PNG figures used in the project README.

This small script runs lightweight examples and saves the two images that
are embedded in ``README.md`` (space-time slab diagram and a sample
solution surface). Run it from the repository root; it requires the
package installed in editable mode and a working dolfinx/mpi environment
if you want full fidelity. For CI or documentation builds you can mock
the heavy dependencies instead.
"""

from math import ceil
from pathlib import Path
import numpy as np
from ctg.brownian_motion import param_LC_W
from ctg.config import AppConfig, load_config
from ctg.ctg_solver import CTGSolver
from ctg.plotting import plot_xt_slabs
import matplotlib.pyplot as plt
from ctg.FE_spaces import SpaceFE
from dolfinx import fem, mesh
from mpi4py import MPI

from ctg.post_process import inverse_DS_transform
from ctg.utils import compute_time_slabs


def plot_slabs():
    # Set data
    n_cells_space = 4
    order_x = 1

    def boundary_D(x):
        return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))

    t_slab_size = 0.1
    start_time = 0.0
    end_time = 1.0

    # Compute variables
    comm = MPI.COMM_SELF
    msh_x = mesh.create_unit_interval(comm, n_cells_space)
    V_x = fem.functionspace(msh_x, ("Lagrange", order_x, (1,)))
    space_fe = SpaceFE(V_x, boundary_D)
    time_slabs = compute_time_slabs(start_time, end_time, t_slab_size)

    # Plot
    plot_xt_slabs(space_fe, time_slabs)


def plot_solution_xt():
    # Data
    text_size = 14
    config_path = Path("examples/data_examples/data_swe.yaml")
    cfg: AppConfig = load_config(config_path)
    np.random.seed(cfg.numerics.seed)
    y = np.random.standard_normal(100)

    # Call API to solve
    def W_t(tt):
        return param_LC_W(y, tt, T=cfg.physics.end_time)[0]

    ctg_solver = CTGSolver(cfg.numerics)
    sol_slabs, time_slabs, space_time_fe, total_n_dofs = ctg_solver.run(cfg.physics, W_t)

    # Apply inverse Doss-Sussmann transform
    XX = [
        inverse_DS_transform(
            sol_slabs[i],
            W_t,
            space_time_fe.space_fe,
            time_slabs[i],
            cfg.numerics.comm,
            cfg.numerics.order_t,
        )
        for i in range(len(sol_slabs))
    ]

    # Plot
    plt.figure("xt_plot_u")
    s_t = cfg.physics.start_time
    e_t = cfg.physics.end_time
    tt = np.linspace(s_t, e_t, ceil((e_t - s_t) / cfg.numerics.t_slab_size))

    X, Y = np.meshgrid(space_time_fe.space_fe.dofs, tt)
    UU = np.array([X[: space_time_fe.space_fe.n_dofs] for X in XX])
    ax = plt.axes(projection="3d")

    ax.plot_surface(X, Y, UU, cmap="viridis")
    ax.set_xlabel("Space domain D", fontsize=text_size + 4)
    ax.set_ylabel("Time domain", fontsize=text_size + 4)
    ax.set_xticks([])
    ax.set_zticks([])
    ax.set_yticks(np.linspace(s_t, e_t, 10))
    ax.set_yticklabels(["0"] + ["" for _ in range(8)] + ["T"], fontsize=text_size + 4)
    ax.set_ylim(s_t, e_t)
    # set box aspect so that y is twice x (relative scaling)
    ax.set_box_aspect((1.0, 2.0, 1.0))

    for spine in ax.spines.values():
        spine.set_visible(False)


if __name__ == "__main__":
    plot_slabs()
    plot_solution_xt()

    plt.show()
