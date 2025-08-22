"""CTG approximation wave equation."""

import sys
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh

sys.path.append("./")
from CTG.utils import compute_time_slabs, float_f, cart_prod_coords
from CTG.ctg_hyperbolic import run_CTG_wave
from CTG.FE_spaces import SpaceFE


if __name__ == "__main__":
    # SETTINGS
    comm = MPI.COMM_SELF
    np.set_printoptions(formatter={"float_kind": float_f})

    # DATA
    # Numerics
    n_cells_space = 4
    msh_x = mesh.create_unit_interval(comm, n_cells_space)
    order_x = 1
    V_x = fem.functionspace(msh_x, ("Lagrange", 1, (2,)))

    t_slab_size = 0.01
    order_t = order_x  # NB polynomial degree in time TRIAL space -> TEST space: -1
    n_time = 1  # number of temporal elements per time slab

    # Physics
    start_time = 0.0
    end_time = 1.0
    boundary_D = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))  # noqa: E731
    from data.exact_solution_wave import (
        exact_rhs,
        boundary_data,
        initial_data,
        exact_sol,
    )

    # COMPUTE
    time_slabs = compute_time_slabs(start_time, end_time, t_slab_size)
    space_fe = SpaceFE(msh_x, V_x, boundary_data, boundary_D)

    dofs_X_slabs, errs_slabs, norms_slabs, n_dofs = run_CTG_wave(
        comm,
        space_fe,
        n_time,
        order_t,
        time_slabs,
        boundary_data,
        exact_rhs,
        initial_data,
        exact_sol,
        err_type_x="h1",
        err_type_t="linf",
    )

    # POST-PROCESS
    total_err = sqrt(np.sum(np.square(errs_slabs)))
    total_rel_err = total_err / sqrt(np.sum(np.square(norms_slabs)))
    print("Total error", float_f(total_err), "Total relative error", float_f(total_rel_err))

    # Plot u only
    n_dofs_tx_scalar = int(dofs_X_slabs[0].size / 2)
    xx = space_fe.dofs  # msh_x.geometry.x[:, 0]  # msh_x.geometry.x has shape (# nodes, 3)
    dt = t_slab_size / n_time
    n_dofs_t = n_time+1
    n_dofs_x = space_fe.n_dofs
    for i_s, slab in enumerate(time_slabs):
        dofs_X_slab = dofs_X_slabs[i_s]
        dofs_u_slab = dofs_X_slab[:n_dofs_tx_scalar]
        for i_t, t in enumerate(slab):
            u_t_dofs = dofs_X_slab[i_t * n_dofs_x : (i_t + 1) * n_dofs_x]
            plt.plot(xx, u_t_dofs, "o")

            X = cart_prod_coords(np.array([t]), xx)
            u_ex = exact_sol(X)[0, :]
            plt.plot(xx, u_ex, ".-")

            plt.title(f"Slabs[{i_s}] = {np.round(slab, 2)}; time {i_t}")
            plt.ylim([-1, 1])
            plt.show()