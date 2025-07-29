"""CTG approximation wave equation."""

import sys
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh

from CTG.utils import compute_time_slabs, float_f
from CTG.ctg_parabolic import run_CTG_parabolic
from CTG.FE_spaces import SpaceFE
from CTG.utils import 


if __name__ == "__main__":
    # SETTINGS
    comm = MPI.COMM_SELF
    np.set_printoptions(formatter={'float_kind':float_f})

    
    # DATA
    # Numerics
    n_space = 9
    msh_x = mesh.create_unit_interval(comm, n_space)
    order_x = 1
    V_x = fem.functionspace(msh_x, ("Lagrange", 1))

    t_slab_size = 1.0
    order_t = 1  # polynomial degree in time
    n_time = 3  # number of temporal elements per time slab

    # Physics
    start_time = 0.0
    end_time = t_slab_size  # for now don't change
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

    sol_slabs, errs_slabs, norms_slabs, n_dofs = run_CTG_hyperbolic(
        comm,
        space_fe,
        n_time,
        order_t,
        time_slabs,
        boundary_data,
        exact_rhs,
        initial_data,
        exact_sol,
    )

    # Plot
    from cont_t_galerkin.utils import cart_prod_coords

    sol = sol_slabs[0]
    xx = msh_x.geometry.x[:, 0]  # shape (# nodes, 3)
    dt = t_slab_size / n_time
    for i_t in range(n_time):
        t = i_t * dt
        u_t_dofs = sol[i_t * (n_space + 1) : (i_t + 1) * (n_space + 1)]
        plt.plot(xx, u_t_dofs, "o")

        X = cart_prod_coords(np.array([t]), xx)
        u_ex = exact_sol(X)
        plt.plot(
            xx,
            u_ex,
            ".-",
        )
        plt.show()

    # POST-PROCESS
    total_err = sqrt(np.sum(np.square(errs_slabs)))
    total_rel_err = total_err / sqrt(np.sum(np.square(norms_slabs)))
    print(
        "Total error",
        float_f(total_err),
        "Total relative error",
        float_f(total_rel_err),
    )
