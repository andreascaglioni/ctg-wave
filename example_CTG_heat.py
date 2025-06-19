"""CTG approximation wave equation. Some material is taken from

https://github.com/mathmerizing/SpaceTimeFEM_2023-2024/blob/main/Exercise3/Exercise_3_Linear_PDE.ipynb

"""

import sys
from math import sqrt
import numpy as np
from mpi4py import MPI

from dolfinx import fem, mesh

from cont_t_galerkin import (
    SpaceFE,
    cart_prod_coords,
    compute_time_slabs,
    run_CTG_elliptic,
)
from data.exact_solution_elliptic import exact_rhs, boundary_data, initial_data, exact_sol

sys.path.append("../stochllg")
from utils import float_f


if __name__ == "__main__":
    # ------------------------------------------------------------------------ #
    #                                   DATA                                   #
    # ------------------------------------------------------------------------ #
    # Numerics data
    comm = MPI.COMM_SELF
    msh_x = mesh.create_unit_interval(comm, 100)
    order_x = 1
    V_x = fem.functionspace(msh_x, ("Lagrange", 1))
    slab_size = 0.1
    order_t = 1  # polynomial degree in time
    n_time = 40  # number of temporal elements per time-slab

    # Physical data
    start_time = 0.0
    end_time = 0.5

    # Exact solution
    from data.exact_solution_elliptic import exact_sol, boundary_data

    # ------------------------------------------------------------------------ #
    #                                  COMPUTE                                 #
    # ------------------------------------------------------------------------ #

    time_slabs = compute_time_slabs(start_time, end_time, slab_size)

    # FE object for SPACE discretization
    boundary_D = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))  # noqa: E731
    Space = SpaceFE(msh_x, V_x, boundary_data, boundary_D)

        # Time marching
    L2_errs, L2_norms = run_CTG_elliptic(
        comm,
        Space,
        n_time,
        order_t,
        time_slabs,
        boundary_data,
        exact_rhs,
        initial_data,
        exact_sol,
    )

    # -------------------------------------------------------------------- #
    #                             POST-PROCESS                             #
    # -------------------------------------------------------------------- #
    L2_err = sqrt(np.sum(np.square(L2_errs)))
    rel_L2_err = L2_err / sqrt(np.sum(np.square(L2_norms)))
    print(
        "Total L2 error",
        float_f(L2_err),
        "Total L2 relative error",
        float_f(rel_L2_err),
    )
