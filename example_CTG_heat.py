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
    compute_time_slabs,
    run_CTG_elliptic,
)

sys.path.append("../stochllg")
from utils import float_f


if __name__ == "__main__":
    # ------------------------------------------------------------------------ #
    #                                   DATA                                   #
    # ------------------------------------------------------------------------ #
    # Numerics data
    comm = MPI.COMM_SELF
    n_space = 1600 * 2
    msh_x = mesh.create_unit_interval(comm, n_space)
    order_x = 1
    V_x = fem.functionspace(msh_x, ("Lagrange", 1))
    t_slab_size = 0.1
    order_t = 1  # polynomial degree in time
    n_time = 40  # number of temporal elements per time-slab

    # Physical data
    start_time = 0.0
    end_time = 0.5
    boundary_D = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))  # noqa: E731
    from data.exact_solution_heat import (
        exact_rhs,
        boundary_data,
        initial_data,
        exact_sol,
    )

    # ------------------------------------------------------------------------ #
    #                                  COMPUTE                                 #
    # ------------------------------------------------------------------------ #

    time_slabs = compute_time_slabs(start_time, end_time, t_slab_size)

    # FE object for SPACE discretization
    Space = SpaceFE(msh_x, V_x, boundary_data, boundary_D)

    # Time marching
    errs_slabs, norms_slabs = run_CTG_elliptic(
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
    total_err = sqrt(np.sum(np.square(errs_slabs)))
    total_rel_err = total_err / sqrt(np.sum(np.square(norms_slabs)))
    print(
        "Total error",
        float_f(total_err),
        "Total relative error",
        float_f(total_rel_err),
    )
