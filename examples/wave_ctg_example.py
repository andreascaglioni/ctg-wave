"""Implmenet CTG for 1st order formulation of the wave equation.
Use separate variables for u and v instead of a 2d vectorial unknown.

The wave equation reads: u_tt -  Δu = f
and is equipped with initial and boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh

import sys
sys.path.insert(0, '.')
from CTG.post_process_utils import plot_error_tt, plot_uv_at_T, plot_uv_tt
from CTG.utils import float_f
from CTG.ctg_hyperbolic import compute_err_ndofs, ctg_wave
import os




if __name__ == "__main__":
    # SETTINGS
    seed = 0
    np.random.seed(0)
    comm = MPI.COMM_SELF
    np.set_printoptions(formatter={"float_kind": float_f})

    # PARAMETERS
    # space
    order_x = 1
    n_cells_space = 40
    msh_x = mesh.create_unit_interval(comm, n_cells_space)
    V_x = fem.functionspace(msh_x, ("Lagrange", 1, (1,)))  # 1d space
    boundary_D = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))  # noqa: E731

    # time
    start_time = 0.0
    end_time = 1.0
    t_slab_size = 0.01
    order_t = 1

    # Exact sol
    from data.exact_solution_wave_sep2 import (
        exact_sol_u,
        exact_sol_v,
        exact_rhs_0,
        exact_rhs_1,
        boundary_data_u,
        boundary_data_v,
        initial_data_u,
        initial_data_v,
    )

    # error
    err_type_x = "h1"
    err_type_t = "linf"

    print("COMPUTE")
    time_slabs, space_fe, sol_slabs = ctg_wave(comm, boundary_D, V_x, start_time, end_time, t_slab_size, order_t, boundary_data_u, boundary_data_v, exact_rhs_0, exact_rhs_1, initial_data_u, initial_data_v)
    
    print("POST PROCESS")
    # Compute error, total number of dofs
    n_dofs, total_err, total_rel_err, err_slabs, norm_u_slabs = compute_err_ndofs(comm, order_t, err_type_x, err_type_t, time_slabs, space_fe, sol_slabs, exact_sol_u)    
    print("Total error", float_f(total_err), "Total relative error", float_f(total_rel_err))
    print("error over slabls", err_slabs)

    # Plot
    plot_error_tt(time_slabs, err_slabs, norm_u_slabs)
    # plot_uv_tt(time_slabs, space_fe, sol_slabs, exact_sol_u, exact_sol_v)
    plot_uv_at_T(time_slabs, space_fe, sol_slabs, exact_sol_u, exact_sol_v)

    plt.show()