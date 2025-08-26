"""Implmenet CTG for 1st order formulation of the wave equation.
Use separate variables for u and v instead of a 2d vectorial unknown."""

import sys
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import scipy.sparse
from dolfinx import fem, mesh

from CTG.utils import float_f, compute_time_slabs, cart_prod_coords, compute_error_slab
from CTG.FE_spaces import SpaceFE, TimeFE
from CTG.ctg_hyperbolic import impose_IC_BC, assemble, compute_err_ndofs, ctg_wave






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
    order_x = 1
    boundary_D = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))  # noqa: E731
    V_x = fem.functionspace(msh_x, ("Lagrange", 1, (1,)))  # 1d space

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
    time_slabs, space_fe, n_x, sol_slabs, n_dofs_scalar = ctg_wave(comm, boundary_D, V_x, start_time, end_time, t_slab_size, order_t, boundary_data_u, boundary_data_v, exact_rhs_0, exact_rhs_1, initial_data_u, initial_data_v)
    
    print("POST PROCESS")
    # Compute error, total number of dofs
    n_dofs, total_err, total_rel_err, err_slabs, norm_u_slabs = compute_err_ndofs(comm, order_t, err_type_x, err_type_t, time_slabs, space_fe, sol_slabs, exact_sol_u)    
    print("Total error", float_f(total_err), "Total relative error", float_f(total_rel_err))
    print("error over slabls", err_slabs)

    # Plot relative error over time slabs
    times = [slab[1] for slab in time_slabs]
    rel_errs = err_slabs / norm_u_slabs
    plt.figure()
    plt.plot(times, err_slabs, marker='o', label="error")
    plt.plot(times, rel_errs, marker='o', label="relative error")
    plt.xlabel("Time")
    plt.title("Error over time")
    plt.legend()
    plt.show()

    # Plot solution
    for i, slab in enumerate(time_slabs[::10]):
        print(f"Slab_{i} = D x ({round(slab[0], 4)}, {round(slab[1], 4)}) ...")
        X = sol_slabs[i]
        plt.plot(space_fe.dofs, X[0:n_x], ".", label="u")
        plt.plot(space_fe.dofs, X[n_dofs_scalar:n_dofs_scalar+n_x], ".", label="v")
        tx = cart_prod_coords(np.array([slab[0]]), space_fe.dofs)
        plt.plot(space_fe.dofs, exact_sol_u(tx), "-", label="u exa")
        plt.plot(space_fe.dofs, exact_sol_v(tx), "-", label="v exa")
        plt.legend()
        plt.show()