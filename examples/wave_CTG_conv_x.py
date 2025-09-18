"""Mesh refinement convergence test for CTG on 1st order formulation of the wave equation.
Use separate variables for u and v instead of a 2d vectorial unknown.


The wave equation reads: u_tt -  Δu = f
and is equipped with initial and boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh

from CTG.utils import float_f, plot_error_tt, plot_uv_at_T, plot_uv_tt, compute_rate
from CTG.ctg_hyperbolic import compute_err_ndofs, ctg_wave


if __name__ == "__main__":
    # SETTINGS
    seed = 0
    np.random.seed(0)
    comm = MPI.COMM_SELF
    np.set_printoptions(formatter={"float_kind": float_f})

    # PARAMETERS
    # space
    order_x = 1
    nn_cells = 2**np.arange(1, 7)
    boundary_D = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))  # noqa: E731

    # time
    start_time = 0.0
    end_time = 1.0
    # t_slab_size = 0.01
    dt = 2.**(-7.)
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

    print("CONVERNGENCE TEST")
    err = np.zeros(nn_cells.size, dtype=float)
    nn_dofs = np.zeros_like(nn_cells)
    for i, n_cells_space in enumerate(nn_cells):
        print("\nn cells = ", n_cells_space)

        msh_x = mesh.create_unit_interval(comm, n_cells_space)
        V_x = fem.functionspace(msh_x, ("Lagrange", 1, (1,)))  # 1d space

        time_slabs, space_fe, sol_slabs = ctg_wave(comm, boundary_D, V_x, start_time, end_time, dt, order_t, boundary_data_u, boundary_data_v, exact_rhs_0, exact_rhs_1, initial_data_u, initial_data_v)
        err[i], rel_err, nn_dofs[i], err_slabs, norm_u_slabs = compute_err_ndofs(comm, order_t, err_type_x, err_type_t, time_slabs, space_fe, sol_slabs, exact_sol_u)    
        print("Total error", float_f(err[i]) )
        # plot_uv_at_T(time_slabs, space_fe, sol_slabs, exact_sol_u, exact_sol_v)
        # plt.show()

    print("\nCONVERGENCE SUMMARY:")
    hh = 1./nn_cells
    print("hh", hh)
    print("nn_dofs", nn_dofs)
    print("err", err)
    rr = compute_rate(hh, err)
    r = np.median(rr)
    C = err[0] / (hh[0]**r)

    # Plot
    plt.figure()
    plt.loglog(hh, err, '.-')
    plt.loglog(hh, C * hh**r, 'k-', label=f"rate {r:.2f}")

    plt.xlabel("h")
    plt.title("Convergence L^{inf}(t)-H^1(x) error CTG")
    plt.legend()

    plt.show()