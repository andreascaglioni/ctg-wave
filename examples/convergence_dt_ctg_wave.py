"""Time step size refinement convergence test for CTG on 1st order formulation of the wave equation.
Use separate variables for u and v instead of a 2d vectorial unknown."""

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
    n_cells_space = 40
    msh_x = mesh.create_unit_interval(comm, n_cells_space)
    V_x = fem.functionspace(msh_x, ("Lagrange", 1, (1,)))  # 1d space
    boundary_D = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))  # noqa: E731

    # time
    start_time = 0.0
    end_time = 1.0
    # t_slab_size = 0.01
    ddt = 2.**(-np.arange(1, 7))
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
    err = np.zeros_like(ddt)
    nn_dofs = np.zeros(ddt.size, dtype=int)
    for i, dt in enumerate(ddt):
        print("\ndt = ", dt)

        time_slabs, space_fe, sol_slabs = ctg_wave(comm, boundary_D, V_x, start_time, end_time, dt, order_t, boundary_data_u, boundary_data_v, exact_rhs_0, exact_rhs_1, initial_data_u, initial_data_v)
        err[i], rel_err, nn_dofs[i], err_slabs, norm_u_slabs = compute_err_ndofs(comm, order_t, err_type_x, err_type_t, time_slabs, space_fe, sol_slabs, exact_sol_u)    
        print("Total error", float_f(err[i]) )
        # plot_uv_at_T(time_slabs, space_fe, sol_slabs, exact_sol_u, exact_sol_v)
        # plt.show()

    print("CONVERGENCE SUMMARY:")
    print("ddt", ddt)
    print("nn_dofs", nn_dofs)
    print("err", err)
    rr = compute_rate(ddt, err)
    r = np.median(rr)
    C = err[0] / (ddt[0]**r)

    # Plot
    plt.figure()
    plt.loglog(ddt, err, '.-')
    plt.loglog(ddt, C * ddt**r, 'k-', label=f"rate {r:.2f}")

    plt.xlabel("dt")
    plt.title("Convergence L^{inf}(t)-H^1(x) error CTG")
    plt.legend()

    plt.show()