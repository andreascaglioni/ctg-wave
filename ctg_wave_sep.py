"""Implmenet CTG for 1st order formulation of the wave equation.
Use separate variables for u and v instead of a 2d vectorial unknown."""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh

from CTG.utils import float_f, cart_prod_coords
from CTG.ctg_hyperbolic import compute_err_ndofs, ctg_wave


def plot_uv_at_T(time_slabs, space_fe, sol_slabs):
    n_x = space_fe.n_dofs
    assert sol_slabs[0].size % 2 == 0, "sol_slabs[0].size must be even, got {}".format(sol_slabs[0].size)
    n_dofs_scalar = int(sol_slabs[0].size / 2)
    
    final_slab = time_slabs[-1]
    X_final = sol_slabs[-1]
    tx_final = cart_prod_coords(np.array([final_slab[0]]), space_fe.dofs)

    plt.figure(figsize=(8, 5))
    u = X_final[n_dofs_scalar-n_x:n_dofs_scalar]
    plt.plot(space_fe.dofs, u, "o-", label="u (numerical)")
    plt.plot(space_fe.dofs, exact_sol_u(tx_final), "--", label="u (exact)")
    plt.plot(space_fe.dofs, X_final[-n_x:], "s-", label="v (numerical)")
    plt.plot(space_fe.dofs, exact_sol_v(tx_final), ":", label="v (exact)")
    plt.title(f"u and v at final time t={round(final_slab[1], 4)}")
    plt.legend()
    plt.tight_layout()

def plot__uv_tt(time_slabs, space_fe, sol_slabs):
    n_x = space_fe.n_dofs
    assert sol_slabs[0].size % 2 == 0, "sol_slabs[0].size must be even, got {}".format(sol_slabs[0].size)
    n_dofs_scalar = int(sol_slabs[0].size / 2)

    # Compute bounds y axis
    uu = np.array([X[0:n_dofs_scalar] for X in sol_slabs])
    umin = np.amin(uu)
    umax = np.amax(uu)
    vv = np.array([X[n_dofs_scalar:] for X in sol_slabs])
    vmin = np.amin(vv)
    vmax = np.amax(vv)

    plt.figure(figsize=(10, 4))
    for i, slab in enumerate(time_slabs):
        X = sol_slabs[i]
        plt.clf()

        # Plot u on the left subplot
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(space_fe.dofs, X[0:n_x], ".", label=f"u at t={round(slab[0], 4)}")
        tx = cart_prod_coords(np.array([slab[0]]), space_fe.dofs)
        ax1.plot(space_fe.dofs, exact_sol_u(tx), "-", label="u exact")
        ax1.set_title(f"u at t={round(slab[0], 4)}")
        ax1.legend()
        ax1.set_ylim((umin, umax))

        # Plot v on the right subplot
        ax2 = plt.subplot(1, 2, 2)
        vv = X[n_dofs_scalar:n_dofs_scalar+n_x]
        ax2.plot(space_fe.dofs, vv, ".", label=f"v at t={round(slab[0], 4)}")
        ax2.plot(space_fe.dofs, exact_sol_v(tx), "-", label="v exact")
        ax2.set_title(f"v at t={round(slab[0], 4)}")
        ax2.legend()
        ax2.set_ylim((vmin, vmax))
        plt.tight_layout()
        plt.pause(0.5)
    plt.show()

def plot_error_tt(time_slabs, err_slabs, norm_u_slabs):
    times = [slab[1] for slab in time_slabs]
    rel_errs = err_slabs / norm_u_slabs
    plt.figure()
    plt.plot(times, err_slabs, marker='o', label="error")
    plt.plot(times, rel_errs, marker='o', label="relative error")
    plt.xlabel("Time")
    plt.title("Error over time")
    plt.legend()

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
    time_slabs, space_fe, sol_slabs = ctg_wave(comm, boundary_D, V_x, start_time, end_time, t_slab_size, order_t, boundary_data_u, boundary_data_v, exact_rhs_0, exact_rhs_1, initial_data_u, initial_data_v)
    
    print("POST PROCESS")
    # Compute error, total number of dofs
    n_dofs, total_err, total_rel_err, err_slabs, norm_u_slabs = compute_err_ndofs(comm, order_t, err_type_x, err_type_t, time_slabs, space_fe, sol_slabs, exact_sol_u)    
    print("Total error", float_f(total_err), "Total relative error", float_f(total_rel_err))
    print("error over slabls", err_slabs)

    # Plot
    plot_error_tt(time_slabs, err_slabs, norm_u_slabs)
    # plot__uv_tt(time_slabs, space_fe, sol_slabs)
    plot_uv_at_T(time_slabs, space_fe, sol_slabs)

    plt.show()