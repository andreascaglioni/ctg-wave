"""Example of CTG for 1st order formulation of a parametric wave equation (PWE) arising from a stochastic wave equation using Doss-Sussmann trasnform and Levy-Ciesielski parametrization of Brownian motion.

The first order formulation of the parametric wave equation (PWE) is:

    u_t = v + W u
    v_t = Δu + W v + W² u + f

where:
    u: unknown function
    v: time derivative of u
    W: stochastic process (e.g., Brownian motion)
    f: source term

and is equipped with initial and boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh

import sys
sys.path.insert(0, ".")
from CTG.error import compute_err
from CTG.utils import float_f, plot_error_tt, plot_uv_at_T, plot_uv_tt, param_LC_W, plot_energy_tt
from CTG.ctg_hyperbolic import ctg_wave
from scipy.interpolate import interp1d
import csv






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

    # Data problem
    from data.data_param_wave_eq import (
        exact_rhs_0,
        exact_rhs_1,
        boundary_data_u,
        boundary_data_v,
        initial_data_u,
        initial_data_v
    )

    # error
    err_type_x = "h1"
    err_type_t = "linf"

    print("COMPUTE")
    # Sample a path for wiener process
    y = 1.*np.random.standard_normal(100)
    W_t = lambda tt : 1.*param_LC_W(y, tt, T=end_time)[0]  # output: 1D array
    time_slabs, space_fe, sol_slabs, total_n_dofs = ctg_wave(comm, boundary_D, V_x, start_time, end_time, t_slab_size, order_t, boundary_data_u, boundary_data_v, exact_rhs_0, exact_rhs_1, initial_data_u, initial_data_v, W_t)
    
    print("POST PROCESS")
    # Compute error, total number of dofs
    # n_dofs, total_err, total_rel_err, err_slabs, norm_u_slabs = compute_err_ndofs(comm, order_t, err_type_x, err_type_t, time_slabs, space_fe, sol_slabs, exact_sol_u)    
    # print("Total error", float_f(total_err), "Total relative error", float_f(total_rel_err))
    # print("error over slabls", err_slabs)

    # Plot

    # PLot W over tt 
    tt = np.linspace(start_time, end_time, n_cells_space+1)
    WW = W_t(tt)
    plt.plot(tt, WW, '.-')
    plt.xlabel("t")
    plt.title("Browniann motion sample path")

    # Plot Energy over tt
    EE = plot_energy_tt(space_fe, sol_slabs, tt)

    # Export to CSV file
    csv_filename = "wave_energy.csv"
    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tt", "WW", "EE"])
        for t, w, e in zip(tt, WW, EE):
            writer.writerow([t, w, e])
    print(f"Exported data to {csv_filename}")

    # plot_uv_tt(time_slabs, space_fe, sol_slabs)
    
    u_fianl, v_final = plot_uv_at_T(time_slabs, space_fe, sol_slabs)

    # Export DOFs, u_final, v_final to CSV
    dofs = space_fe.dofs.flatten()
    csv_filename_uv = "wave_uv_final.csv"
    with open(csv_filename_uv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dof", "u_final", "v_final"])
        for dof, u, v in zip(dofs, u_fianl, v_final):
            writer.writerow([dof, u, v])
    print(f"Exported DOFs, u_final, v_final to {csv_filename_uv}")

    plt.show()