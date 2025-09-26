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
from math import ceil
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh
import csv
import sys
sys.path.insert(0, ".")
from CTG.brownian_motion import param_LC_W
from CTG.post_process_utils import float_f, compute_energy_tt, plot_uv_at_T, plot_uv_tt
from CTG.ctg_hyperbolic import ctg_wave
from CTG.utils import inverse_DS_transform


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
    V_x = fem.functionspace(msh_x, ("Lagrange", order_x, (1,)))
    
    # Time
    t_slab_size = 0.1
    order_t = 1
    start_time = 0.
    end_time = 1.
    y = 1*np.random.standard_normal(100)
    def W_t(tt):  # return Callable[[numpy.ndarray], numpy.ndarray] 
        tt = np.atleast_1d(tt)  # hande scalar tt
        if len(tt.shape) == 1:
            pass    # ok
        if len(tt.shape) == 2 and tt.shape[1] == 3:  # input dolfinx interpolate. Purge last 2 rows
            tt = tt[0, :]
        WW = 1.*param_LC_W(y, tt, T=end_time)[0]
        return WW  
            
    
    # Problem data 
    from data.data_param_wave_eq import (
        exact_rhs_0,
        exact_rhs_1,
        boundary_data_u,
        boundary_data_v,
        initial_data_u,
        initial_data_v
    )

    numerics_params = {
        "comm": comm, 
        "V_x": V_x,
        "t_slab_size": t_slab_size,
        "order_t": order_t
    }
    
    physics_params = {
        "boundary_D": lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
        "start_time": start_time,
        "end_time": end_time,
        "boundary_data_u": boundary_data_u,
        "boundary_data_v": boundary_data_v,
        "exact_rhs_0": exact_rhs_0,
        "exact_rhs_1": exact_rhs_1,
        "initial_data_u": initial_data_u,
        "initial_data_v": initial_data_v,
        "W_t": W_t
    }
    
    # error
    err_type_x = "h1"
    err_type_t = "linf"

    
    
    print("COMPUTE")
    time_slabs, sol_slabs, space_fe, time_fe_last = ctg_wave(physics_params, numerics_params)
    

    
    print("POST PROCESS")
    # Compute post-processed quantities
    # n_dofs, total_err, total_rel_err, err_slabs, norm_u_slabs = compute_err_ndofs(comm, order_t, err_type_x, err_type_t, time_slabs, space_fe, sol_slabs, exact_sol_u)    
    # print("Total error", float_f(total_err), "Total relative error", float_f(total_rel_err))
    # print("error over slabls", err_slabs)
    tt = np.linspace(start_time, end_time, ceil(1/t_slab_size))
    WW = physics_params["W_t"](tt)
    EE, ppot, kkin = compute_energy_tt(space_fe, sol_slabs)
    n_x = space_fe.n_dofs
    n_scalar=int(sol_slabs[0].size/2)
    u_final = sol_slabs[-1][n_scalar-n_x:n_scalar]
    v_final = sol_slabs[-1][-n_x:]
    XX = inverse_DS_transform(sol_slabs[-1], physics_params["W_t"], space_fe, time_fe_last)
    UU_final = XX[n_scalar-n_x:n_scalar]
    VV_final = XX[-n_x:]

    # Print
    print("Total energy (kinetic+potential):", EE)

    # Plots
    plt.figure()
    plt.plot(tt, WW, '.-')
    plt.title("Browniann motion sample path")
    plt.tight_layout()
    plt.xlabel("t")

    plt.figure()
    plt.plot(tt, EE, '.-')
    plt.title("Energy (kinetic + potential) of PWE sample")
    plt.tight_layout()
    plt.xlabel("t")
    
    # plot_uv_tt(time_slabs, space_fe, sol_slabs)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # Left axis: u_final and UU_final
    axs[0].plot(space_fe.dofs, u_final, "o-", label="u numerical")
    axs[0].plot(space_fe.dofs, UU_final, ".-", label="u (DS transform)")
    axs[0].set_title(f"u at final time t={round(time_slabs[-1][1], 4)}")
    axs[0].set_xlabel("x")
    axs[0].legend()
    axs[0].grid(True)
    # Right axis: v_final and VV_final
    axs[1].plot(space_fe.dofs, v_final, "s-", label="v numerical")
    axs[1].plot(space_fe.dofs, VV_final, ".-", label="v (DS transform)")
    axs[1].set_title(f"v at final time t={round(time_slabs[-1][1], 4)}")
    axs[1].set_xlabel("x")
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()

    # Export to CSV u_final, v_final
    dofs = space_fe.dofs.flatten()
    csv_filename_uv = "wave_uv_final.csv"
    with open(csv_filename_uv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dof", "u_final", "v_final"])
        for dof, u_final, v_final in zip(dofs, u_final, v_final):
            writer.writerow([dof, u_final, v_final])
    print(f"Exported DOFs, u_final, v_final to {csv_filename_uv}")

    # Export to CSV WW, EE
    csv_filename = "wave_energy.csv"
    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tt", "WW", "EE"])
        for t, w, e in zip(tt, WW, EE):
            writer.writerow([t, w, e])
    print(f"Exported data to {csv_filename}")

    plt.show()

    
