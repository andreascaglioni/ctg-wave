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
import os
import argparse
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from ctg.brownian_motion import param_LC_W
from ctg.post_process import float_f, compute_energy_tt, plot_uv_tt, inverse_DS_transform
from ctg.ctg_solver import CTGSolver
from ctg.post_process import inverse_DS_transform



def main():
    # SETTINGS
    seed = 1
    np.random.seed(seed)
    comm = MPI.COMM_SELF
    np.set_printoptions(formatter={"float_kind": float_f})
    dir_save = os.path.join("results", "dir_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dir_save, exist_ok=True)

    # PARAMETERS
    # Parse inputs from call
    parser = argparse.ArgumentParser(description="Parametric Wave Equation CTG Example")
    parser.add_argument("--n_cells_space", type=int, default=100, help="Number of spatial cells (default: 100)")
    parser.add_argument("--t_slab_size", type=float, default=0.01, help="Time slab size (default: 0.01)")
    parser.add_argument("--end_time", type=float, default=1., help="End time of simulation (default: 1.0)")
    args = parser.parse_args()
    
    # space
    order_x = 1
    n_cells_space = args.n_cells_space
    msh_x = mesh.create_unit_interval(comm, n_cells_space)
    V_x = fem.functionspace(msh_x, ("Lagrange", order_x, (1,)))
    
    # Time
    t_slab_size = args.t_slab_size
    order_t = 1
    start_time = 0.
    end_time = args.end_time
    y = np.random.standard_normal(100)
    W_t = lambda tt: param_LC_W(y, tt, T=end_time)[0]
    
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
    
    print("COMPUTE")
    ctg_solver = CTGSolver(numerics_params, verbose=False)
    sol_slabs, time_slabs, space_time_fe, total_n_dofs = ctg_solver.run(physics_params)
    
    # Inverse DOss Sussmann transform
    XX = [inverse_DS_transform(sol_slabs[i], physics_params["W_t"], space_time_fe.space_fe, time_slabs[i], comm, order_t) for i in range(len(sol_slabs))]

    print("POST PROCESS")
    space_fe = space_time_fe.space_fe
    dofs = space_fe.dofs.flatten()
    n_x = space_fe.n_dofs
    n_scalar=int(sol_slabs[0].size/2)
    XX_T = XX[-1]
    U_T = XX_T[n_scalar-n_x:n_scalar]  
    V_T = XX_T[-n_x:]
    tt = np.linspace(start_time, end_time, ceil((end_time-start_time)/t_slab_size))

    # Compute metrics
    WW = physics_params["W_t"](tt)
    EE, ppot, kkin = compute_energy_tt(space_fe, XX)
    
    # Export to CSV u_final, v_final
    csv_filename_uv = os.path.join(dir_save, "wave_uv_final.csv")
    with open(csv_filename_uv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dof", "U_T", "V_T"])
        for d, uf, vf in zip(dofs, U_T, V_T):
            writer.writerow([d, uf, vf])
    
    # Export to CSV tt, WW, EE
    csv_filename = os.path.join(dir_save, "WW_wave_energy.csv")
    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tt", "WW", "EE"])
        for t, w, e in zip(tt, WW, EE):
            writer.writerow([t, w, e])
    print(f"Exported data to {csv_filename}")

    # Plots
    plt.figure("Brownian motion")
    plt.plot(tt, WW, '.-')
    plt.title("Browniann motion sample path")
    plt.tight_layout()
    plt.xlabel("t")

    plt.figure("Energy")
    plt.plot(tt, EE, '.-')
    plt.title("Energy (kinetic + potential) of PWE sample")
    plt.tight_layout()
    plt.xlabel("t")

    # Plot solution original SDE at final time 
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    y_min, y_max = np.amin(sol_slabs[-1]), np.amax(sol_slabs[-1])
    axs[0].plot(space_fe.dofs, U_T, ".-", label="U(T) (numerical)")
    axs[0].set_title(f"U at final time t={round(time_slabs[-1][1], 4)}")
    axs[0].set_xlabel("x")
    axs[0].legend()
    axs[1].plot(space_fe.dofs, V_T, ".-", label="V(T) (numerical)")
    axs[1].set_title(f"V at final time t={round(time_slabs[-1][1], 4)}")
    axs[1].set_xlabel("x")
    axs[1].legend()
    plt.tight_layout()
    ymin = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0])
    ymax = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
    axs[0].set_ylim(ymin, ymax)
    axs[1].set_ylim(ymin, ymax)

    # Plot U(t,x0) for fixed x0, as function of t
    plt.figure("U(t, x0) vs t")
    n_dof_track = int(space_fe.n_dofs/4)
    U_T = np.array([X[n_dof_track] for X in XX])
    plt.plot(tt, U_T, '.-')
    plt.title("U(t, x0) for x0 = "+str(space_fe.dofs[n_dof_track]))
    plt.xlabel("t")
    plt.tight_layout()

    # plot_uv_tt(time_slabs, space_fe, sol_slabs)

    # 3D plot u sspace-time
    plt.figure()
    X, Y = np.meshgrid(space_fe.dofs, tt)
    UU = np.array([X[:space_fe.n_dofs] for X in XX])
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, UU, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Space-time solution u surface')

    plt.show()

    
if __name__ == "__main__":
    main()