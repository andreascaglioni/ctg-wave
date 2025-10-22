"""Compute an ensamble of solutions of the PWE for parameters sampled as a high-dimensional standard Gaussian random variable. Plot quantites of interest related to time.
"""

import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh
import csv
import sys
sys.path.insert(0, ".")
from ctg.brownian_motion import param_LC_W
from ctg.ctg_solver import CTGSolver
from ctg.post_process import float_f, compute_energy_tt, plot_uv_tt, inverse_DS_transform


if __name__ == "__main__":
    # SETTINGS
    seed = 1
    np.random.seed(seed)
    comm = MPI.COMM_SELF
    np.set_printoptions(formatter={"float_kind": float_f})

    # PARAMETERS
    # space
    order_x = 1
    n_cells_space = 100
    msh_x = mesh.create_unit_interval(comm, n_cells_space)
    V_x = fem.functionspace(msh_x, ("Lagrange", order_x, (1,)))
    
    # Time
    t_slab_size = 0.01
    order_t = 1
    start_time = 0.
    end_time = 1.
    
    
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
        "initial_data_v": initial_data_v
    }
    
    # error
    err_type_x = "h1"
    err_type_t = "l2"

    tt = np.linspace(start_time, end_time, ceil(1/t_slab_size))
    n_samples = 10

    for seed in range(n_samples):
        print("Sample", seed+1)
        np.random.seed(seed)
        y = np.random.standard_normal(100)
        W_t = lambda tt: param_LC_W(y, tt, T=end_time)[0]
        physics_params["W_t"] = W_t

        # COMPUTE
        ctg_solver = CTGSolver(numerics_params, verbose=False)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = ctg_solver.run(physics_params)

        XX = [inverse_DS_transform(sol_slabs[i], W_t, space_time_fe.space_fe, time_slabs[i], comm, order_t) for i in range(len(sol_slabs))]
   
        # POST PROCESS
        space_fe = space_time_fe.space_fe

        # Plot energy
        EE, _, _ = compute_energy_tt(space_fe, XX)
        plt.figure("energy")
        plt.plot(tt, EE, '-', linewidth=2)

        # Plot U(t,x0) for fixed x0, as finction of t
        n_dof_track = int(space_fe.n_dofs/4)
        plt.figure("track")
        U_t = np.array([X[n_dof_track] for X in XX])
        plt.plot(tt, U_t, '-', linewidth=2)

        plt.figure("W")
        plt.plot(tt, W_t(tt), '-', linewidth=2)
    
    plt.figure("energy")
    plt.title("Energy (kinetic + potential) of PWE sample")
    plt.tight_layout()
    plt.xlabel("t")

    plt.figure("track")
    plt.title("Value uf u(t, x0) for x0 = "+str(space_fe.dofs[n_dof_track]))
    plt.tight_layout()
    plt.grid(True)
    plt.xlabel("t")

    plt.figure("W")
    plt.title("Browniann motion sample path")
    plt.tight_layout()
    plt.xlabel("t")
    plt.show()
