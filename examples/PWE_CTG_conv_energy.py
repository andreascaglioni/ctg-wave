"""Convergence of total energy (kinetic+potential) of CTG approximation of PWE for 1 fixed parameter."""

import numpy as np
from math import ceil, sqrt
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
from scipy.interpolate import interp1d


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
    tt_slab_size = 2.**(-np.arange(8))
    order_t = 1
    start_time = 0.
    end_time = 1.
    y = 1*np.random.standard_normal(1)
    def W_t(tt):  # return Callable[[numpy.ndarray], numpy.ndarray] 
        # tt must be rank 1
        tt = np.atleast_1d(tt)  # hande scalar tt
        if len(tt.shape) == 2 and tt.shape[1] == 3:  # input dolfinx interpolate. Purge last 2 rows
            tt = tt[0, :]
        
        return 1.*param_LC_W(y, tt, T=end_time)[0]    
    
    # Problem data 
    from data.data_param_wave_eq import (
        exact_rhs_0,
        exact_rhs_1,
        boundary_data_u,
        boundary_data_v,
        initial_data_u,
        initial_data_v
    )

    
    
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

    
    
    print("CONVERGENCE TEST")
    EE_save = []
    tt_save = []
    for i, t_slab_size in enumerate(tt_slab_size):
        print("slab size", t_slab_size)

        numerics_params = {
        "comm": comm, 
        "V_x": V_x,
        "t_slab_size": t_slab_size,
        "order_t": order_t
    }
        
        time_slabs, sol_slabs, space_fe, time_fe_last = ctg_wave(physics_params, numerics_params)
        
        # check energy convergence
        tt = np.linspace(start_time, end_time, ceil(1/t_slab_size))
        WW = physics_params["W_t"](tt)
        EE, ppot, kkin = compute_energy_tt(space_fe, sol_slabs)
        print("Total energy (kinetic+potential):", EE)
        plt.figure("energy")
        plt.plot(tt, EE, '.-', label="dt="+str(t_slab_size))

        EE_save.append(EE)
        tt_save.append(tt)
        # Export tt and EE as 2 columns to CSV
        with open(f"energy_iter_{i}_dt_{t_slab_size:.2e}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["t", "EE"])
            for t_val, ee_val in zip(tt, EE):
                writer.writerow([t_val, ee_val])
        
    
    
    plt.title("Energy (kinetic + potential) of PWE sample")
    plt.tight_layout()
    plt.xlabel("t")
    plt.legend()

    # Compute error energyu over tt
    tt_ref = tt_save.pop()
    E_ref = EE_save.pop()
    dt_ref = tt_slab_size[-1]
    tt_slab_size = tt_slab_size[:-1]
    err_l2_E = []
    err_E_tt = []
    for i, dt in enumerate(tt_slab_size):
        E_curr = EE_save[i]
        t_curr = tt_save[i]
        interp_E = interp1d(t_curr, E_curr, kind="linear", fill_value="extrapolate", bounds_error=False)
        E_curr_interp = interp_E(tt_ref)
        err_E_tt.append(np.abs(E_curr_interp - E_ref))
    
    # Plot error energy over tt
    plt.figure("conv E over tt")
    for i, dt in enumerate(tt_slab_size):
        plt.semilogy(tt_ref, err_E_tt[i], '.-', label=str(f"dt={tt_slab_size[i]}"))
        err_l2_E.append(dt * np.linalg.norm(err_E_tt[i]))
    plt.legend()

    from post_process_utils import compute_rate
    r = compute_rate(np.array(tt_slab_size), np.array(err_l2_E))
    rate_line = 1.e2*tt_slab_size**(r[-1])

    # Export tt_slab_size, err_l2_E, and rate_line as CSV
    with open("conv_energy_error.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tt_slab_size", "err_l2_E", "rate_line"])
        for dt, err, rate in zip(tt_slab_size, err_l2_E, rate_line):
            writer.writerow([dt, err, rate])

    plt.figure("L2 error energy")
    plt.loglog(tt_slab_size, err_l2_E, '.-')
    
    plt.loglog(tt_slab_size, rate_line, 'k-', label="dt**("+str(r[-1])+")")
    plt.title("L2-Error energy")
    plt.xlabel("dt")
    plt.legend()
    
    plt.show()
