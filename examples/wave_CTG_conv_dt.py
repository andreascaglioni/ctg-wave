"""Time step size refinement convergence test for CTG on a classical Wave Equation (WE).

The wave equation reads: u_tt -  Δu = f
and is equipped with initial and boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh
import sys

sys.path.insert(0, "./")
from CTG.brownian_motion import param_LC_W
from CTG.ctg_solver import CTGSolver
from CTG.error import compute_err
from CTG.post_process import float_f
from CTG.post_process import compute_rate


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

    # time
    start_time = 0.0
    end_time = 1.0
    order_t = 1
    ddt = 2.**(-np.arange(2, 7))  # time step sizes

    # Parameter
    y = np.random.standard_normal(100)
    W_t = lambda tt: param_LC_W(y, tt, T=end_time)[0]


    # Exact sol
    from data.data_wave_eq import (
        exact_sol_u,
        exact_sol_v,
        exact_rhs_0,
        exact_rhs_1,
        boundary_data_u,
        boundary_data_v,
        initial_data_u,
        initial_data_v,
    )

    numerics_params = {
        "comm": comm, 
        "V_x": V_x,
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

    print("CONVERNGENCE TEST")
    err = np.zeros_like(ddt)
    nn_dofs = np.zeros(ddt.size, dtype=int)
    for i, dt in enumerate(ddt):
        print("\nDT = ", dt)
        numerics_params["t_slab_size"] = ddt[i]

        # Compute
        ctg_solver = CTGSolver(numerics_params, verbose=False)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = ctg_solver.run(physics_params)
        
        # Error
        n_scalar = space_time_fe.n_dofs
        sol_u = [s[:n_scalar] for s in sol_slabs]
        sol_v = [s[n_scalar:] for s in sol_slabs]
        err_u, _, _, _ = compute_err(comm, order_t, "h1", "l2", time_slabs, space_time_fe.space_fe, sol_u, exact_sol_u)
        err_v, _, _, _ = compute_err(comm, order_t, "l2", "l2", time_slabs, space_time_fe.space_fe, sol_v, exact_sol_v)
        err[i] = err_u + err_v
        print("Total error", float_f(err[i]) )

    print("CONVERGENCE SUMMARY:")
    print("ddt", ddt)
    print("nn_dofs", nn_dofs)
    print("err", err)
    rr = compute_rate(ddt, err)  # rate of convergence: err = C * ddt**rr
    r = np.median(rr)
    C = err[0] / (ddt[0]**r)

    # Plot
    plt.figure()
    plt.loglog(ddt, err, 'o-', linewidth=2)
    plt.loglog(ddt, C * ddt**r, 'k-', label=f"dt**{r:.2f}")
    plt.xlabel("dt")
    plt.title("Convergence energy norm error CTG")
    plt.legend()
    plt.show()