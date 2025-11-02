"""Time step size refinement convergence test for CTG on a classical Wave Equation (WE).

The wave equation reads: u_tt -  Δu = f
and is equipped with initial and boundary conditions.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

from ctg.plotting import compute_rate, float_f

sys.path.insert(0, "./")
from ctg.ctg_solver import CTGSolver
from ctg.error import compute_err
from ctg.post_process import compute_energy_tt
from ctg.config import load_config


if __name__ == "__main__":
    # SETTINGS
    np.set_printoptions(formatter={"float_kind": float_f})

    # PARAMETERS
    data_file = Path("examples/data_examples/data_we.yaml")
    cfg = load_config(data_file)

    ddt = 2.0 ** (-np.arange(2, 7))  # time step sizes

    # error
    err_type_x = "h1"
    err_type_t = "l2"

    print("CONVERNGENCE TEST")
    err = np.zeros_like(ddt)
    nn_dofs = np.zeros(ddt.size, dtype=int)
    EE = []
    for i, dt in enumerate(ddt):
        print("\nDT = ", dt)
        # numerics_params["t_slab_size"] = ddt[i]
        cfg.numerics.t_slab_size = ddt[i]

        # Compute
        ctg_solver = CTGSolver(cfg.numerics)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = ctg_solver.run(cfg.physics)

        # Error
        n_scalar = space_time_fe.n_dofs
        sol_u = [s[:n_scalar] for s in sol_slabs]
        sol_v = [s[n_scalar:] for s in sol_slabs]
        err_u, _, _, _ = compute_err(
            cfg.numerics.comm,
            cfg.numerics.order_t,
            "h1",
            "l2",
            time_slabs,
            space_time_fe.space_fe,
            sol_u,
            cfg.physics.exact_sol_u,
        )
        err_v, _, _, _ = compute_err(
            cfg.numerics.comm,
            cfg.numerics.order_t,
            "l2",
            "l2",
            time_slabs,
            space_time_fe.space_fe,
            sol_v,
            cfg.physics.exact_sol_v,
        )
        err[i] = err_u + err_v
        print("Total error", float_f(err[i]))

        E, _, _ = compute_energy_tt(space_time_fe.space_fe, sol_slabs)
        EE.append(E)

    print("CONVERGENCE SUMMARY:")
    print("ddt", ddt)
    print("nn_dofs", nn_dofs)
    print("err", err)
    rr = compute_rate(ddt, err)  # rate of convergence: err = C * ddt**rr
    r = np.median(rr)
    C = err[0] / (ddt[0] ** r)

    # Plot
    plt.figure()
    plt.loglog(ddt, err, "o-", linewidth=2)
    plt.loglog(ddt, C * ddt**r, "k-", label=f"dt**{r:.2f}")
    plt.xlabel("dt")
    plt.title("Convergence energy norm error CTG")
    plt.legend()

    plt.figure()
    for i, dt in enumerate(ddt):
        tt = np.linspace(cfg.physics.start_time, cfg.physics.end_time, int(1 / dt))
        color_intensity = (i + 1) / len(ddt)
        col = plt.get_cmap("Blues")(0.3 + 0.7 * color_intensity)
        plt.plot(tt, EE[i], "o-", linewidth=2, label="dt = " + str(dt), color=col)
    plt.legend()
    plt.ylim(np.min([np.min(E) for E in EE]), np.max([np.max(E) for E in EE]))
    plt.xlabel("Time")
    plt.title("Total energy (kinetic + potential) for different dt")
    plt.show()
