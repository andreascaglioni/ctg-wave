"""Compute an ensamble of solutions of the PWE for parameters sampled as a high-dimensional standard Gaussian random variable. Plot quantites of interest related to time.
"""

from pathlib import Path
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from ctg.brownian_motion import param_LC_W
from ctg.ctg_solver import CTGSolver
from ctg.post_process import float_f, compute_energy_tt, inverse_DS_transform
from ctg.config import load_config


if __name__ == "__main__":
    # SETTINGS
    np.set_printoptions(formatter={"float_kind": float_f})

    # PARAMETERS
    # Import data from Config
    data_dir = Path(__file__).parent.parent / "data"
    data_file = data_dir / "data_pwe.yaml"
    config = load_config(data_file)

    s_t = config.physics.start_time
    e_t = config.physics.end_time
    dt = config.numerics.t_slab_size
    tt = np.linspace(s_t, e_t, ceil(1 / dt))
    n_samples = 10

    print(f"COMPUTE {n_samples} SAMPLES")
    for seed in range(n_samples):
        print("Sample", seed + 1)

        np.random.seed(seed)
        y = np.random.standard_normal(100)

        def W_t(tt):
            return param_LC_W(y, tt, T=config.physics.end_time)[0]

        print("RUN CTG")
        ctg_solver = CTGSolver(config.numerics)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = ctg_solver.run(config.physics, W_t)

        XX = [
            inverse_DS_transform(
                sol_slabs[i],
                W_t,
                space_time_fe.space_fe,
                time_slabs[i],
                config.numerics.comm,
                config.numerics.order_t,
            )
            for i in range(len(sol_slabs))
        ]

        print("POST PROCESS")
        space_fe = space_time_fe.space_fe
        EE, _, _ = compute_energy_tt(space_fe, XX)

        # Plots
        plt.figure("W")
        plt.plot(tt, W_t(tt), "-", linewidth=2)

        plt.figure("energy")
        plt.plot(tt, EE, "-", linewidth=2)

        # Plot U(t,x0) for fixed x0, as finction of t
        n_dof_track = int(space_fe.n_dofs / 4)
        plt.figure("track")
        U_t = np.array([X[n_dof_track] for X in XX])
        plt.plot(tt, U_t, "-", linewidth=2)

    plt.figure("energy")
    plt.title("Energy (kinetic + potential) of sample paths")
    plt.tight_layout()
    plt.xlabel("t")

    plt.figure("track")
    plt.title("Value uf u(t, x0) for x0 = " + str(space_fe.dofs[n_dof_track]))
    plt.tight_layout()
    plt.xlabel("t")

    plt.figure("W")
    plt.title("Browniann motion sample path")
    plt.tight_layout()
    plt.xlabel("t")
    plt.show()
