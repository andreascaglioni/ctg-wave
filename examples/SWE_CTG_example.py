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

from pathlib import Path
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
from ctg.brownian_motion import param_LC_W
from ctg.post_process import float_f, compute_energy_tt, inverse_DS_transform
from ctg.ctg_solver import CTGSolver
from ctg.config import load_config


def main():
    # SETTINGS
    np.set_printoptions(formatter={"float_kind": float_f})

    # PARAMETERS
    # Import data from Config
    data_dir = Path(__file__).parent.parent / "data"
    data_file = data_dir / "data_pwe.yaml"
    config = load_config(data_file)

    print("COMPUTE")
    np.random.seed(config.numerics.seed)
    y = np.random.standard_normal(100)

    def W_t(tt):
        return param_LC_W(y, tt, T=config.physics.end_time)[0]

    ctg_solver = CTGSolver(config.numerics)
    sol_slabs, time_slabs, space_time_fe, total_n_dofs = ctg_solver.run(config.physics, W_t)

    # Inverse Doss-Sussmann transform (reover solution SWE)
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
    dir_save = os.path.join("results", "result_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(dir_save)

    space_fe = space_time_fe.space_fe
    dofs = space_fe.dofs.flatten()
    n_x = space_fe.n_dofs
    n_scalar = int(sol_slabs[0].size / 2)
    XX_T = XX[-1]
    U_T = XX_T[n_scalar - n_x : n_scalar]
    V_T = XX_T[-n_x:]
    s_t = config.physics.start_time
    e_t = config.physics.end_time
    tt = np.linspace(s_t, e_t, ceil((e_t - s_t) / config.numerics.t_slab_size))

    print("Total number of DOFs:", total_n_dofs)

    # Compute metrics
    WW = W_t(tt)
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
    plt.figure("broanian")
    plt.plot(tt, WW, ".-")
    plt.title("Browniann motion sample path")
    plt.tight_layout()
    plt.xlabel("t")

    plt.figure("energy_vs_tt")
    plt.plot(tt, EE, ".-", label="Total")
    plt.plot(tt, ppot, ".-", label="Potential")
    plt.plot(tt, kkin, ".-", label="Kinetic")
    plt.title("Energy of PWE sample")
    plt.tight_layout()
    plt.legend()
    plt.xlabel("t")

    # Solution at final time
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
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
    plt.figure("time_track")
    n_dof_track = int(space_fe.n_dofs / 4)
    U_T = np.array([X[n_dof_track] for X in XX])
    plt.plot(tt, U_T, ".-")
    plt.title("U(t, x0) vs t for x0 = " + str(space_fe.dofs[n_dof_track]))
    plt.xlabel("t")
    plt.tight_layout()

    # Animation of oslution over time steps
    # plot_uv_tt(time_slabs, space_fe, sol_slabs)

    # 3D plot u space-time
    plt.figure("xt_plot_u")
    X, Y = np.meshgrid(space_fe.dofs, tt)
    UU = np.array([X[: space_fe.n_dofs] for X in XX])
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, UU, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title("Solution U of the SWE")

    plt.show()


if __name__ == "__main__":
    main()
