"""CTG approximation heat equation. Convergence test in t or x

"""

import sys
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from dolfinx import fem, mesh

sys.path.append("./")
from CTG.utils import compute_time_slabs, float_f, compute_rate
from CTG.ctg_parabolic import run_CTG_parabolic
from CTG.FE_spaces import SpaceFE



if __name__ == "__main__":
    # SETTINGS
    np.set_printoptions(formatter={"float_kind": float_f})
    comm = MPI.COMM_SELF

    # DATA
    # Physical
    start_time = 0.0
    end_time = 1.0
    boundary_D = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))  # noqa: E731
    from data.exact_solution_heat_2 import (
        exact_rhs,
        boundary_data,
        initial_data,
        exact_sol,
    )

    # Numerics data
    n_refs = 7  # number of refinement

    # ---------------------- Ex 1: Space mesh refinement --------------------- #
    nn_x = 2 ** np.arange(n_refs, dtype=int)  # refine mesh shape
    pp_x = np.ones(n_refs, dtype=int)  # polynomial degree in space
    tt_slab_size = end_time * np.ones(n_refs)
    pp_t = np.ones(n_refs, dtype=int)
    nn_t = 300 * np.ones(n_refs, dtype=int)  # number of t elements per time-slab

    # ---------------------- Ex. 2: Time mesh refinement --------------------- #
    # nn_x = 400 * np.ones(n_refs, dtype=int)
    # pp_x = np.ones(n_refs, dtype=int)
    # tt_slab_size = end_time * np.ones(n_refs)
    # pp_t = np.ones(n_refs, dtype=int)
    # nn_t = 2 ** np.arange(n_refs, dtype=int)  # refine mesh time

    # ------------------------------------------------------------------------ #
    #                             CONVERGENCE TEST                             #
    # ------------------------------------------------------------------------ #
    ee = np.zeros((n_refs))
    rre = np.zeros_like(ee)
    nn_dofs = np.zeros(n_refs, dtype=int)

    for n_exp in range(n_refs):
        print("Refinement", n_exp, flush=True)
        n_x = nn_x[n_exp]
        p_x = pp_x[n_exp]

        t_slab_size = tt_slab_size[n_exp]
        p_t = pp_t[n_exp]
        n_t = nn_t[n_exp]

        print("n_x:", n_x, "p_x:", p_x)
        print("t_slab_size:", t_slab_size, "p_t:", p_t, "n_t:", n_t)

        # -------------------------------------------------------------------- #
        #                                  COMPUTE                             #
        # -------------------------------------------------------------------- #
        msh_x = mesh.create_unit_interval(comm, n_x)
        V_x = fem.functionspace(msh_x, ("Lagrange", p_x))
        time_slabs = compute_time_slabs(start_time, end_time, t_slab_size)

        # FE object for SPACE discretization
        Space = SpaceFE(msh_x, V_x, boundary_data, boundary_D)

        # Time marching
        _, err_slabs, norm_slabs, nn_dofs[n_exp] = run_CTG_parabolic(
            comm,
            Space,
            n_t,
            p_t,
            time_slabs,
            boundary_data,
            exact_rhs,
            initial_data,
            exact_sol,
            err_type_x="h1",
            err_type_t="linf",
            verbose=False,
        )

        # Current-iteration post-processing
        ee[n_exp] = sqrt(np.sum(np.square(err_slabs)))
        rre[n_exp] = ee[n_exp] / sqrt(np.sum(np.square(norm_slabs)))
        print(
            "Total error",
            float_f(ee[n_exp]),
            "Total relative error",
            float_f(rre[n_exp]),
            "\n",
        )

    # ------------------------------------------------------------------------ #
    #                               POST-PROCESS                               #
    # ------------------------------------------------------------------------ #

    hh = 1 / nn_x
    ddt = tt_slab_size / nn_t
    xx = hh

    # Compute rate
    if xx.size > 1:
        rr = compute_rate(xx, rre)
        r = rr[-1]
        C = rre[0] / (xx[0] ** r)

    # Print
    print("Numbed of dofs", nn_dofs)
    print("hh", hh)
    print("ddt", ddt)
    print("Error", ee)
    print("Rel. err.", rre)
    print("Convergence rate rel. err.:", rr)

    # X refinement plot
    plt.figure()
    plt.loglog(xx, rre, marker="s", label="Relative Error")
    if xx.size > 1:
        plt.loglog(xx, C * xx**r, "k-", label=f"x^{r:.4g}")
    plt.xlabel("h")
    plt.legend()
    plt.title("Error vs mesh spacing (log-log scale)")
    plt.grid(True, which="both", ls="--")
    plt.show()

    # T refinement plot
    # plt.figure()
    # plt.loglog(xx, rre, marker="s", label="Relative Error")
    # if xx.size>1:
    #     plt.loglog(xx, C * xx**r, "k-", label=f"x^{r:.4g}")
    # plt.xlabel("dt")
    # plt.legend()
    # plt.title("Error vs # t-dofs (log-log scale)")
    # plt.grid(True, which="both", ls="--")
    # plt.show()
