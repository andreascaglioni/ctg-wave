import matplotlib.pyplot as plt
import numpy as np
from ctg.FE_spaces import SpaceFE
from ctg.utils import cart_prod_coords


def plot_on_slab(dofs_x, dofs_t, X):
    A, B = np.meshgrid(dofs_t, dofs_x, indexing="ij")
    A_flat = A.flatten()
    B_flat = B.flatten()
    C_flat = X.flatten()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(A_flat, B_flat, C_flat, cmap="viridis", edgecolor="none")
    ax.set_xlabel("Time (a)")
    ax.set_ylabel("Space (b)")
    ax.set_zlabel("Value (c)")
    plt.tight_layout()
    plt.show()


def plot_error_tt(time_slabs, err_slabs, norm_u_slabs):
    times = [slab[1] for slab in time_slabs]
    rel_errs = err_slabs / norm_u_slabs
    plt.figure()
    plt.plot(times, err_slabs, marker="o", label="error")
    plt.plot(times, rel_errs, marker="o", label="relative error")
    plt.xlabel("Time")
    plt.title("Error over time")
    plt.legend()


def plot_uv_tt(time_slabs, space_fe, sol_slabs, exact_sol_u=None, exact_sol_v=None):
    n_x = space_fe.n_dofs
    assert sol_slabs[0].size % 2 == 0, "sol_slabs[0].size must be even, got {}".format(
        sol_slabs[0].size
    )
    n_dofs_scalar = int(sol_slabs[0].size / 2)

    # Compute bounds y axis
    uu = np.array([X[0:n_dofs_scalar] for X in sol_slabs])
    umin = np.amin(uu)
    umax = np.amax(uu)
    vv = np.array([X[n_dofs_scalar:] for X in sol_slabs])
    vmin = np.amin(vv)
    vmax = np.amax(vv)

    plt.figure(figsize=(10, 4))
    for i, slab in enumerate(time_slabs):
        tx = cart_prod_coords(np.array([slab[0]]), space_fe.dofs)
        X = sol_slabs[i]
        plt.clf()

        # Plot u on the left subplot
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(space_fe.dofs, X[0:n_x], ".", label=f"u at t={round(slab[0], 4)}")
        if exact_sol_u is not None:
            ax1.plot(space_fe.dofs, exact_sol_u(tx), "-", label="u exact")
        ax1.set_title(f"u at t={round(slab[0], 4)}")
        ax1.legend()
        ax1.set_ylim((umin, umax))

        # Plot v on the right subplot
        ax2 = plt.subplot(1, 2, 2)
        vv = X[n_dofs_scalar : n_dofs_scalar + n_x]
        ax2.plot(space_fe.dofs, vv, ".", label=f"v at t={round(slab[0], 4)}")
        if exact_sol_v is not None:
            ax2.plot(space_fe.dofs, exact_sol_v(tx), "-", label="v exact")
        ax2.set_title(f"v at t={round(slab[0], 4)}")
        ax2.legend()
        ax2.set_ylim((vmin, vmax))
        plt.tight_layout()

        plt.pause(0.1)


def compute_rate(xx, yy):
    """
    Compute the logarithmic rate of change between consecutive elements of two arrays.

    Args:
        xx (numpy.ndarray): 1D array of x-coordinates.
        yy (numpy.ndarray): 1D array of y-coordinates.

    Returns:
        numpy.ndarray: Logarithmic rates of change.
    """

    return np.log(yy[1:] / yy[:-1]) / np.log(xx[1:] / xx[:-1])


def float_f(x):
    """
    Format a float variable in scientific notation.

    Args:
        x (float): Input float.

    Returns:
        str: Formatted float as a string.
    """
    return f"{x:.4e}"


def plot_xt_slabs(space_fe: SpaceFE, time_slabs):

    text_size = 14

    min_x = np.amin(space_fe.dofs)
    max_x = np.amax(space_fe.dofs)
    tt = [slab[0] for slab in time_slabs]
    tt.append(time_slabs[-1][1])
    min_t = np.amin(tt)
    max_t = np.amax(tt)

    plt.figure("xt_slabs")
    ax = plt.gca()
    ax.set_aspect(2)
    # Move y-axis to the right and put ticks/labels inside the plot
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    # Show tick labels on the right only, draw ticks inward (into axes)
    ax.tick_params(axis="y", which="both", direction="in", labelright=True, labelleft=False)
    # Draw space-time slabs
    for x in [min_x, max_x]:
        ax.plot([x, x], [min_t, max_t], linewidth=2, color="black")
    for i, t in enumerate(tt):
        ax.plot([min_x, max_x], [t, t], linewidth=2, color="black")

    # Add annotations
    for i, t in enumerate(tt[:-1]):
        x_c = (min_x + max_x) / 2
        t_c = (tt[i] + tt[i + 1]) / 2
        ax.text(
            x_c,
            t_c,
            f"Slab {i}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=text_size,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    ax.set_xlabel("Space domain D", fontsize=text_size + 4)
    ax.set_ylabel("Time domain", fontsize=text_size + 4, labelpad=-10)
    ax.set_xticks([])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels(["0", "T"], fontsize=text_size + 4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
