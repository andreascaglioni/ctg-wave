"""Functions for the Continuous Time Galerkin method for the space-time integration
of space-time problems, such as paraboic and hyperbolic PDEs."""

from math import sqrt, ceil, log, sqrt, pi
import numpy as np
import sys
import matplotlib.pyplot as plt
import warnings

sys.path.append("./")
import copy


def cart_prod_coords(t_coords, x_coords):
    if len(x_coords.shape) == 1:  # coordinates i wrong format (rank 1 array). assume 1d.
        x_coords = np.expand_dims(x_coords, 1)
    if len(t_coords.shape) == 1:  # t coords in wron format assume 1d
        t_coords = np.expand_dims(t_coords, 1)
        
    long_t_coords = np.kron(t_coords, np.ones((x_coords.shape[0], 1)))
    long_x_coords = np.kron(np.ones((t_coords.shape[0], 1)), x_coords)
    return np.hstack((long_t_coords, long_x_coords))


def compute_time_slabs(start_time, end_time, slab_size):
    time_slabs = [(start_time, start_time + slab_size)]  # current time interval
    while time_slabs[-1][1] < end_time - 1e-10:
        time_slabs.append((time_slabs[-1][1], time_slabs[-1][1] + slab_size))
    return time_slabs


def float_f(x):
    return f"{x:.4e}"


def compute_rate(xx, yy):
    return np.log(yy[1:] / yy[:-1]) / np.log(xx[1:] / xx[:-1])


def plot_uv_at_T(time_slabs, space_fe, sol_slabs, exact_sol_u=None, exact_sol_v=None):
    n_x = space_fe.n_dofs
    assert sol_slabs[0].size % 2 == 0, "sol_slabs[0].size must be even, got {}".format(sol_slabs[0].size)
    n_dofs_scalar = int(sol_slabs[0].size / 2)
    
    X_final = sol_slabs[-1]
    tx_final = cart_prod_coords(np.array([time_slabs[-1][1]]), space_fe.dofs)

    plt.figure(figsize=(8, 5))
    
    u = X_final[n_dofs_scalar-n_x:n_dofs_scalar]
    plt.plot(space_fe.dofs, u, "o-", label="u numerical")

    if exact_sol_u is not None:
        plt.plot(space_fe.dofs, exact_sol_u(tx_final), "--", label="u exact")
    
    v = X_final[-n_x:]
    plt.plot(space_fe.dofs, v, "s-", label="v numerical")

    if exact_sol_v is not None:
        plt.plot(space_fe.dofs, exact_sol_v(tx_final), ":", label="v exact")

    plt.title(f"u and v at final time t={round(time_slabs[-1][1], 4)}")
    plt.legend()
    plt.tight_layout()
    return u, v
    
    

def plot_uv_tt(time_slabs, space_fe, sol_slabs, exact_sol_u=None, exact_sol_v=None):
    n_x = space_fe.n_dofs
    assert sol_slabs[0].size % 2 == 0, "sol_slabs[0].size must be even, got {}".format(sol_slabs[0].size)
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
        vv = X[n_dofs_scalar:n_dofs_scalar+n_x]
        ax2.plot(space_fe.dofs, vv, ".", label=f"v at t={round(slab[0], 4)}")
        if exact_sol_v is not None:
            ax2.plot(space_fe.dofs, exact_sol_v(tx), "-", label="v exact")
        ax2.set_title(f"v at t={round(slab[0], 4)}")
        ax2.legend()
        ax2.set_ylim((vmin, vmax))
        plt.tight_layout()
        plt.pause(0.05)

def plot_error_tt(time_slabs, err_slabs, norm_u_slabs):
    times = [slab[1] for slab in time_slabs]
    rel_errs = err_slabs / norm_u_slabs
    plt.figure()
    plt.plot(times, err_slabs, marker='o', label="error")
    plt.plot(times, rel_errs, marker='o', label="relative error")
    plt.xlabel("Time")
    plt.title("Error over time")
    plt.legend()

def plot_energy_tt(space_fe, sol_slabs, tt):
    M =space_fe.matrix["mass"]  # mass
    A = space_fe.matrix["laplace"]  # stiffness
    n_x = space_fe.n_dofs
    n_scalar = int(sol_slabs[0].size/2)
    EE = np.zeros(tt.size)
    for i, t, in enumerate(tt):
        X = sol_slabs[i]
        u = X[0:n_x]
        v = X[n_scalar:n_scalar+n_x]
        EE[i] = v @ M @ v + u @ A @ u  # np.dot(v, np.dot(M, v)) + np.dot(u, np.dot(A, u)) # + potential
    plt.figure()
    plt.plot(tt, EE, '.-')
    plt.title("Energy (kinetic + potential) of PWE sample")
    plt.tight_layout()
    plt.xlabel(t)

    return EE





def param_LC_W(yy, tt, T):
    """Pythonic computation of the LC expansion of the Wiener process.

    Args:
        yy (numpy.ndarray[float]): Parameter vector for the expansion.
        tt (numpy.ndarray[float]): 1D array of discrete times in [0, T].
        T (float): Final time of approximation.

    Returns:
        numpy.ndarray[float]: 2D array. Each *ROW* is a sample path of W over tt.
    """

    # Check input shape and make it 2D
    if yy.size == 1 or yy is int:  # 1-element array
        yy = np.array([yy], dtype=float).reshape((1, 1))
    if len(yy.shape) == 1:  # 1 parameter vector
        yy = np.array([yy], dtype=float).reshape((1, yy.size))
    assert len(yy.shape) == 2, (
        "param_LC_Brownian_motion: yy must be 2D (1 ROW per sample array)"
    )

    if tt is int:
        tt = np.array([tt])
    tt = tt.flatten()
    tol = 1e-12
    assert np.amin(tt) >= -tol and np.amax(tt) <= T + tol, (
        "param_LC_Brownian_motion: tt not within [0,T] (with tolerance)"
    )


    # Get # LC-levels
    L = ceil(log(yy.shape[1], 2))  # levels

    # Extend yy to the next power of 2
    fill = np.zeros((yy.shape[0], 2**L - yy.shape[1]))
    yy = np.column_stack((yy, fill))

    (n_y, dim_y) = yy.shape
    n_t = tt.size

    # Rescale tt (to be reverted!)
    tt = tt / T

    # Compute basis B
    BB = np.zeros((dim_y, n_t))

    BB[0, :] = tt  # first basis function is the linear one
    for lev in range(1, L + 1):
        n_j = 2 ** (lev - 1)  # number of basis functions at level l
        for j in range(1, n_j + 1):
            basis_fun = 0 * tt  # basis is 0 where not assegned below

            # define increasing part basis function
            ran1 = np.where(
                (tt >= (2 * j - 2) / (2**lev)) & (tt <= (2 * j - 1) / (2**lev))
            )
            basis_fun[ran1] = tt[ran1] - (2 * j - 2) / 2**lev

            # define decreasing part basis function
            ran2 = np.where((tt >= (2 * j - 1) / (2**lev)) & (tt <= (2 * j) / (2**lev)))
            basis_fun[ran2] = -tt[ran2] + (2 * j) / 2**lev

            n_b = 2 ** (lev - 1) + j - 1  # prev. lev.s (complete) + curr. lev (partial)
            BB[n_b, :] = 2 ** ((lev - 1) / 2) * basis_fun

    W = np.matmul(yy, BB)

    # Revert rescaling
    W = W * sqrt(T)

    return W