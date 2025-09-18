"""Functions for the Continuous Time Galerkin method for the space-time integration
of space-time problems, such as paraboic and hyperbolic PDEs."""

from math import sqrt
from dolfinx import fem, mesh
import numpy as np
import scipy.sparse
from scipy.interpolate import griddata
import sys
import matplotlib.pyplot as plt

sys.path.append("./")
from CTG.FE_spaces import SpaceFE, TimeFE
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


def compute_error_slab(sol_slab, exact_sol, space_fe, time_fe, err_type_x, err_type_t):

    # refine Time
    msh_t = time_fe.mesh
    msh_t_ref = mesh.refine(msh_t)[0]
    p_t_trial = time_fe.V_trial.element.basix_element.degree
    V_t_trial_ref = fem.functionspace(msh_t_ref, ("Lagrange", p_t_trial))
    p_t_test = time_fe.V_test.element.basix_element.degree
    V_t_test_ref = fem.functionspace(msh_t_ref, ("DG", p_t_test))
    time_fe_ref = TimeFE(msh_t_ref, V_t_trial_ref, V_t_test_ref)

    # refine Space
    msh_x = space_fe.mesh
    msh_x_ref = mesh.refine(msh_x)[0]
    p_Space = space_fe.V.element.basix_element.degree
    V_x_ref = fem.functionspace(msh_x_ref, ("Lagrange", p_Space))
    space_fe_ref = SpaceFE(V_x_ref)

    # Interpolate exact sol in fine space # TODO works only for Lagrangian FE
    fine_coords = cart_prod_coords(time_fe_ref.dofs_trial, space_fe_ref.dofs)
    ex_sol_ref = exact_sol(fine_coords)

    # Interpolate numerical sol using griddata (linear interpolation)
    # TODO works only for Lagrangian FE
    coarse_coords = cart_prod_coords(time_fe.dofs_trial, space_fe.dofs)
    sol_slab_ref = griddata(
        coarse_coords, sol_slab, fine_coords, method="linear", fill_value=0.0
    )

    # Adapt to error type
    if err_type_x == "h1":
        ip_space_ref = space_fe_ref.matrix["mass"] + space_fe_ref.matrix["laplace"]
    elif err_type_x == "l2":
        ip_space_ref = space_fe_ref.matrix["mass"]
    else:
        raise ValueError(f"Unknown error type x: {err_type_x}")

    err_fun_ref = ex_sol_ref - sol_slab_ref

    if err_type_t == "l2":
        ip_tx = scipy.sparse.kron(time_fe_ref.matrix["mass"], ip_space_ref)
        err = sqrt(ip_tx.dot(err_fun_ref).dot(err_fun_ref))
        norm_u = sqrt(ip_tx.dot(ex_sol_ref).dot(ex_sol_ref))
    elif err_type_t == "linf":
        err = -1.0
        norm_u = -1.0
        for i, t in enumerate(time_fe.dofs_trial):
            coords_u_t = ex_sol_ref[i * space_fe_ref.n_dofs : (i + 1) * space_fe_ref.n_dofs]
            norm_u_t = sqrt(ip_space_ref.dot(coords_u_t).dot(coords_u_t))
            norm_u = max(norm_u, norm_u_t)

            coords_err_t = err_fun_ref[i * space_fe_ref.n_dofs : (i + 1) * space_fe_ref.n_dofs]
            err_t = sqrt(ip_space_ref.dot(coords_err_t).dot(coords_err_t))
            err = max(err, err_t)
    else:
        raise ValueError(f"Unknown error type t: {err_type_t}")
    return err, norm_u


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

    if exact_sol_u is not None:
        plt.plot(space_fe.dofs, exact_sol_v(tx_final), ":", label="v exact")

    plt.title(f"u and v at final time t={round(time_slabs[-1][1], 4)}")
    plt.legend()
    plt.tight_layout()
    
    

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
    tx = cart_prod_coords(np.array([slab[0]]), space_fe.dofs)

    plt.figure(figsize=(10, 4))
    for i, slab in enumerate(time_slabs):
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
    plt.show()

def plot_error_tt(time_slabs, err_slabs, norm_u_slabs):
    times = [slab[1] for slab in time_slabs]
    rel_errs = err_slabs / norm_u_slabs
    plt.figure()
    plt.plot(times, err_slabs, marker='o', label="error")
    plt.plot(times, rel_errs, marker='o', label="relative error")
    plt.xlabel("Time")
    plt.title("Error over time")
    plt.legend()

"""
Parametric expansions of the Wiener process.

This module provides functions to construct the Wiener process using
either a Levy-Ciesielski (LC) or Karhunen-Loeve (KL) expansion.

Functions:
    ``param_LC_W(tt, yy, T)``: Construct Wiener process using LC expansion.
    ``param_KL_Brownian_motion(tt, yy)``: Construct Wiener process using KL expansion.
"""

from math import ceil, log, sqrt, pi
import warnings
import numpy as np


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