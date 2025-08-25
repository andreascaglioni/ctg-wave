"""Functions for the Continuous Time Galerkin method for the space-time integration
of space-time problems, such as paraboic and hyperbolic PDEs."""

from math import sqrt
from dolfinx import fem, mesh
import numpy as np
import scipy.sparse
from scipy.interpolate import griddata
import sys

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


def compute_error_slab(
    space_fe, exact_sol, err_type_x, err_type_t, time_fe, sol_slab
):

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
    space_fe_ref = SpaceFE(msh_x_ref, V_x_ref)

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
        norm_u = sqrt(ip_tx.dot(sol_slab_ref).dot(sol_slab_ref))
    elif err_type_t == "linf":
        err = -1.0
        norm_u = -1.0
        for i, t in enumerate(time_fe.dofs_trial):
            coords_u_t = sol_slab_ref[i * space_fe_ref.n_dofs : (i + 1) * space_fe_ref.n_dofs]
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
