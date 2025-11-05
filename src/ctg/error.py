"""Error computation helpers for CTG solutions.

The functions in this module provide utilities to compute L2/H1 and
temporal norms of the error by interpolating the numerical solution on a
refined space-time mesh and comparing it with the exact solution.
"""

import numpy as np
from ctg.FE_spaces import SpaceFE, TimeFE, SpaceTimeFE
from ctg.utils import cart_prod_coords


import scipy.sparse
from dolfinx import fem, mesh
from scipy.interpolate import griddata


from math import sqrt


def compute_error_slab(sol_slab, exact_sol, space_time_fe, err_type_x, err_type_t):
    """Compute spatial/temporal error for a single time slab.

    The numerical solution is interpolated on a refined space-time grid
    and compared to ``exact_sol``. Supported spatial norms are ``l2`` and
    ``h1``, and temporal measures are ``l2`` and ``linf``.

    Args:
        sol_slab: Flattened numerical solution for the slab.
        exact_sol: Callable returning exact solution values at query points.
        space_time_fe: Instance of :class:`SpaceTimeFE` for the slab.
        err_type_x: Spatial error type, either ``'l2'`` or ``'h1'``.
        err_type_t: Temporal error aggregation, either ``'l2'`` or ``'linf'``.

    Returns:
        Tuple ``(err, norm_u)`` where ``err`` is the chosen error metric and
        ``norm_u`` is the corresponding norm of the numerical solution.
    """

    # refine Time

    msh_t = space_time_fe.time_fe.V.mesh
    msh_t_ref = mesh.refine(msh_t)[0]
    p_t_trial = space_time_fe.time_fe.V.element.basix_element.degree
    V_t_ref = fem.functionspace(msh_t_ref, ("Lagrange", p_t_trial))
    time_fe_ref = TimeFE(V_t_ref)

    # refine Space
    msh_x = space_time_fe.space_fe.V.mesh
    msh_x_ref = mesh.refine(msh_x)[0]
    p_Space = space_time_fe.space_fe.V.element.basix_element.degree
    V_x_ref = fem.functionspace(msh_x_ref, ("Lagrange", p_Space))
    space_fe_ref = SpaceFE(V_x_ref)

    # Interpolate exact sol in fine space # TODO works only for Lagrangian FE
    fine_coords = cart_prod_coords(time_fe_ref.dofs, space_fe_ref.dofs)
    ex_sol_ref = exact_sol(fine_coords)

    # Interpolate numerical sol using griddata (linear interpolation)
    # TODO works only for Lagrangian FE
    coarse_coords = space_time_fe.dofs
    sol_slab_ref = griddata(coarse_coords, sol_slab, fine_coords, method="linear", fill_value=0.0)

    # Adapt to error type
    if err_type_x == "h1":
        ip_space_ref = space_fe_ref.matrix["mass"] + space_fe_ref.matrix["laplace"]
    elif err_type_x == "l2":
        ip_space_ref = space_fe_ref.matrix["mass"]
    else:
        raise ValueError(f"Unknown error type x: {err_type_x}")

    err_fun_ref = ex_sol_ref - sol_slab_ref

    if err_type_t == "l2":
        ip_tx = scipy.sparse.kron(time_fe_ref.matrix["mass_err"], ip_space_ref)
        err = sqrt((ip_tx @ err_fun_ref) @ err_fun_ref)  # ip_tx.dot(err_fun_ref).dot(err_fun_ref))
        norm_u = sqrt((ip_tx @ (sol_slab_ref)) @ (sol_slab_ref))
    elif err_type_t == "linf":
        err = -1.0
        norm_u = -1.0
        for i in range(time_fe_ref.n_dofs):
            coords_u_t = sol_slab_ref[i * space_fe_ref.n_dofs : (i + 1) * space_fe_ref.n_dofs]
            norm_u_t = sqrt(ip_space_ref.dot(coords_u_t).dot(coords_u_t))
            norm_u = max(norm_u, norm_u_t)
            coords_err_t = err_fun_ref[i * space_fe_ref.n_dofs : (i + 1) * space_fe_ref.n_dofs]
            err_t = sqrt(ip_space_ref.dot(coords_err_t).dot(coords_err_t))
            err = max(err, err_t)
    else:
        raise ValueError(f"Unknown error type t: {err_type_t}")
    return err, norm_u


def compute_err(comm, order_t, err_type_x, err_type_t, time_slabs, space_fe, sol_slabs, sol_exa):
    """Compute aggregated error over all time slabs.

    Args:
        comm: MPI communicator used to create temporal meshes.
        order_t: Temporal FE degree used for refinement in error estimates.
        err_type_x: Spatial error type ('l2' or 'h1').
        err_type_t: Temporal aggregation ('l2' or 'linf').
        time_slabs: List of slab tuples produced by the solver.
        space_fe: Spatial FE object used by the solver.
        sol_slabs: List of per-slab numerical solutions.
        sol_exa: Exact solution callable used for comparison.

    Returns:
        Tuple ``(total_err, total_rel_err, err_slabs, norm_u_slabs)``.
    """

    err_slabs = -1.0 * np.ones(len(time_slabs))
    norm_u_slabs = -1.0 * np.ones_like(err_slabs)
    for i, slab in enumerate(time_slabs):
        # Current time FE
        msh_t = mesh.create_interval(comm, 1, [slab[0], slab[1]])
        V_t = fem.functionspace(msh_t, ("Lagrange", order_t))
        time_fe = TimeFE(V_t)
        space_time_fe = SpaceTimeFE(space_fe, time_fe)
        X = sol_slabs[i]
        err_slabs[i], norm_u_slabs[i] = compute_error_slab(
            X, sol_exa, space_time_fe, err_type_x, err_type_t
        )

    if err_type_t == "linf":
        total_err = np.amax(err_slabs)
        total_norm_u = np.amax(norm_u_slabs)
    elif err_type_t == "l2":
        ddt = np.array([ts[1] - ts[0] for ts in time_slabs])
        total_err = sqrt(np.dot(ddt, np.square(err_slabs)))
        total_norm_u = sqrt(np.dot(ddt, np.square(norm_u_slabs)))
    else:
        raise ValueError(f"Unknown error type in time: {err_type_t}")
    total_rel_err = total_err / total_norm_u

    return total_err, total_rel_err, err_slabs, norm_u_slabs
