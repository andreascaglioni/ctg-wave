import numpy as np
from CTG.FE_spaces import SpaceFE, TimeFE
from CTG.utils import cart_prod_coords


import scipy.sparse
from dolfinx import fem, mesh
from scipy.interpolate import griddata


from math import sqrt


def compute_error_slab(sol_slab, exact_sol, space_fe, time_fe, err_type_x, err_type_t):

    # refine Time
    msh_t = time_fe.mesh
    msh_t_ref = mesh.refine(msh_t)[0]
    p_t_trial = time_fe.V_trial.element.basix_element.degree
    V_t_ref = fem.functionspace(msh_t_ref, ("Lagrange", p_t_trial))
    
    time_fe_ref = TimeFE(msh_t_ref, V_t_ref)

    # refine Space
    msh_x = space_fe.mesh
    msh_x_ref = mesh.refine(msh_x)[0]
    p_Space = space_fe.V.element.basix_element.degree
    V_x_ref = fem.functionspace(msh_x_ref, ("Lagrange", p_Space))
    space_fe_ref = SpaceFE(V_x_ref)

    # Interpolate exact sol in fine space # TODO works only for Lagrangian FE
    fine_coords = cart_prod_coords(time_fe_ref.dofs, space_fe_ref.dofs)
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


def compute_err(comm, order_t, err_type_x, err_type_t, time_slabs, space_fe, sol_slabs, sol_slabs_ref, space_fe_ref = None, order_t_ref = None, time_slabs_ref = None):
    
    # exact_sol_u can be either Callable (analytic) or fem.Function (reference)
    if type(sol_slabs_ref) is fem.Function:
        assert (space_fe_ref is not None) and (order_t_ref is not None) and (time_slabs_ref is not None)
        raise NotImplementedError("Reference solution as fem.Function is not implemented yet.")

    err_slabs = -1.0 * np.ones(len(time_slabs))
    norm_u_slabs = -1.0 * np.ones_like(err_slabs)
    for i, slab in enumerate(time_slabs):
        # Furrent time FE
        msh_t = mesh.create_interval(comm, 1, [slab[0], slab[1]])
        V_t_trial = fem.functionspace(msh_t, ("Lagrange", order_t))
        time_fe = TimeFE(msh_t, V_t_trial)

        # Extract u curr slab
        X = sol_slabs[i]
        u = X[:int(X.size/2)]

        # Get dofs sol_slab_ref for current slab
        if type(sol_slabs_ref) is fem.Function:
            X = sol_slabs[i]
        u = X[:int(X.size/2)]

        err_slabs[i], norm_u_slabs[i] = compute_error_slab(u, sol_slabs_ref, space_fe, time_fe, err_type_x, err_type_t)


    if err_type_t == "linf":
        total_err = np.amax(err_slabs)
        total_norm_u = np.amax(norm_u_slabs)
    elif err_type_t == "l2":
        ddt = np.array([ts[1]-ts[0] for ts in time_slabs])
        total_err = sqrt(np.dot(ddt, np.square(err_slabs)))
        total_norm_u = sqrt(np.dot(ddt, np.square(norm_u_slabs)))
    else:
        raise ValueError(f"Unknown error type in time: {err_type_t}")
    total_rel_err = total_err / total_norm_u
    return total_err, total_rel_err, err_slabs, norm_u_slabs