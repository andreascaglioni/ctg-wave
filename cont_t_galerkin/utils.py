"""Functions for the Continuous Time Galerkin method for the space-time integration
of space-time problems, such as paraboic and hyperbolic PDEs."""

from math import sqrt
from dolfinx import fem, mesh
import numpy as np
import scipy.sparse
from utils import float_f
from scipy.interpolate import griddata
import sys
sys.path.append("./")
from cont_t_galerkin.FE_spaces import SpaceFE, TimeFE


def cart_prod_coords(t_coords, x_coords):
    if len(x_coords.shape) == 1:
        x_coords = np.expand_dims(x_coords, 1)
    long_t_coords = np.kron(t_coords, np.ones((x_coords.shape[0], 1)))
    long_x_coords = np.kron(np.ones((t_coords.shape[0], 1)), x_coords)
    return np.hstack((long_t_coords, long_x_coords))





def compute_time_slabs(start_time, end_time, slab_size):
    time_slabs = [(start_time, start_time + slab_size)]  # current time interval
    while time_slabs[-1][1] < end_time - 1e-10:
        time_slabs.append((time_slabs[-1][1], time_slabs[-1][1] + slab_size))
    return time_slabs


def assemble_ctg_slab(Space, u0, Time, exact_rhs, boundary_data):
    # Assemble space-time matrices (linear PDEs -> Kronecker product t & x matrices)
    mass_matrix = scipy.sparse.kron(Time.matrix["mass"], Space.matrix["mass"])
    stiffness_matrix = scipy.sparse.kron(Time.matrix["mass"], Space.matrix["laplace"])
    system_matrix = (
        scipy.sparse.kron(Time.matrix["derivative"], Space.matrix["mass"])
        + stiffness_matrix
    )

    # Assemble RHS vector as space-time mass * RHS on dofs
    # TODO better to use projection? I'll use higher order FEM!
    space_time_coords = cart_prod_coords(Time.dofs, Space.dofs)
    rhs = mass_matrix.dot(exact_rhs(space_time_coords))

    # Impose initial condition strongly
    dofs_at_t0 = np.zeros((Time.n_dofs * Space.n_dofs))  # indicator dofs at t_0
    dofs_at_t0[: Space.n_dofs] = 1.0

    system_matrix = system_matrix.multiply((1.0 - dofs_at_t0).reshape(-1, 1))
    system_matrix += scipy.sparse.diags(dofs_at_t0)
    rhs[: Space.n_dofs] = u0

    # Impose boundary conditions
    # Idea: modify A and RHS so that, if the i-th dof belongs to the boundary, then the i-th equation enforces the BC instead of the equation wrt the i-th test function. This means that:
    # * the i-th row of A becomes delta_{i,j}
    # * the i-th entry of RHS becomes i-th coordinate of BC

    # 1. recover data
    dofs_boundary = np.kron(
        np.ones((Time.dofs.shape[0], 1)), Space.boundary_dof_vector.reshape(-1, 1)
    ).flatten()
    bc_curr_slab = boundary_data(space_time_coords)

    # 2. Edit system matrix
    system_matrix = system_matrix.multiply(
        (1.0 - dofs_boundary).reshape(-1, 1)
    )  # put to 0 entries corresponding to boundary
    system_matrix += scipy.sparse.diags(dofs_boundary)

    # 3. Edit RHS vector
    rhs = rhs * (1.0 - dofs_boundary)
    rhs += bc_curr_slab * dofs_boundary

    return system_matrix, mass_matrix, stiffness_matrix, rhs, bc_curr_slab


def run_CTG_parabolic(
    comm,
    space_fe,
    n_time,
    order_t,
    time_slabs,
    boundary_data,
    exact_rhs,
    initial_data,
    exact_sol=None,
    err_type_x="h1",
    err_type_t="l2",
    verbose=False,
):

    # coordinates initial condition wrt space-time basis
    init_time = time_slabs[0][0]
    u0 = initial_data(cart_prod_coords(np.array([[init_time]]), space_fe.dofs))

    total_n_dofs_t = 0
    sol_slabs = []
    err_slabs = np.zeros((len(time_slabs),))  # square L2 error current slab
    norm_u_slabs = np.zeros_like(err_slabs)  # square L2 norm apx. sol.

    # Time marching over slabs
    for i, slab in enumerate(time_slabs):
        if verbose:
            print(
                f"Solving on slab_{i} = D x ({round(slab[0], 5)}, {round(slab[1], 5)}) ...",
                flush=True,
            )

        # Compute FE object for current slab TIME discretization
        msh_t = mesh.create_interval(comm, n_time, [slab[0], slab[1]])
        V_t = fem.functionspace(msh_t, ("Lagrange", order_t))
        Time = TimeFE(msh_t, V_t)
        total_n_dofs_t += Time.n_dofs

        # Assemble linear system
        system_matrix, mass_matrix, stiffness_matrix, rhs, ex_sol_slab = (
            assemble_ctg_slab(space_fe, u0, Time, exact_rhs, boundary_data)
        )

        # Solve linear system (sparse direct solver)
        sol_slab_dofs = scipy.sparse.linalg.spsolve(system_matrix, rhs)
        sol_slabs.append(sol_slab_dofs)

        # Check residual
        residual_slab = system_matrix.dot(sol_slab_dofs) - rhs
        rel_res_slab = np.linalg.norm(residual_slab) / np.linalg.norm(sol_slab_dofs)
        warn = False
        if rel_res_slab > 1.0e-4:
            warn = True
            print("WARNING: ", end="")
        if verbose or warn:
            print(f"Relative residual solver slab {i}:", float_f(rel_res_slab))

        # Get initial condition on next slab = final condition from this slab
        last_time_dof = Time.dofs.argmax()
        u0 = sol_slab_dofs[last_time_dof * space_fe.n_dofs : (last_time_dof + 1) * space_fe.n_dofs]

        # Error curr slab
        if callable(exact_sol):  # compute error only if exact_sol is a function
            err_slabs[i], norm_u_slabs[i] = compute_error_slab(
                space_fe,
                exact_sol,
                err_type_x,
                err_type_t,
                Time,
                sol_slab_dofs,
                # mass_matrix,
                # stiffness_matrix
            )

            if verbose:
                print("Current " + err_type_x + " error:", float_f(err_slabs[i]))
                print(
                    "Current " + err_type_x + " relative error:", float_f(err_slabs[i] / norm_u_slabs[i])
                )
                print("Done.\n")

    n_dofs = space_fe.n_dofs * total_n_dofs_t
    return sol_slabs, err_slabs, norm_u_slabs, n_dofs


def compute_error_slab(
    Space, exact_sol, err_type_x, err_type_t, Time, sol_slab
):  # mass_mat, stif_mat, ):
    # --------------------------------- WRONG -------------------------------- #
    # space_time_coords = cart_prod_coords(Time.dofs, Space.dofs)
    # ex_sol_slab = exact_sol(space_time_coords)  
    # # compute exact sol on dofs; i.e. PROJECT exact sol in discrete sapce
    # --------------------------------- WRONG -------------------------------- #

    # refine Time
    msh_t = Time.mesh
    msh_t_ref = mesh.refine(msh_t)[0]
    p_Time = Time.V.element.basix_element.degree
    V_t_ref = fem.functionspace(msh_t_ref, ("Lagrange", p_Time))
    Time_ref = TimeFE(msh_t_ref, V_t_ref)

    # refine Space
    msh_x = Space.mesh
    msh_x_ref = mesh.refine(msh_x)[0]
    p_Space = Space.V.element.basix_element.degree
    V_x_ref = fem.functionspace(msh_x_ref, ("Lagrange", p_Space))
    Space_ref = SpaceFE(msh_x_ref, V_x_ref)

    # Interpolate exact sol in fine space # TODO works only for P=1!
    fine_coords = cart_prod_coords(Time_ref.dofs, Space_ref.dofs)
    ex_sol_ref = exact_sol(fine_coords)

    # Interpolate numerical sol using griddata (linear interpolation) 
    # TODO works only for P=1!
    coarse_coords = cart_prod_coords(Time.dofs, Space.dofs)
    sol_slab_ref = griddata(
        coarse_coords, sol_slab, fine_coords, method="linear", fill_value=0.0
    )

    # Adapt to error type
    if err_type_x == "h1":
        ip_space_ref =  Space_ref.matrix["mass"] + Space_ref.matrix["laplace"]
    elif err_type_x == "l2":
        ip_space_ref =  Space_ref.matrix["mass"]
    else:
        raise ValueError(f"Unknown error type x: {err_type_x}")
    
    if err_type_t == "l2":
        ip_tx = scipy.sparse.kron(Time_ref.matrix["mass"], ip_space_ref)
    elif err_type_t == "linf":  # take max
        pass
    else:
        raise ValueError(f"Unknown error type t: {err_type_t}")

    err_fun_ref = ex_sol_ref - sol_slab_ref

    if err_type_t == "l2":
        err = sqrt(ip_tx.dot(err_fun_ref).dot(err_fun_ref))
        norm_u = sqrt(ip_tx.dot(sol_slab_ref).dot(sol_slab_ref))
    elif err_type_t == "linf":
        err = -1.
        norm_u = -1.
        for i, t in enumerate(Time.dofs):
            dofs_c = sol_slab_ref[i*Space_ref.n_dofs:(i+1)*Space_ref.n_dofs]
            norm_c = sqrt(ip_space_ref.dot(dofs_c).dot(dofs_c))
            norm_u = max(norm_u, norm_c)
            err_dofs_c = err_fun_ref[i*Space_ref.n_dofs:(i+1)*Space_ref.n_dofs]
            err_c = sqrt(ip_space_ref.dot(err_dofs_c).dot(err_dofs_c))
            err = max(err, err_c)
    return err, norm_u
