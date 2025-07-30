import sys
import numpy as np
import scipy.sparse
from dolfinx import fem, mesh

sys.path.append("./")
from CTG.utils import cart_prod_coords, compute_error_slab, float_f
from CTG.FE_spaces import TimeFE


def _impose_IC_strong(u0, n_dofs_t, n_dofs_x, system_mat, rhs):
    """
    Impose initial conditions strongly on the system matrix and right-hand side vector by overweiting the corresponding  DOFs in the system matrix and right-hand side.
    Args:
        u0 (array-like): Initial condition values for the spatial degrees of freedom at t=0.
        n_dofs_t (int): Number of temporal degrees of freedom.
        n_dofs_x (int): Number of spatial degrees of freedom.
        system_mat (scipy.sparse matrix): System matrix to be modified.
        rhs (array-like): Right-hand side vector to be modified.
    Returns:
        tuple: Modified system matrix and right-hand side vector with initial conditions imposed.
    """

    dofs_at_t0 = np.zeros((n_dofs_t * n_dofs_x))  # indicator dofs at t_0
    dofs_at_t0[:n_dofs_x] = 1.0

    system_mat = system_mat.multiply((1.0 - dofs_at_t0).reshape(-1, 1))
    system_mat += scipy.sparse.diags(dofs_at_t0)
    rhs[:n_dofs_x] = u0
    return system_mat, rhs


def _impose_boundary_conditions(sys_mat, rhs, t_dofs, bd_dofs_x, bd_data, tx_coords):
    """
    Modifies the system matrix and right-hand side vector to impose boundary conditions. For degrees of freedom that belong to the boundary, the corresponding row in the system matrix is replaced with a delta function (identity), and the RHS entry is set to the boundary condition value.

    Args:
        sys_mat (scipy.sparse.spmatrix): The system matrix to be modified.
        rhs (np.ndarray): The right-hand side vector to be modified.
        t_dofs (np.ndarray): Array of time DOFs.
        bd_dofs_x (np.ndarray): Array indicating which DOFs are on the boundary.
        bd_data (Callable): Function that returns boundary condition values given space-time coordinates.
        tx_coords (np.ndarray): Array of space-time coordinates for each DOF.

    Returns:
        Tuple[scipy.sparse.spmatrix, np.ndarray, np.ndarray]:
            - System matrix with boundary conditions imposed.
            - RHS vector with boundary conditions imposed.
            - Array of boundary condition values for the current slab.
    """

    # 1. recover data
    # Indicator function bd dofs in tx coords
    dofs_boundary = np.kron(
        np.ones((t_dofs.shape[0], 1)), bd_dofs_x.reshape(-1, 1)
    ).flatten()
    bc_curr_slab = bd_data(tx_coords)

    # 2. Edit system matrix: Put to 0 entries corresponding to boundary
    sys_mat = sys_mat.multiply((1.0 - dofs_boundary).reshape(-1, 1))
    sys_mat += scipy.sparse.diags(dofs_boundary)

    # 3. Edit RHS vector
    rhs = rhs * (1.0 - dofs_boundary)
    rhs += bc_curr_slab * dofs_boundary

    return sys_mat, rhs


def _assemble_heat(Space, u0, Time, exact_rhs, boundary_data):
    # Assemble space-time matrices (linear PDEs -> Kronecker product t & x matrices)
    mass_mat = scipy.sparse.kron(Time.matrix["mass"], Space.matrix["mass"])
    stiffness_mat = scipy.sparse.kron(Time.matrix["mass"], Space.matrix["laplace"])
    system_mat = (
        scipy.sparse.kron(Time.matrix["derivative"], Space.matrix["mass"])
        + stiffness_mat
    )

    # Assemble RHS vector as space-time mass * RHS on dofs
    # TODO better to use projection? I'll use higher order FEM!
    space_time_coords = cart_prod_coords(Time.dofs, Space.dofs)
    rhs = mass_mat.dot(exact_rhs(space_time_coords))

    # Impose initial condition strongly
    system_mat, rhs = _impose_IC_strong(u0, Time.n_dofs, Space.n_dofs, system_mat, rhs)

    system_mat, rhs = _impose_boundary_conditions(
        system_mat,
        rhs,
        Time.dofs,
        Space.boundary_dof_vector,
        boundary_data,
        space_time_coords,
    )

    return system_mat, rhs


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
    tx_coords_t0 = cart_prod_coords(np.array([[init_time]]), space_fe.dofs)
    u0 = initial_data(tx_coords_t0)

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
        time_fe = TimeFE(msh_t, V_t)
        total_n_dofs_t += time_fe.n_dofs_trial

        # Assemble linear system
        system_matrix, rhs = _assemble_heat(
            space_fe, u0, time_fe, exact_rhs, boundary_data
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
        last_time_dof = time_fe.dofs_trial.argmax()
        u0 = sol_slab_dofs[
            last_time_dof * space_fe.n_dofs : (last_time_dof + 1) * space_fe.n_dofs
        ]

        # Error curr slab
        if callable(exact_sol):  # compute error only if exact_sol is a function
            err_slabs[i], norm_u_slabs[i] = compute_error_slab(
                space_fe, exact_sol, err_type_x, err_type_t, time_fe, sol_slab_dofs
            )

            if verbose:
                print("Current " + err_type_x + " error:", float_f(err_slabs[i]))
                print(
                    "Current " + err_type_x + " relative error:",
                    float_f(err_slabs[i] / norm_u_slabs[i]),
                )
                print("Done.\n")

    n_dofs = space_fe.n_dofs * total_n_dofs_t
    return sol_slabs, err_slabs, norm_u_slabs, n_dofs
