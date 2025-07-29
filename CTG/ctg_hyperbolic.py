import sys
import numpy as np
import scipy.sparse
from dolfinx import fem, mesh

sys.path.append("./")
from CTG.utils import cart_prod_coords, compute_error_slab, float_f
from CTG.FE_spaces import TimeFE, SpaceFE


def _impose_IC_strong(X0, n_dofs_t, n_dofs_x, system_mat, rhs):
    """
    Impose initial conditions strongly on the system matrix and right-hand side vector by overweiting the corresponding  DOFs in the system matrix and right-hand side.
    Args:
        X0 (array-like): Initial condition values for the spatial degrees of freedom at t=0.
        n_dofs_t (int): Number of temporal degrees of freedom (scalar variable).
        n_dofs_x (int): Number of spatial degrees of freedom (scalar variable).
        system_mat (scipy.sparse matrix): System matrix to be modified.
        rhs (array-like): Right-hand side vector to be modified.
    Returns:
        tuple: Modified system matrix and right-hand side vector with initial conditions imposed.
    """

    n_dofs_scalar = n_dofs_t * n_dofs_x

    dofs_at_t0 = np.zeros((2 * n_dofs_scalar))  # indicator dofs at t_0
    dofs_at_t0[:n_dofs_x] = 1.0
    dofs_at_t0[n_dofs_scalar : n_dofs_scalar + n_dofs_x] = 1.0

    system_mat = system_mat.multiply((1.0 - dofs_at_t0).reshape(-1, 1))
    system_mat += scipy.sparse.diags(dofs_at_t0)

    rhs[:n_dofs_x] = X0[0]
    rhs[n_dofs_scalar : n_dofs_scalar + n_dofs_x] = X0[1]

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

    # 1. recover data:

    # Indicator function bd dofs in tx coords SCALAR unknown
    dofs_boundary = np.kron(
        np.ones((t_dofs.shape[0], 1)), bd_dofs_x.reshape(-1, 1)
    ).flatten()

    # Then for 2 components
    # n_dofs_tx = tx_coords.shape[0]
    # dofs_boundary = np.concat((dofs_boundary, dofs_boundary + n_dofs_tx))  # shift

    bc_curr_slab = bd_data(tx_coords)  # shape (2, n_tx)
    bc_curr_slab = np.concat((bc_curr_slab[0, :], bc_curr_slab[1, :]))

    # 2. Edit system matrix: Put to 0 entries corresponding to boundary
    sys_mat = sys_mat.multiply((1.0 - dofs_boundary).reshape(-1, 1))
    sys_mat += scipy.sparse.diags(dofs_boundary)

    # 3. Edit RHS vector
    rhs = rhs * (1.0 - dofs_boundary)
    rhs += bc_curr_slab * dofs_boundary

    return sys_mat, rhs


def _assemble_wave(space_fe, X0, time_fe, exact_rhs, boundary_data):
    # Space-time matrices for 1d unknows
    mass_mat = scipy.sparse.kron(time_fe.matrix["mass"], space_fe.matrix["mass"])
    stiffness_mat = scipy.sparse.kron(
        time_fe.matrix["mass"], space_fe.matrix["laplace"]
    )
    derivative_mat = scipy.sparse.kron(
        time_fe.matrix["derivative"], space_fe.matrix["mass"]
    )

    # Space-time matrices for 2d unknows
    derivative_mat_2 = scipy.sparse.block_diag((derivative_mat, derivative_mat))
    sys_mat_2 = scipy.sparse.block_array([[None, -mass_mat], [stiffness_mat, None]])

    system_mat = derivative_mat_2 + sys_mat_2

    # Assemble RHS vector as space-time mass * RHS on dofs
    space_time_coords = cart_prod_coords(time_fe.dofs, space_fe.dofs)
    rhs = mass_mat.dot(exact_rhs(space_time_coords).T)  # (n, 2)
    rhs = rhs.flatten()  # keep ROWS intact, stack one after the other

    system_mat, rhs = _impose_IC_strong(
        X0, time_fe.n_dofs, space_fe.n_dofs, system_mat, rhs
    )

    system_mat, rhs = _impose_boundary_conditions(
        system_mat,
        rhs,
        time_fe.dofs,
        space_fe.boundary_dof_vector,  # Indicator fun. boundary for *vectorial* f
        boundary_data,
        space_time_coords,
    )

    return system_mat, rhs


def run_CTG_wave(
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
    X0 = initial_data(tx_coords_t0)  # shape (2, n_dofs_x)

    # Scalar space FE
    V_x_scalar = fem.functionspace(space_fe.mesh, ("Lagrange", 1))
    space_fe_scalar = SpaceFE(space_fe.mesh, V_x_scalar)


    
    total_n_dofs_t = 0
    sol_slabs = []
    err_slabs = np.zeros(len(time_slabs))
    norm_u_slabs = np.zeros_like(err_slabs)
    # Time marching over slabs
    for i, slab in enumerate(time_slabs):
        if verbose:
            print(
                f"Solving on slab_{i} = D x ({round(slab[0], 4)}, {round(slab[1], 4)}) ...",
                flush=True,
            )

        # Time FE current slab
        msh_t = mesh.create_interval(comm, n_time, [slab[0], slab[1]])
        V_t = fem.functionspace(msh_t, ("Lagrange", order_t))
        time_fe = TimeFE(msh_t, V_t)
        total_n_dofs_t += time_fe.n_dofs

        # Assemble linear system
        system_matrix, rhs = _assemble_wave(
            space_fe, X0, time_fe, exact_rhs, boundary_data
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
        dofs_last_t = time_fe.dofs.argmax()
        n_dofs_scalar = int(rhs.size / 2)  # number of t-x dofs for 1 scalar variable
        u0 = sol_slab_dofs[
            dofs_last_t * space_fe.n_dofs : (dofs_last_t + 1) * space_fe.n_dofs
        ]
        v0 = sol_slab_dofs[
            n_dofs_scalar + dofs_last_t * space_fe.n_dofs : n_dofs_scalar
            + (dofs_last_t + 1) * space_fe.n_dofs
        ]
        X0 = np.vstack((u0, v0))

        # Error on u curr slab
        if callable(exact_sol):
            dofs_u_slab = sol_slab_dofs[:n_dofs_scalar]
            exact_u = lambda X: exact_sol(X)[0, :]

            err_slabs[i], norm_u_slabs[i] = compute_error_slab(
                space_fe_scalar, exact_u, err_type_x, err_type_t, time_fe, dofs_u_slab
            )

            if verbose:
                print(
                    "Current " + err_type_t + " - " + err_type_x + " error:",
                    float_f(err_slabs[i]),
                )
                print(
                    "Current " + err_type_t + " - " + err_type_x + " relative error:",
                    float_f(err_slabs[i] / norm_u_slabs[i]),
                )
                print("Done.\n")

    n_dofs = space_fe.n_dofs * total_n_dofs_t
    return sol_slabs, err_slabs, norm_u_slabs, n_dofs
