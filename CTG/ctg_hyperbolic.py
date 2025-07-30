import sys
import numpy as np
import scipy.sparse
from dolfinx import fem, mesh

sys.path.append("./")
from CTG.utils import cart_prod_coords, compute_error_slab, float_f
from CTG.FE_spaces import TimeFE, SpaceFE


def _impose_IC_strong(X0, n_dofs_trial, n_dofs_x, system_mat, rhs):

    # USE LIFTING METHOD 
    # x = x_0 + x_D
    # Ax = b -> A x_0 = f - A x_D
    # x = x_0 + x_D
    
    n_dofs_test, n_dofs_trial = system_mat.shape


    n_dofs_scalar = int(n_dofs_trial/2)

    x_D = np.zeros((n_dofs_trial,))  # seee IC as Dirichlet BC

    where_t0 = np.arange(n_dofs_x)  
    where_t0 = np.append(where_t0, np.arange(n_dofs_scalar, n_dofs_scalar + n_dofs_x))

    x_D[where_t0] = X0.flatten()  # dofs for t=t_0

    A_x_D = system_mat.dot(x_D)

    rhs = rhs - A_x_D

    

    # n_dofs_scalar = n_dofs_trial * n_dofs_x
    # # Indicator dofs with t=t_0
    # dofs_at_t0 = np.zeros((2 * n_dofs_scalar))
    # where_t_0 = np.arange(n_dofs_x)  
    # where_t_0 = np.append(where_t_0, np.arange(n_dofs_scalar, n_dofs_scalar + n_dofs_x))  
    # dofs_at_t0[where_t_0] = 1.0

    # system_mat = system_mat.multiply((1.0 - dofs_at_t0).reshape(-1, 1))
    # system_mat += scipy.sparse.diags(dofs_at_t0, shape=system_mat.shape)

    # rhs[:n_dofs_x] = X0[0]
    # rhs[n_dofs_scalar : n_dofs_scalar + n_dofs_x] = X0[1]

    return system_mat, rhs, x_D


def _impose_boundary_conditions(sys_mat, rhs, dofs_t_trial, dofs_x_bd, bd_data, tx_coords):
    # Use lifting method 
    # x = x_0 + x_D
    # A x_0 = f - A x_D
    # x = x_0 +  x_D

    n_dofs_test, n_dofs_trial = sys_mat.shape

    
    x_D = bd_data(tx_coords).flatten()  # NB bd_data(tx_coords).shape=(2, n_dofs)

    rhs = rhs - sys_mat.dot(x_D)

    # # 1. recover data:

    # # Indicator function bd dofs in tx coords
    # dofs_boundary = np.kron(
    #     np.ones((dofs_t_trial.shape[0], 1)), dofs_x_bd.reshape(-1, 1)
    # ).flatten()

    # bc_data_vals = bd_data(tx_coords)  # shape (2, n_tx)
    # bc_data_vals = np.concat((bc_data_vals[0, :], bc_data_vals[1, :]))

    # # 2. Edit system matrix: Put to 0 entries corresponding to boundary

    # # expand system matrix to have as many rows as columns
    # if sys_mat.shape[0] < sys_mat.shape[1]:
    #     sys_mat = sys_mat.tocsr()
    #     diff = abs(sys_mat.shape[0] - sys_mat.shape[1])
    #     shape_addit = (diff, sys_mat.shape[1])
    #     addit = scipy.sparse.csr_matrix(shape_addit)
    #     sys_mat = scipy.sparse.vstack([sys_mat, addit])
    
    # sys_mat = sys_mat.multiply((1.0 - dofs_boundary).reshape(-1, 1))
    # sys_mat += scipy.sparse.diags(dofs_boundary)

    # # 3. Edit RHS vector
    # if rhs.shape[0] < sys_mat.shape[0]:
    #     diff = sys_mat.shape[0] - rhs.shape[0]
    #     rhs = np.concatenate([rhs, np.zeros(diff)])
    
    # rhs = rhs * (1.0 - dofs_boundary)
    # rhs += bc_data_vals * dofs_boundary

    return sys_mat, rhs, x_D


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
    space_time_coords = cart_prod_coords(time_fe.dofs_trial, space_fe.dofs)
    rhs = mass_mat.dot(exact_rhs(space_time_coords).T)  # (n, 2)
    rhs = rhs.flatten("F")  # keep COLUMNS intact, stack one after the other

    system_mat, rhs, x_IC = _impose_IC_strong(
        X0, time_fe.n_dofs_trial, space_fe.n_dofs, system_mat, rhs
    )

    system_mat, rhs, x_D = _impose_boundary_conditions(
        system_mat,
        rhs,
        time_fe.dofs_trial,
        space_fe.boundary_dof_vector,  # Indicator fun. boundary for *vectorial* f
        boundary_data,
        space_time_coords,
    )

    return system_mat, rhs, x_IC + x_D


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
        V_t_trial = fem.functionspace(msh_t, ("Lagrange", order_t))
        V_t_test = fem.functionspace(msh_t, ("DG", order_t-1))
        time_fe = TimeFE(msh_t, V_t_trial, V_t_test)
        total_n_dofs_t += time_fe.n_dofs_trial

        # Assemble linear system
        system_matrix, rhs, x_D = _assemble_wave(
            space_fe, X0, time_fe, exact_rhs, boundary_data
        )

        # Solve linear system (sparse direct solver)
        x0, info = scipy.sparse.linalg.lsqr(system_matrix, rhs)[:2]
        x = x0 + x_D  # solution current slab with INITIAL and DIRICH conditions
        sol_slabs.append(x)

        # Check residual
        # residual_slab = system_matrix.dot(x0) - rhs
        # rel_res_slab = np.linalg.norm(residual_slab) / np.linalg.norm(x0)
        # warn = False
        # if rel_res_slab > 1.0e-4:
        #     warn = True
        #     print("WARNING: ", end="")
        # if verbose or warn:
        #     print(f"Relative residual solver slab {i}:", float_f(rel_res_slab))

        # Get initial condition on next slab = final condition from this slab
        dof_last_t = time_fe.dofs_trial.argmax()
        
        # number of t-x dofs for 1 scalar variable of 2
        n_dofs_scalar = int(x0.size/2)  
        
        u0 = x0[
            dof_last_t * space_fe.n_dofs : (dof_last_t + 1) * space_fe.n_dofs
        ]
        v0 = x0[
            n_dofs_scalar + dof_last_t * space_fe.n_dofs : n_dofs_scalar
            + (dof_last_t + 1) * space_fe.n_dofs
        ]
        X0 = np.vstack((u0, v0))  # must have shape (2, n_dofs_tx)

        # Error on u curr slab
        if callable(exact_sol):
            dofs_u_slab = x0[:n_dofs_scalar]
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
