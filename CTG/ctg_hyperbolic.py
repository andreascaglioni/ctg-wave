import sys
import numpy as np
import scipy.sparse
from dolfinx import fem, mesh

sys.path.append("./")
from CTG.utils import cart_prod_coords, compute_error_slab, float_f
from CTG.FE_spaces import TimeFE, SpaceFE


def _impose_initial_conditions(sys_mat, rhs, space_fe, indic_dof_t0_t_trial, indic_dof_t0_t_test, X0):
    # We use the lifting method, i.e. for x = x_0 + x_IC, solve the system A x_0 = f - A x_D with homogenoeus Dirichlet BCs in the IC dofs, then set x = x_0 + x_IC.
    # this functions assembles A and f-A*x_IC with 0 Dirichlet BCs and return initial datum x_IC (through its dofs array) as well.

    # NB X0 is array of shape(2, n_dofs_tx) of coords of IC

    # ----------------------------------- - ---------------------------------- #
    # f = f - A * x_IC
    x_IC = X0.flatten()  # keep ROWS intact
    rhs = rhs - sys_mat.dot(x_IC)

    # impose homogeneous D condition on rhs
    ones_x = np.ones((space_fe.n_dofs, ))
    indic_t0_tx_test = np.kron(indic_dof_t0_t_test, ones_x)
    rhs = rhs * indic_t0_tx_test
    # impose homogeneous D condition on system matrix
    indic_t0_tx_trial = np.kron(indic_dof_t0_t_trial, ones_x)
    sys_mat = sys_mat.multiply(indic_t0_tx_trial.reshape((-1, 1)))

    return sys_mat, rhs, x_IC


def _impose_boundary_conditions(sys_mat, rhs, time_fe, indicator_bd_x, bd_data, tx_coords):
    # We use the lifting method, i.e. for x = x_0 + x_D, solve the system A x_0 = f - A x_D with homogenoeus Dirichlet BCs, then set x = x_0 + x_D.
    # this functions assembles A and f-A*x_D with 0 Dirichlet BCs and return dirchlet datum x_D as well.

    # ----------------------------------- - ---------------------------------- #

    # extract dirichlet BC in tx coordinates
    x_D = bd_data(tx_coords).flatten()  # TODO works only for Lagr FE
    
    # lift BC u_D from u and impose on f: f-A*x_D
    rhs = rhs - sys_mat.dot(x_D)

    # impose homogenous conditions on A and f
    ones_tt_test = np.ones((time_fe.n_dofs_test, 1))
    indic_bd_tx_test = np.kron(ones_tt_test, indicator_bd_x.reshape((-1, 1)))
    rhs = rhs * (1-indic_bd_tx_test)

    ones_tt_trial = np.ones((time_fe.n_dofs_trial, 1))
    indic_bd_tx_trial = np.kron(ones_tt_trial, indicator_bd_x.reshape((-1, 1)))
    sys_mat = sys_mat.multiply(1.-indic_bd_tx_trial)
    sys_mat = sys_mat + scipy.sparse.diags(indic_bd_tx_test, shape=sys_mat.shape)

    return sys_mat, rhs, x_D

def _assemble_wave(space_fe, X0, time_fe, exact_rhs, boundary_data, boundary_IC):
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
    sys_mat_2 = scipy.sparse.block_array([[None, -mass_mat], [-stiffness_mat, None]])

    sys_mat = derivative_mat_2 + sys_mat_2

    # Assemble RHS vector as space-time mass * RHS on dofs 
    # TODO generalize to non-interpolatory
    xt_dofs = cart_prod_coords(time_fe.dofs_trial, space_fe.dofs)
    rhs = mass_mat.dot(exact_rhs(xt_dofs).T)  # (n, 2)
    rhs = rhs.flatten("F")  # keep COLUMNS intact, stack one after the other
    
    Init_data_dummy = lambda x : np.zeros((2, x.shape[0]))  # I only care about dof vector # noqa: E731
    dofs_t0_t_trial, dofs_t0_t_test = time_fe.get_IC_dofs(Init_data_dummy, boundary_IC)

    sys_mat, rhs, x_IC = _impose_initial_conditions(
        # X0, time_fe.n_dofs_trial, space_fe.n_dofs, system_mat, rhs
        sys_mat, rhs, space_fe, dofs_t0_t_trial, dofs_t0_t_test, X0)

    sys_mat, rhs, x_D = _impose_boundary_conditions(
        sys_mat,
        rhs,
        time_fe,
        space_fe.boundary_dof_vector,  # Indicator fun. on dof vector
        boundary_data,
        xt_dofs,
    )

    return sys_mat, rhs, x_IC + x_D


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
    t0 = time_slabs[0][0]
    tx_coords_t0 = cart_prod_coords(np.array([[t0]]), space_fe.dofs) # shape (n_dofs_x, 2)
    X0 = initial_data(tx_coords_t0)  # shape (2, n_dofs_x)

    # Scalar space FE
    V_x_scalar = fem.functionspace(space_fe.mesh, ("Lagrange", 1))
    fe_space_scalar = SpaceFE(space_fe.mesh, V_x_scalar)
    
    # Time marching over slabs
    total_n_dofs_t = 0
    sol_slabs = []
    err_slabs = -1. * np.ones(len(time_slabs))
    norm_u_slabs = -1. * np.ones_like(err_slabs)
    for i, slab in enumerate(time_slabs):
        if verbose:
            print(f"Slab_{i} = D x ({round(slab[0], 4)}, {round(slab[1], 4)}) ...", flush=True)

        boundary_IC = lambda t: np.isclose(t[0], slab[0])  # noqa: E731

        # Time FE current slab
        msh_t = mesh.create_interval(comm, n_time, [slab[0], slab[1]])

        V_t_trial = fem.functionspace(msh_t, ("Lagrange", order_t, (2,)))
        V_t_test = fem.functionspace(msh_t, ("DG", order_t-1, (2,)))

        time_fe = TimeFE(msh_t, V_t_trial, V_t_test)
        total_n_dofs_t += time_fe.n_dofs_trial

        # Assemble linear system
        system_matrix, rhs, x_D = _assemble_wave(
            space_fe, X0, time_fe, exact_rhs, boundary_data, boundary_IC
        )
        # Solve linear system (sparse direct solver)
        x, info = scipy.sparse.linalg.lsqr(system_matrix, rhs)[:2]
        x = x + x_D  # add IC and BC
        sol_slabs.append(x)

        # Check residual of linear system
        residual = np.linalg.norm(system_matrix.dot(x) - rhs)
        if verbose:
            print(f"Residual of linear system on slab {i}: {residual:.2e}")

        # Get initial condition on next slab = final condition from this slab
        dof_last_t = time_fe.dofs_trial.argmax()
        n_dofs_scalar = int(x.size/2)  
        ii_u_end = np.arange(dof_last_t * space_fe.n_dofs, (dof_last_t + 1) * space_fe.n_dofs)
        u0 = x[ii_u_end]
        ii_v_end = ii_u_end + n_dofs_scalar
        v0 = x[ii_v_end]
        X0 = np.vstack((u0, v0))  # must have shape (2, n_dofs_tx)

        # Error on u curr slab
        if callable(exact_sol):
            coords_u_slab = x[:n_dofs_scalar]
            exact_u = lambda X: exact_sol(X)[0, :]  # exact_sol returns (2, n_dofs)

            err_slabs[i], norm_u_slabs[i] = compute_error_slab(
                fe_space_scalar, exact_u, err_type_x, err_type_t, time_fe, coords_u_slab
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
