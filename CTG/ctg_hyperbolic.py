import sys
from math import sqrt
import numpy as np
import scipy.sparse
from dolfinx import fem, mesh

sys.path.append("./")
from CTG.utils import cart_prod_coords, compute_error_slab, float_f, compute_time_slabs
from CTG.FE_spaces import TimeFE, SpaceFE

def impose_IC_BC(sys_mat, rhs, space_fe, time_fe, boundary_data_u, boundary_data_v, X0):
    xt_dofs = cart_prod_coords(time_fe.dofs_trial, space_fe.dofs)
    n_dofs_trial_scalar = int(sys_mat.shape[1]/2)
    n_x = space_fe.n_dofs
    n_t = time_fe.n_dofs_trial

    # Indicator dofs IC    
    ic_dofs_scalar = np.kron(time_fe.dof_IC_vector, np.ones((n_x, )))
    # Find dofs BD
    bd_dofs_scalar = np.kron(np.ones((n_t, )), space_fe.boundary_dof_vector)
    # Find compatibility dofs: those where initial and boundary conditions both imposed
    compat_dofs_scalar = np.logical_and(ic_dofs_scalar == 1, bd_dofs_scalar == 1)
    
    # global indicator scalar
    ic_bd_dofs_scalar = np.logical_or(ic_dofs_scalar, bd_dofs_scalar).astype(float)
    ic_bd_dofs = np.tile(ic_bd_dofs_scalar, 2)
    compat_dofs = np.tile(compat_dofs_scalar, 2)
    
    # Boundary dofs 
    XD = np.concatenate((boundary_data_u(xt_dofs), boundary_data_v(xt_dofs)))

    # Assert XD and X0 are equal on compat_dofs entries
    # assert np.allclose(XD[compat_dofs], X0[compat_dofs]), "XD and X0 differ on compatibility dofs"
    
    # Lift BC+IC
    X0D = X0 + XD - np.where(compat_dofs, X0, 0.)
    # here we removed the values of X0 on the compatibility dofs. This is better than the other way round because the boundary condition is always givena and exact, the initial dcondition may be only appproximately computed.

    rhs = rhs - sys_mat.dot(X0D)

    # Impose Homogenoeous BC+IC on rhs
    rhs = rhs * (1-ic_bd_dofs)

    # Impose homogeneous BC+IC on matrix
    sys_mat = sys_mat.multiply((1-ic_bd_dofs).reshape((-1, 1)))
    sys_mat += scipy.sparse.diags(ic_bd_dofs, offsets=0, shape=sys_mat.shape)  # u
    sys_mat += scipy.sparse.diags(ic_bd_dofs, offsets=n_dofs_trial_scalar, shape=sys_mat.shape)  # v

    return sys_mat, rhs, X0D
 

def assemble(space_fe, time_fe, boundary_data_u, boundary_data_v, X0, exact_rhs_0, exact_rhs_1):
    # Space-time matrices for scalar unknowns
    mass_mat = scipy.sparse.kron(time_fe.matrix["mass"], space_fe.matrix["mass"])
    stiffness_mat = scipy.sparse.kron(
        time_fe.matrix["mass"], space_fe.matrix["laplace"]
    )
    derivative_mat = scipy.sparse.kron(
        time_fe.matrix["derivative"], space_fe.matrix["mass"]
    )

    # Space-time matrices for vectorial unknowns
    sys_mat = scipy.sparse.block_array([[derivative_mat, -mass_mat], [stiffness_mat, derivative_mat]])
    
    # Right hand side vector
    xt_dofs = cart_prod_coords(time_fe.dofs_trial, space_fe.dofs)
    rhs0 = mass_mat.dot(exact_rhs_0(xt_dofs))
    rhs1 = mass_mat.dot(exact_rhs_1(xt_dofs))
    rhs = np.concatenate((rhs0, rhs1))

    # Impose IC+BC
    sys_mat, rhs, X0D = impose_IC_BC(sys_mat, rhs, space_fe, time_fe, boundary_data_u, boundary_data_v, X0)
    
    return sys_mat, rhs, X0D


def compute_err_ndofs(comm, order_t, err_type_x, err_type_t, time_slabs, space_fe, sol_slabs, exact_sol_u):
    err_slabs = -1.0 * np.ones(len(time_slabs))
    norm_u_slabs = -1.0 * np.ones_like(err_slabs)
    total_n_dofs_t = 0
    for i, slab in enumerate(time_slabs):
        msh_t = mesh.create_interval(comm, 1, [slab[0], slab[1]])
        V_t_trial = fem.functionspace(msh_t, ("Lagrange", order_t))
        V_t_test = fem.functionspace(msh_t, ("DG", order_t))
        time_fe = TimeFE(msh_t, V_t_trial, V_t_test)
        total_n_dofs_t += time_fe.n_dofs_trial

        X = sol_slabs[i]
        u = X[:int(X.size/2)]

        err_slabs[i], norm_u_slabs[i] = compute_error_slab(u, exact_sol_u, space_fe, time_fe, err_type_x, err_type_t)
    
    total_n_dofs = space_fe.n_dofs * total_n_dofs_t

    if err_type_t == "linf":
        total_err = np.amax(err_slabs)
        total_norm_u = np.amax(norm_u_slabs)
    elif err_type_t == "l2":
        total_err = sqrt(np.sum(np.square(err_slabs)))
        total_norm_u = sqrt(np.sum(np.square(norm_u_slabs)))
    else:
        raise ValueError(f"Unknown error type in time: {err_type_t}")
    total_rel_err = total_err / total_norm_u
    return total_err, total_rel_err, total_n_dofs, err_slabs, norm_u_slabs

def ctg_wave(comm, boundary_D, V_x, 
            start_time, end_time, t_slab_size, order_t,
            boundary_data_u, boundary_data_v, exact_rhs_0, exact_rhs_1, initial_data_u, initial_data_v):

    time_slabs = compute_time_slabs(start_time, end_time, t_slab_size)

    space_fe = SpaceFE(V_x, boundary_D)
    n_x = space_fe.n_dofs

    # Vector of dofs IC (over first slab)
    tx_coords = cart_prod_coords(np.array(time_slabs[0]), space_fe.dofs)  # shape (n_dofs_tx_scalar, 2) 
    u0 = initial_data_u(tx_coords)  # shape (n_dofs_tx_scalar, )
    v0 = initial_data_v(tx_coords)  # shape (n_dofs_tx_scalar, )
    X0 = np.concatenate((u0, v0))  # shape (2*n_dofs_tx_scalar, )
    
    # time stepping
    sol_slabs = []
    for i, slab in enumerate(time_slabs):
        print(f"Slab_{i} = D x ({round(slab[0], 4)}, {round(slab[1], 4)}) ...")

        # Assemble time FE curr slab
        msh_t = mesh.create_interval(comm, 1, [slab[0], slab[1]])
        V_t_trial = fem.functionspace(msh_t, ("Lagrange", order_t))
        V_t_test = fem.functionspace(msh_t, ("DG", order_t))
        time_fe = TimeFE(msh_t, V_t_trial, V_t_test)
        n_t = time_fe.n_dofs_trial
        n_dofs_scalar = n_t*n_x

        # Assemble space-time linear system
        sys_mat, rhs, X0D = assemble(space_fe, time_fe, boundary_data_u, boundary_data_v, X0, exact_rhs_0, exact_rhs_1)

        # Solve
        X = scipy.sparse.linalg.spsolve(sys_mat, rhs)        
        residual = np.linalg.norm(sys_mat.dot(X) - rhs) / np.linalg.norm(X)
        print(f"Relative residual norm: {residual:.2e}")

        X = X + X0D  # add IC and BC
        sol_slabs.append(X)

        # Extract IC dofs next time slab
        X0 = np.zeros_like(X0)
        dofs_ic_tx_scalar = np.kron(time_fe.dof_IC_vector, np.ones(space_fe.n_dofs))
        dofs_ic_tx = np.tile(dofs_ic_tx_scalar, 2).astype(bool)
        dofs_fc_tx_scalar = np.kron(time_fe.dof_FC_vector, np.ones(space_fe.n_dofs))
        dofs_fc_tx = np.tile(dofs_fc_tx_scalar, 2).astype(bool)
        X0[dofs_ic_tx]=X[dofs_fc_tx]
    
    return time_slabs, space_fe,n_x,sol_slabs,n_dofs_scalar