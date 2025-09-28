import sys
import numpy as np
import scipy.sparse
from dolfinx import fem, mesh

sys.path.append("./")
from CTG.utils import cart_prod_coords, float_f, compute_time_slabs
from CTG.FE_spaces import TimeFE, SpaceFE


def impose_IC_BC(sys_mat, rhs, space_fe, time_fe, boundary_data_u, boundary_data_v, X_0):
    xt_dofs = cart_prod_coords(time_fe.dofs, space_fe.dofs)
    n_dofs_trial_scalar = int(sys_mat.shape[1]/2)
    n_x = space_fe.n_dofs
    n_t = time_fe.n_dofs

    # Indicator dofs IC    
    ic_dofs_scalar = np.kron(time_fe.dof_IC_vector, np.ones((n_x, )))
    # Indicator dofs BD
    bd_dofs_scalar = np.kron(np.ones((n_t, )), space_fe.boundary_dof_vector)
    # Find compatibility dofs: those where IC nad BC are both imposed
    compat_dofs_scalar = np.logical_and(ic_dofs_scalar == 1, bd_dofs_scalar == 1)
    
    # Vectorial indicator functions
    ic_bd_dofs_scalar = np.logical_or(ic_dofs_scalar, bd_dofs_scalar).astype(float)
    ic_bd_dofs = np.tile(ic_bd_dofs_scalar, 2)
    compat_dofs = np.tile(compat_dofs_scalar, 2)
    
    # Boundary dofs 
    X_D = np.concatenate((boundary_data_u(xt_dofs), boundary_data_v(xt_dofs)))
    
    # Compbine IC and BC
    X_0D = X_0 + X_D - np.where(compat_dofs, X_0, 0.)
    # here we removed the values of X0 on the compatibility dofs. This is better than the other way round because the boundary condition is always givena and exact, the initial dcondition may be only appproximately computed.

    # Lift IC+BC
    rhs = rhs - sys_mat.dot(X_0D)

    # Impose Homogenoeous BC+IC on rhs
    rhs = rhs * (1-ic_bd_dofs)

    # Impose homogeneous BC+IC on matrix
    sys_mat = sys_mat.multiply((1-ic_bd_dofs).reshape((-1, 1)))
    sys_mat += scipy.sparse.diags(ic_bd_dofs, offsets=0, shape=sys_mat.shape)  # u
    sys_mat += scipy.sparse.diags(ic_bd_dofs, offsets=n_dofs_trial_scalar, shape=sys_mat.shape)  # v

    return sys_mat, rhs, X_0D
 

def assemble(space_fe, time_fe, boundary_data_u, boundary_data_v, X0, exact_rhs_0, exact_rhs_1, W_path):
    # Space-time matrices for scalar unknowns
    mass_mat = scipy.sparse.kron(time_fe.matrix["mass"], space_fe.matrix["mass"])
    W_mass_mat = scipy.sparse.kron(time_fe.matrix["W_mass"], space_fe.matrix["mass"])
    WW_mass_mat = scipy.sparse.kron(time_fe.matrix["WW_mass"], space_fe.matrix["mass"])
    stiffness_mat = scipy.sparse.kron(
        time_fe.matrix["mass"], space_fe.matrix["laplace"]
    )
    derivative_mat = scipy.sparse.kron(
        time_fe.matrix["derivative"], space_fe.matrix["mass"]
    )

    # Space-time matrices for vectorial unknowns
    sys_mat = scipy.sparse.block_array([[derivative_mat, None], [None, derivative_mat]]) 
    sys_mat += scipy.sparse.block_array([[None, -mass_mat], [stiffness_mat, None]])
    # the next term from the PWE
    sys_mat += scipy.sparse.block_array([[W_mass_mat, None], [WW_mass_mat, W_mass_mat]])
    
    # Right hand side vector
    xt_dofs = cart_prod_coords(time_fe.dofs, space_fe.dofs)
    rhs0 = mass_mat.dot(exact_rhs_0(xt_dofs))
    rhs1 = mass_mat.dot(exact_rhs_1(xt_dofs))
    rhs = np.concatenate((rhs0, rhs1))

    # Impose IC+BC
    sys_mat, rhs, X0D = impose_IC_BC(sys_mat, rhs, space_fe, time_fe, boundary_data_u, boundary_data_v, X0)
    
    return sys_mat, rhs, X0D


def ctg_wave(comm, boundary_D, V_x, 
            start_time, end_time, t_slab_size, order_t,
            boundary_data_u, boundary_data_v, exact_rhs_0, exact_rhs_1, initial_data_u, initial_data_v, W_t = None, verbose=False):

    time_slabs = compute_time_slabs(start_time, end_time, t_slab_size)
    space_fe = SpaceFE(V_x, boundary_D)

    # Vector of dofs IC (over first slab)
    tx_coords = cart_prod_coords(np.array(time_slabs[0]), space_fe.dofs)  # shape (n_dofs_tx_scalar, 2) 
    u0 = initial_data_u(tx_coords)  # shape (n_dofs_tx_scalar, )
    v0 = initial_data_v(tx_coords)  # shape (n_dofs_tx_scalar, )
    X0 = np.concatenate((u0, v0))  # shape (2*n_dofs_tx_scalar, )
    
    # time stepping
    sol_slabs = []
    total_n_dofs_t = 0
    for i, slab in enumerate(time_slabs):
        if verbose:
            print(f"Slab_{i} = D x ({round(slab[0], 4)}, {round(slab[1], 4)}) ...")

        # Assemble time FE curr slab
        msh_t = mesh.create_interval(comm, 1, [slab[0], slab[1]])
        V_t_trial = fem.functionspace(msh_t, ("Lagrange", order_t))
        time_fe = TimeFE(msh_t, V_t_trial, W_t)

        # Compute n dofs curr slab
        total_n_dofs_t += time_fe.n_dofs

        # Assemble space-time linear system
        sys_mat, rhs, X0D = assemble(space_fe, time_fe, boundary_data_u, boundary_data_v, X0, exact_rhs_0, exact_rhs_1, W_t)

        # Solve
        X = scipy.sparse.linalg.spsolve(sys_mat, rhs)        
        residual = np.linalg.norm(sys_mat.dot(X) - rhs) / np.linalg.norm(X)

        if verbose:
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
    
    total_n_dofs = space_fe.n_dofs * total_n_dofs_t
    return time_slabs, space_fe, sol_slabs, total_n_dofs