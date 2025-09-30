"""Assembly functions for WE approximation using CTG.

TODO add possibility to use piecewsie polynomial in time (not only global polynomial) for each time slab."""

import numpy as np
import scipy
from CTG.brownian_motion import param_LC_W
from CTG.utils import cart_prod_coords
from data.data_param_wave_eq import boundary_data_u, boundary_data_v


def impose_IC_BC(sys_mat, rhs, space_fe, time_fe, boundary_data_u, boundary_data_v, X_0):
    """
    Modifies the system matrix and RHS to impose initial and boundary conditions (IC and BC) for a hyperbolic PDE. Results in a system matrix and RHS for the corresponding homogeneous problem. To obtain again complete solution just add the IBC array.

    Args:
        sys_mat (scipy.sparse.spmatrix): System matrix before IC/BC.
        rhs (np.ndarray): Right-hand side vector before IC/BC.
        space_fe: Space FE object.
        time_fe: Time FE object.
        boundary_data_u (callable): Dirichlet BC for 1. component.
        boundary_data_v (callable): Dirichlet BC for 2. component.
        X_0 (np.ndarray): Initial condition vector.

    Returns:
        sys_mat (scipy.sparse.spmatrix): Modivfied, homogenous system matrix.
        rhs (np.ndarray): Modified RHS vector.
        X_0D (np.ndarray): Lifted solution vector with IC/BC imposed.

    Notes:
        Boundary conditions take precedence at overlapping DOFs.
    """

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


def assemble_A0_F(space_fe, time_fe, f):
    """Assemble the part of the linear system that is independent of y."""

    L_mat = scipy.sparse.kron(time_fe.matrix["mass"], space_fe.matrix["laplace"])
    D_mat = scipy.sparse.kron(time_fe.matrix["derivative"], space_fe.matrix["mass"])
    M_mat = scipy.sparse.kron(time_fe.matrix["mass"], space_fe.matrix["mass"])
    A0 = scipy.sparse.block_array([[D_mat, -M_mat], [-L_mat, D_mat]])
    F = np.concatenate([f, np.zeros_like(f)], axis=0)
    
    return A0, F


def assemble_Ay(y, space_fe, time_fe):
    # NB everithing is bathed, as input y has shape (batch, Ny) 

    # check y is 1d array(1 parameter vector)
    y = np.atleast_1d(y)
    assert len(y.shape) == 1

    # Get operators in time and space
    M_x = space_fe.matrix["mass"]
    T = np.amax(time_fe.dofs.flatten())
    W_t = lambda t: param_LC_W(y, t, T)[0]  # (t.size, )  # allow later evaluation

    time_fe.assemble_matrices_W(W_t)  
    M_Wt = time_fe.matrix["W_mass"]  # (Ndofs_t, Ndofs_t)
    M_W2t = time_fe.matrix["WW_mass"]  # (Ndofs_t, Ndofs_t)
    
    # Compute XT operators
    M_W = scipy.sparse.kron(M_Wt, M_x)
    M_W2 = scipy.sparse.kron(M_W2t, M_x)

    # Assemble first order system for each batch
    A_y = [[-M_W, None], [-M_W2, -M_W]]
    A_y = scipy.sparse.block_array(A_y)
    
    return A_y  #(Ndofs, Ndofs)


def assemble_complete_system(y,  # (Ny, )
                             A0_nbc_np, # (Ndofs, Ndofs)  NB numpy!!
                             F_nbc, # (Ndofs, )
                             space_fe, 
                             time_fe, 
                             X0): # (Ndofs, )
    
    assert len(y.shape) == 1

    # Assemble linear system operator
    Ay = assemble_Ay(y, space_fe, time_fe)  # (Ndofs, Ndofs)
    

    sys_mat = A0_nbc_np + Ay

    # Impose IBC to get homogenous equation (solution X-X0D)
    sys_mat, rhs, X0D = impose_IC_BC(sys_mat, F_nbc, space_fe, time_fe, boundary_data_u, boundary_data_v, X0)
    
    return sys_mat, rhs, X0D


def assemble_ctg(space_fe, time_fe, boundary_data_u, boundary_data_v, X0, exact_rhs_0, exact_rhs_1, W_path):
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