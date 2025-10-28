import numpy as np
import scipy
from ctg.FE_spaces import SpaceTimeFE


class AssemblerWave:
    """Assembles linear system for the CTG space-time solver. In particular:
    * Get operators from SpaceTimeFE
    * Impose initial and boundary conditions to compute the homogeneous system (to be solved) an the lifting function.

    Some attributes are set only upon construction: space_fe, exact_rhs_0, exact_rhs_1, boundary_data_u, boundary_data_v, initial_data_u, initial_data_v.
    Others, are set by setter methods because they will change over time slabs or over parameters:
        * Changing over time slabs: X0, time_fe
        * Changing over parameters: Ay, A"""

    def __init__(self, space_time_fe: SpaceTimeFE, verbose: bool = False):

        self.verbose = verbose
        self.update_space_time_fe(space_time_fe)

    def update_space_time_fe(self, space_time_fe: SpaceTimeFE):
        """Update SpaceTimeFE instance. Use it moving to a new time step."""
        self.space_time_fe = space_time_fe

    def assemble_A0_b(self, exact_rhs_0, exact_rhs_1):
        """Assemble parameter-independent part of the linear system."""
        if self.space_time_fe is None:
            if self.verbose:
                print("Warning: self.space_time_fe is None. Assembly interrupted.")
            return None, None
        # Matrix
        L_mat = self.space_time_fe.matrix["L"]
        D_mat = self.space_time_fe.matrix["D_t"]
        M_xt = self.space_time_fe.matrix["M"]
        A0_no_bc = scipy.sparse.block_array([[D_mat, -M_xt], [L_mat, D_mat]])
        # L_mat goes with + sign becaus it actually represent grad-grad term
        # Right hand side vector
        xt_dofs = self.space_time_fe.dofs
        rhs0 = M_xt.dot(exact_rhs_0(xt_dofs))
        rhs1 = M_xt.dot(exact_rhs_1(xt_dofs))
        b_no_bc = np.concatenate((rhs0, rhs1))
        return A0_no_bc, b_no_bc

    def assemble_A_W(self, W_t=None):
        """Assemble parameter-dependent part of the linear system."""
        if self.space_time_fe is None:
            if self.verbose:
                print("Warning: self.space_time_fe is None. Assembly interrupted.")
            return None
        if W_t is None:
            if self.verbose:
                print("Assembler Warning: W_T is None. SKipping assemply W_depndent matrices.")
            n = self.space_time_fe.n_dofs
            return scipy.sparse.csr_matrix((2 * n, 2 * n))
        self.space_time_fe.assemble_W(W_t)
        M_W = self.space_time_fe.matrix["M_W"]
        M_W2 = self.space_time_fe.matrix["M_WW"]
        return scipy.sparse.block_array([[-M_W, None], [-M_W2, -M_W]])

    def assemble_system(
        self, W_t, X0: np.ndarray, exact_rhs_0, exact_rhs_1, boundary_data_u, boundary_data_v
    ):
        # TODO Add possiblity to input A0_no_bc and A_W_no_bc

        # Assemble linear system operator
        A0_no_bc, b_no_bc = self.assemble_A0_b(exact_rhs_0, exact_rhs_1)
        A_W_no_bc = self.assemble_A_W(W_t)
        # Check if assembly was successful
        if A_W_no_bc is None or A0_no_bc is None or b_no_bc is None:
            print("Warning: Assembly failed because None matrix or vector. Returning None.")
            return None, None, None
        A = A0_no_bc + A_W_no_bc
        # Get homogenous equation and initial-(Dirichlet) boundary condition X0D for lifting
        A, b, X0D = self.impose_IC_BC(A, b_no_bc, X0, boundary_data_u, boundary_data_v)
        return A, b, X0D

    def impose_IC_BC(self, sys_mat, rhs, X0, boundary_data_u, boundary_data_v):
        if self.space_time_fe is None or self.space_time_fe.time_fe is None:
            print("Warning: self.space_time_fe is None. Skipping impose_IC_BC.")
            return None, None, None
        # Extract vars
        xt_dofs = self.space_time_fe.dofs
        n_dofs_scalar = int(sys_mat.shape[1] / 2)
        n_x = self.space_time_fe.space_fe.n_dofs
        n_t = self.space_time_fe.time_fe.n_dofs
        # Indicator dofs for either IC or BC in space-time
        ic_dofs_t = self.space_time_fe.time_fe.dof_IC_vector
        ic_dofs_scalar = np.kron(ic_dofs_t, np.ones((n_x,)))
        bd_dofs_x = self.space_time_fe.space_fe.boundary_dof_vector
        bd_dofs_scalar = np.kron(np.ones((n_t,)), bd_dofs_x)
        # Compatibility dofs (where IC and BC are both imposed)
        compat_dofs_scalar = np.logical_and(ic_dofs_scalar == 1, bd_dofs_scalar == 1)
        compat_dofs = np.tile(compat_dofs_scalar, 2)
        # Indicator IC or BC
        ic_bd_dofs_scalar = np.logical_or(ic_dofs_scalar, bd_dofs_scalar).astype(float)
        ic_bd_dofs = np.tile(ic_bd_dofs_scalar, 2)
        # Boundary dofs (Dirichlet)
        X_D = np.concatenate((boundary_data_u(xt_dofs), boundary_data_v(xt_dofs)))
        # Combined lifting for IC and BC. NB Remove X0 on compatibility dofs to avoind double imposition.
        X_0D = X0 + X_D - np.where(compat_dofs, X0, 0.0)
        rhs = rhs - sys_mat.dot(X_0D)
        # Impose Homogenoeous BC and IC
        rhs = rhs * (1 - ic_bd_dofs)
        sys_mat = sys_mat.multiply((1 - ic_bd_dofs).reshape((-1, 1)))
        sys_mat += scipy.sparse.diags(ic_bd_dofs, offsets=0, shape=sys_mat.shape)  # u
        sys_mat += scipy.sparse.diags(ic_bd_dofs, offsets=n_dofs_scalar, shape=sys_mat.shape)  # v
        return sys_mat, rhs, X_0D
