import numpy as np
import scipy
import sys
sys.path.insert(0, "./")
from CTG.FE_spaces import SpaceTimeFE
from data.data_param_wave_eq import boundary_data_u, boundary_data_v



class AssemblerWave:
    """Assembles linear system for the CTG space-time solver. In particular:
    * Get operators from SpaceTimeFE 
    * Impose initial and boundary conditions to compute the homogeneous system (to be solved) an the lifting function.

    Some attributes are set only upon construction: space_fe, exact_rhs_0, exact_rhs_1, boundary_data_u, boundary_data_v, initial_data_u, initial_data_v.
    Others, are set by setter methods because they will change over time slabs or over parameters:
        * Changing over time slabs: X0, time_fe
        * Changing over parameters: Ay, A"""

    def __init__(self, space_time_fe: SpaceTimeFE | None = None):
        if space_time_fe is not None:
            self.update_space_time_fe(space_time_fe)
        else:
            self.space_time_fe = None
            self.A0_no_bc = None
            self.b_no_bc = None

    def update_space_time_fe(self,
                             space_time_fe: SpaceTimeFE):
        """Update SpaceTimeFE instance. Use it moving to a new time step."""
        self.space_time_fe = space_time_fe

    def assemble_A0_b(self, exact_rhs_0, exact_rhs_1):
        """Assemble parameter-independent part of the linear system."""
        if self.space_time_fe is None:
            print("Warning: self.space_time_fe is None. Assembly interrupted.")
            return None, None
        # Matrix
        L_mat = self.space_time_fe.matrix["L"]
        D_mat = self.space_time_fe.matrix["D_t"]
        M_xt = self.space_time_fe.matrix["M"]
        A0_no_bc = scipy.sparse.block_array([[D_mat, -M_xt], [L_mat, D_mat]])
        # L_mat goes with + sign becaus eit actualkly represent grad-grad term
        # Right hand side vector
        xt_dofs = self.space_time_fe.dofs
        rhs0 = M_xt.dot(exact_rhs_0(xt_dofs))
        rhs1 = M_xt.dot(exact_rhs_1(xt_dofs))
        b_no_bc = np.concatenate((rhs0, rhs1))
        return A0_no_bc, b_no_bc

    def assemble_A_W(self, W_t):
        """Assemble parameter-dependent part of the linear system."""
        if self.space_time_fe is None:
            print("Warning: self.space_time_fe is None. Assembly interrupted.")
            return None
        self.space_time_fe.assemble_W(W_t)
        M_W = self.space_time_fe.matrix["M_W"]
        M_W2 = self.space_time_fe.matrix["M_WW"]
        return scipy.sparse.block_array([[-M_W, None], [-M_W2, -M_W]])

    def assemble_system(self,
                        W_t,
                        X0: np.ndarray,
                        exact_rhs_0,
                        exact_rhs_1):
        
        # Assemble linear system operator
        A_W_no_bc = self.assemble_A_W(W_t)
        A0_no_bc, b_no_bc = self.assemble_A0_b(exact_rhs_0, exact_rhs_1)
        # Check if assembly was successful
        if A_W_no_bc is None or A0_no_bc is None or b_no_bc is None:
            print("Warning: Assembly failed. Returning None.")
            return None, None, None
        A = A0_no_bc + A_W_no_bc
        # Get homogenous equation and initial-(Dirichlet) boundary condition X0D for lifting
        A, b, X0D = self.impose_IC_BC(A, b_no_bc, X0)
        return A, b, X0D

    def impose_IC_BC(self, sys_mat, rhs, X0):
        if self.space_time_fe is None or self.space_time_fe.time_fe is None:
            print("Warning: self.space_time_fe is None. Skipping impose_IC_BC.")
            return None, None, None

        xt_dofs = self.space_time_fe.dofs
        n_dofs_scalar = int(sys_mat.shape[1]/2)
        n_x = self.space_time_fe.space_fe.n_dofs
        n_t = self.space_time_fe.time_fe.n_dofs
        # Indicator dofs IC nad boundary condition (BC) in space-time
        ic_dofs_t = self.space_time_fe.time_fe.dof_IC_vector
        ic_dofs_scalar = np.kron(ic_dofs_t, np.ones((n_x, )))
        bd_dofs_x = self.space_time_fe.space_fe.boundary_dof_vector
        bd_dofs_scalar = np.kron(np.ones((n_t, )), bd_dofs_x)
        # Compatibility dofs (where IC and BC are both imposed)
        compat_dofs_scalar = np.logical_and(ic_dofs_scalar == 1, bd_dofs_scalar == 1)
        # Correspodning indicatiors for vectorial functions
        ic_bd_dofs_scalar = np.logical_or(ic_dofs_scalar, bd_dofs_scalar).astype(float)
        ic_bd_dofs = np.tile(ic_bd_dofs_scalar, 2)
        compat_dofs = np.tile(compat_dofs_scalar, 2)
        # Boundary dofs (Dirichlet)
        X_D = np.concatenate((boundary_data_u(xt_dofs), boundary_data_v(xt_dofs)))
        # Combined lifting for IC and BC. NB Remove X0 on compatibility dofs to avoind double imposition.
        X_0D = X0 + X_D - np.where(compat_dofs, X0, 0.)
        rhs = rhs - sys_mat.dot(X_0D)
        # Impose Homogenoeous BC and IC
        rhs = rhs * (1-ic_bd_dofs)
        sys_mat = sys_mat.multiply((1-ic_bd_dofs).reshape((-1, 1)))
        sys_mat += scipy.sparse.diags(ic_bd_dofs, offsets=0, shape=sys_mat.shape)  # u
        sys_mat += scipy.sparse.diags(ic_bd_dofs, offsets=n_dofs_scalar, shape=sys_mat.shape)  # v
        return sys_mat, rhs, X_0D
    


# Legacy code
    # def assemble_ctg(space_fe, time_fe, boundary_data_u, boundary_data_v, X0, exact_rhs_0, exact_rhs_1, W_path):
    #     # Space-time matrices for scalar unknowns
    #     mass_mat = scipy.sparse.kron(time_fe.matrix["mass"], space_fe.matrix["mass"])
    #     W_mass_mat = scipy.sparse.kron(time_fe.matrix["W_mass"], space_fe.matrix["mass"])
    #     WW_mass_mat = scipy.sparse.kron(time_fe.matrix["WW_mass"], space_fe.matrix["mass"])
    #     stiffness_mat = scipy.sparse.kron(
    #         time_fe.matrix["mass"], space_fe.matrix["laplace"]
    #     )
    #     derivative_mat = scipy.sparse.kron(
    #         time_fe.matrix["derivative"], space_fe.matrix["mass"]
    #     )

    #     # Space-time matrices for vectorial unknowns
    #     sys_mat = scipy.sparse.block_array([[derivative_mat, -mass_mat], [stiffness_mat, derivative_mat]])
    #     # the next term from the PWE
    #     sys_mat += scipy.sparse.block_array([[W_mass_mat, None], [WW_mass_mat, W_mass_mat]])

    #     # Right hand side vector
    #     xt_dofs = cart_prod_coords(time_fe.dofs, space_fe.dofs)
    #     rhs0 = mass_mat.dot(exact_rhs_0(xt_dofs))
    #     rhs1 = mass_mat.dot(exact_rhs_1(xt_dofs))
    #     rhs = np.concatenate((rhs0, rhs1))

    #     # Impose IC+BC
    #     sys_mat, rhs, X0D = impose_IC_BC(sys_mat, rhs, space_fe, time_fe, boundary_data_u, boundary_data_v, X0)

    #     return sys_mat, rhs, X0D