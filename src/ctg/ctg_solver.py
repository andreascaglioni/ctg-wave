import sys
import numpy as np
import scipy.sparse
from dolfinx import fem, mesh
import warnings
import logging

sys.path.append("./")
from ctg.FE_spaces import TimeFE, SpaceFE, SpaceTimeFE
from ctg.Assembler import AssemblerWave
from ctg.utils import compute_time_slabs

logger = logging.getLogger(__name__)


class CTGSolver:
    def __init__(self, numerics_params):
        self.verbose = numerics_params.verbose

        self.comm = numerics_params.comm
        self.t_slab_size = numerics_params.t_slab_size
        self.order_t = numerics_params.order_t

        # Define mesh and FEM for space (no BC)
        msh_x = mesh.create_unit_interval(self.comm, numerics_params.n_cells_space)
        self.V_x = fem.functionspace(msh_x, ("Lagrange", numerics_params.order_x, (1,)))

    def run(self, phy_p, W_t=None):

        if W_t is None:
            warnings.warn("W_t not provided in physics_params, running deterministic WE.")

        time_slabs = compute_time_slabs(phy_p.start_time, phy_p.end_time, self.t_slab_size)
        space_fe = SpaceFE(self.V_x, phy_p.boundary_D)

        # Define space-time FEM space
        slab = time_slabs[0]
        msh_t = mesh.create_interval(self.comm, 1, [slab[0], slab[1]])
        V_t = fem.functionspace(msh_t, ("Lagrange", self.order_t))
        time_fe = TimeFE(V_t)
        space_time_fe = SpaceTimeFE(space_fe, time_fe)

        # Compute coordinates initial condition
        U0 = space_time_fe.interpolate(phy_p.initial_data_u)  # DOFs vector
        V0 = space_time_fe.interpolate(phy_p.initial_data_v)
        X0 = np.concatenate((U0, V0))

        # time stepping
        sol_slabs = []
        total_n_dofs = 0
        for i, slab in enumerate(time_slabs):
            logger.info(f"Slab_{i} = D x ({round(slab[0], 4)}, {round(slab[1], 4)}) ...")

            # Iterate
            n_dofs_curr, X = self.iterate(
                phy_p.boundary_data_u,
                phy_p.boundary_data_v,
                phy_p.rhs_0,
                phy_p.rhs_1,
                W_t,
                slab,
                space_time_fe,
                X0,
            )

            total_n_dofs += n_dofs_curr
            sol_slabs.append(X)

            # Final condition (FC) becomes IC on next slab
            X0 = np.zeros_like(X0)
            X0[space_time_fe.dofs_IC] = X[space_time_fe.dofs_FC]

        return sol_slabs, time_slabs, space_time_fe, total_n_dofs

    def iterate(self, bc_u, bc_v, rhs_0, rhs_1, W_t, slab, space_time_fe, X0):

        # Assemble time FE curr slab
        # TODO factory for time_fe given slab, fem degree, fem type. Make time_fe factory as class.,__init__ input
        msh_t = mesh.create_interval(self.comm, 1, [slab[0], slab[1]])
        V_t = fem.functionspace(msh_t, ("Lagrange", self.order_t))
        time_fe = TimeFE(V_t)

        # Update time_fe in space_time_fe and assemble all needed operators
        space_time_fe.update_time_fe(time_fe)
        space_time_fe.assemble(W_t)

        # TODO make class member and use AssemblerWave.update_space_time_fe
        logger.info("\tAssembling space-time system")
        assembler = AssemblerWave(space_time_fe)
        sys_mat, rhs, X0D = assembler.assemble_system(W_t, X0, rhs_0, rhs_1, bc_u, bc_v)
        if sys_mat is None:
            raise RuntimeError("System matrix is None. Aborting computation.")

        # Solve
        logger.info("\tSolving space-time system")
        X = scipy.sparse.linalg.spsolve(sys_mat, rhs)
        # add self.debug and run this only if self.debug=True
        residual = np.linalg.norm(sys_mat.dot(X) - rhs) / np.linalg.norm(X)
        logger.info(f"\tRelative residual norm: {residual:.2e}")

        # Re-introduce IC and BC
        X = X + X0D

        return space_time_fe.n_dofs, X
