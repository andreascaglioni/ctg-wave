"""Space-time CTG solver entry point.

This module exposes :class:`CTGSolver` which orchestrates the space-time
finite-element solution of the stochastic wave equation used across the
project. The implementation is intentionally thin: it wires together the
finite-element spaces, the assembler and the time-stepping loop.

See ``docs/intro.rst`` for a high-level description of the method and
algorithmic choices.
"""

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
        """Create a CTGSolver.

        Args:
            numerics_params: Object with numeric settings (must expose
                attributes such as ``comm``, ``n_cells_space``,
                ``order_x``, ``order_t`` and ``t_slab_size``).
        """
        self.verbose = numerics_params.verbose

        self.comm = numerics_params.comm
        self.t_slab_size = numerics_params.t_slab_size
        self.order_t = numerics_params.order_t

        # Define mesh and FEM for space (no BC)
        msh_x = mesh.create_unit_interval(self.comm, numerics_params.n_cells_space)
        self.V_x = fem.functionspace(msh_x, ("Lagrange", numerics_params.order_x, (1,)))

    def run(self, phy_p, W_t=None):
        """Run the solver over the configured time interval.

        This method builds the space/time FE spaces, interpolates initial
        conditions and advances the solution through all time slabs.

        Args:
            phy_p: Physics configuration object providing callables for
                initial/boundary data and rhs terms (see ``ctg.config``).
            W_t: Optional callable representing the time-dependent
                coefficient (typically a Brownian path). If ``None`` the
                run is deterministic.

        Returns:
            A tuple ``(sol_slabs, time_slabs, space_time_fe, total_n_dofs)``
            where ``sol_slabs`` is the list of slab solutions and
            ``space_time_fe`` is the last used FE space object.
        """
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
        """Assemble and solve the linear system for a single time slab.

        This method updates the slab-specific temporal FE, assembles the
        space-time matrices via the project assembler and solves the linear
        system. It returns the number of dofs and the flattened solution
        vector for the slab.

        Args:
            bc_u, bc_v: Boundary data callables for displacement and velocity.
            rhs_0, rhs_1: Right-hand side callables.
            W_t: Time-dependent coefficient callable or ``None``.
            slab: Tuple ``(t0, t1)`` for the current time slab.
            space_time_fe: Instance of :class:`SpaceTimeFE`.
            X0: Flattened vector with initial and final condition dofs.

        Returns:
            A tuple ``(n_dofs, X)`` with current slab dofs and solution
            vector ``X`` (numpy array).
        """
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
