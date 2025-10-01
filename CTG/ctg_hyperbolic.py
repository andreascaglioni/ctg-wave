import sys
import numpy as np
import scipy.sparse
from dolfinx import fem, mesh
import warnings

sys.path.append("./")
from CTG.FE_spaces import TimeFE, SpaceFE, SpaceTimeFE
from CTG.Assembler import AssemblerWave
from CTG.utils import compute_time_slabs
from CTG.FE_spaces import SpaceTimeFE
from CTG.post_process import plot_on_slab


def ctg_wave(physics_params, numerics_params, verbose=False):

    # Unpack data from dictionaries
    boundary_D = physics_params["boundary_D"]
    start_time = physics_params["start_time"]
    end_time = physics_params["end_time"]
    boundary_data_u = physics_params["boundary_data_u"]
    boundary_data_v = physics_params["boundary_data_v"]
    exact_rhs_0 = physics_params["exact_rhs_0"]
    exact_rhs_1 = physics_params["exact_rhs_1"]
    initial_data_u = physics_params["initial_data_u"]
    initial_data_v = physics_params["initial_data_v"]
    if "W_t" in physics_params:
        W_t = physics_params["W_t"]
    else:
        warnings.warn("W_t not provided in physics_params, setting W_t to None.")
        W_t = None
    comm = numerics_params["comm"]
    V_x = numerics_params["V_x"]
    t_slab_size = numerics_params["t_slab_size"]
    order_t = numerics_params["order_t"]

    time_slabs = compute_time_slabs(start_time, end_time, t_slab_size)
    space_fe = SpaceFE(V_x, boundary_D)

    # I need time_fe object over 1st time slab to determine tx_coords
    slab = time_slabs[0]
    msh_t = mesh.create_interval(comm, 1, [slab[0], slab[1]])
    V_t = fem.functionspace(msh_t, ("Lagrange", order_t))
    time_fe = TimeFE(msh_t, V_t)
    
    # Get coordinates initial condition
    space_time_fe = SpaceTimeFE(space_fe, time_fe)
    U0 = space_time_fe.interpolate(initial_data_u)  # DOFs vector
    V0 = space_time_fe.interpolate(initial_data_v)
    X0 = np.concatenate((U0, V0))
    
    # time stepping
    sol_slabs = []
    total_n_dofs = 0
    for i, slab in enumerate(time_slabs):
        if verbose:
            print(f"Slab_{i} = D x ({round(slab[0], 4)}, {round(slab[1], 4)}) ...")

        # Assemble time FE curr slab
        msh_t = mesh.create_interval(comm, 1, [slab[0], slab[1]])
        V_t = fem.functionspace(msh_t, ("Lagrange", order_t))
        time_fe = TimeFE(msh_t, V_t)

        # Update time_fe in space_time_fe and assemble all needed operators
        space_time_fe.update_time_fe(time_fe)
        total_n_dofs += space_time_fe.n_dofs
        space_time_fe.assemble(W_t)

        assembler = AssemblerWave(space_time_fe)
        sys_mat, rhs, X0D = assembler.assemble_system(W_t, X0, exact_rhs_0, exact_rhs_1,boundary_data_u, boundary_data_v)

        if sys_mat is None:
            raise RuntimeError("System matrix is None. Aborting computation.")
        
        # Solve
        X = scipy.sparse.linalg.spsolve(sys_mat, rhs)
        residual = np.linalg.norm(sys_mat.dot(X) - rhs) / np.linalg.norm(X)
        if verbose:
            print(f"Relative residual norm: {residual:.2e}")
        
        # Restore IC and BC
        X = X + X0D

        # plot_on_slab(space_fe.dofs, time_fe.dofs, X)
        
        sol_slabs.append(X)

        # Final condition (FC) becomes the IC on the next slab
        X0 = np.zeros_like(X0)
        X0[space_time_fe.dofs_IC]=X[space_time_fe.dofs_FC]
    
    # Return only the LAST space_time_fe
    return sol_slabs, time_slabs, space_time_fe, total_n_dofs