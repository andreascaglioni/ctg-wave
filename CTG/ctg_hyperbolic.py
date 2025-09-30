import sys
import numpy as np
import scipy.sparse
from dolfinx import fem, mesh

sys.path.append("./")
from CTG.FE_spaces import TimeFE, SpaceFE
from CTG.assemble_XT import assemble_ctg
from CTG.utils import cart_prod_coords, compute_time_slabs
import warnings

def ctg_wave(physics_params, numerics_params, verbose=False):

    # Unpack inputs from dictionaries
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

    # Since W_t is determined, assing W-rependent matrices
    time_fe.assemble_matrices_W(W_t)

    tx_coords = cart_prod_coords(time_fe.dofs, space_fe.dofs)  # shape (n_dofs_tx_scalar, 2) 
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
        V_t = fem.functionspace(msh_t, ("Lagrange", order_t))
        time_fe = TimeFE(msh_t, V_t)

        # Compute n dofs curr slab
        total_n_dofs_t += time_fe.n_dofs

        # Compute W-dependent t operators curr slab
        time_fe.assemble_matrices_W(W_t)

        # Assemble space-time linear system
        sys_mat, rhs, X0D = assemble_ctg(space_fe, time_fe, boundary_data_u, boundary_data_v, X0, exact_rhs_0, exact_rhs_1, W_t)

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

    # Return only the LAST time_fe
    return time_slabs, space_fe, sol_slabs, total_n_dofs, time_fe