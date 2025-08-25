"""Implmenet CTG for 1st order formulation of the wave equation.
Use separate variables for u and v instead of a 2d vectorial unknown."""

import sys
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import scipy.sparse
from dolfinx import fem, mesh

from CTG.utils import float_f, compute_time_slabs, compute_rate, cart_prod_coords, compute_error_slab
from CTG.FE_spaces import SpaceFE, TimeFE



def _impose_initial_conditions(sys_mat, rhs, space_fe, indic_t0_test, X0):

    # Get vectorial dofs IC
    n_dofs_trial = sys_mat.shape[1]
    n_x = space_fe.n_dofs
    n_dofs_trial_scalar = int(n_dofs_trial/2)

    # Lift IC
    rhs = rhs - sys_mat.dot(X0)
    
    # Impose Homogenous IC on rhs




# using indic_t0_test makes no sense! the fos are located in the midpoints of the elements!!!







    ones_x = np.ones((n_x, ))
    indic_t0_tx_test_scalar = np.kron(indic_t0_test, ones_x)
    indic_t0_tx_test = np.tile(indic_t0_tx_test_scalar, 2)  # For both u and v
    rhs = rhs * indic_t0_tx_test

    # Impose Homogenous IC on matrix
    sys_mat = sys_mat.multiply(indic_t0_tx_test.reshape((-1, 1)))
    sys_mat += scipy.sparse.diags(indic_t0_tx_test, offsets=0, shape=sys_mat.shape)  # u
    sys_mat += scipy.sparse.diags(indic_t0_tx_test, offsets=n_dofs_trial_scalar, shape=sys_mat.shape)  # v

    return sys_mat, rhs, X0


def _impose_boundary_conditions(sys_mat, rhs, time_fe, indic_bd, boundary_data_u, boundary_data_v, xt_dofs):
    n_dofs_trial_scalar = int(sys_mat.shape[1] / 2)
    n_t_test = time_fe.n_dofs_test

    # Lift BC
    u_D = boundary_data_u(xt_dofs)
    v_D = boundary_data_v(xt_dofs)
    X_D = np.concatenate((u_D, v_D))
    rhs = rhs - sys_mat.dot(X_D)

    # Impose Homogenoeous BC on rhs
    ones_t_test = np.ones((n_t_test, ))
    indic_bd_tx_test_scalar = np.kron(ones_t_test, indic_bd)
    indic_bd_tx_test = np.tile(indic_bd_tx_test_scalar, 2)
    rhs = rhs * indic_bd_tx_test

    # Impose homogeneous BC on matrix
    sys_mat = sys_mat.multiply(indic_bd_tx_test.reshape((-1, 1)))
    sys_mat += scipy.sparse.diags(indic_bd_tx_test, offsets=0, shape=sys_mat.shape)  # u
    sys_mat += scipy.sparse.diags(indic_bd_tx_test, offsets=n_dofs_trial_scalar, shape=sys_mat.shape)  # v

    return sys_mat, rhs, X_D


def assemble(space_fe, time_fe, boundary_data_u, boundary_data_v, X0):
    # Space-time matrices for scalar unknowns
    mass_mat = scipy.sparse.kron(time_fe.matrix["mass"], space_fe.matrix["mass"])
    stiffness_mat = scipy.sparse.kron(
        time_fe.matrix["mass"], space_fe.matrix["laplace"]
    )
    derivative_mat = scipy.sparse.kron(
        time_fe.matrix["derivative"], space_fe.matrix["mass"]
    )

    # Space-time matrices for vectorial unknowns
    derivative_mat_2 = scipy.sparse.block_diag((derivative_mat, derivative_mat))
    sys_mat_2 = scipy.sparse.block_array([[None, -mass_mat], [-stiffness_mat, None]])
    sys_mat = derivative_mat_2 + sys_mat_2

    # Right hand side vector
    xt_dofs = cart_prod_coords(time_fe.dofs_trial, space_fe.dofs)
    rhs0 = mass_mat.dot(exact_rhs_0(xt_dofs))
    rhs1 = mass_mat.dot(exact_rhs_1(xt_dofs))
    rhs = np.concatenate((rhs0, rhs1))

    # Impose IC
    boundary_IC = lambda t : np.isclose(t[0], slab[0])  # noqa: E731
    indic_t0_trial, indic_t0_test = time_fe.get_IC_dofs(boundary_IC)
    sys_mat, rhs, x_IC = _impose_initial_conditions(sys_mat, rhs, space_fe, indic_t0_test, X0)
    
    # Impose BC
    sys_mat, rhs, x_D = _impose_boundary_conditions(
        sys_mat,
        rhs,
        time_fe,
        space_fe.boundary_dof_vector,  # Indicator fun. on dof vector
        boundary_data_u, 
        boundary_data_v,
        xt_dofs,
    )
    
    return sys_mat, rhs, x_IC + x_D


if __name__ == "__main__":
    # SETTINGS
    seed = 0
    np.random.seed(0)
    comm = MPI.COMM_SELF
    np.set_printoptions(formatter={"float_kind": float_f})

    # PARAMETERS
    # space
    order_x = 1
    n_cells_space = 4
    msh_x = mesh.create_unit_interval(comm, n_cells_space)
    order_x = 1
    boundary_D = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))  # noqa: E731
    V_x = fem.functionspace(msh_x, ("Lagrange", 1, (1,)))  # 1d space

    # time
    start_time = 0.0
    end_time = 1.0
    t_slab_size = 0.1
    order_t = 1

    # Exact sol
    from data.exact_solution_wave_sep import (
        exact_sol_u,
        # exact_sol_v,
        exact_rhs_0,
        exact_rhs_1,
        boundary_data_u,
        boundary_data_v,
        initial_data_u,
        initial_data_v,
    )

    # error
    err_type_x = "h1"
    err_type_t = "linf"

    # COMPUTE
    time_slabs = compute_time_slabs(start_time, end_time, t_slab_size)
    t0 = time_slabs[0][0]

    space_fe = SpaceFE(msh_x, V_x, boundary_D)
    n_x = space_fe.n_dofs

    # Vector of dofs IC (over first slab)
    tx_coords = cart_prod_coords(np.array(time_slabs[0]), space_fe.dofs)  # shape (n_dofs_tx_scalar, 2) 
    u0 = initial_data_u(tx_coords)  # shape (n_dofs_tx_scalar, )
    v0 = initial_data_v(tx_coords)  # shape (n_dofs_tx_scalar, )
    X0 = np.concatenate((u0, v0))  # shape (2*n_dofs_tx_scalar, )
    
    # time stepping
    total_n_dofs_t = 0
    sol_slabs = []
    err_slabs = -1.0 * np.ones(len(time_slabs))
    norm_u_slabs = -1.0 * np.ones_like(err_slabs)
    for i, slab in enumerate(time_slabs):
        print(f"Slab_{i} = D x ({round(slab[0], 4)}, {round(slab[1], 4)}) ...")

        # Assemble time FE curr slab
        msh_t = mesh.create_interval(comm, 1, [slab[0], slab[1]])
        V_t_trial = fem.functionspace(msh_t, ("Lagrange", order_t))
        V_t_test = fem.functionspace(msh_t, ("DG", order_t - 1))
        time_fe = TimeFE(msh_t, V_t_trial, V_t_test)
        n_t = time_fe.n_dofs_trial
        total_n_dofs_t += n_t
        n_dofs_scalar = n_t*n_x
        n_dofs = 2*n_dofs_scalar

        # Assemble space-time linear system
        sys_mat, rhs, x_D = assemble(space_fe, time_fe, boundary_data_u, boundary_data_v, X0)

        assert sys_mat.shape[1] == n_dofs

        # Solve
        x, info = scipy.sparse.linalg.lsqr(sys_mat, rhs)[:2]
        x = x + x_D  # add IC and BC
        sol_slabs.append(x)

        # Extract IC dofs next time slab
        X0 = np.zeros_like(X0)
        dof_max_t = time_fe.dofs_trial.argmax()  # index of dof giving max t current slab
        dofs_tx_max_t = np.arange(dof_max_t*n_x, (dof_max_t+1)*n_x)  # the t-x dofs with t=tmax
        X0[:n_x] = x[dofs_tx_max_t]
        dofs_tx_max_t = np.arange(n_dofs_scalar+dof_max_t*n_x, n_dofs_scalar+(dof_max_t+1)*n_x)
        X0[n_dofs_scalar:n_dofs_scalar+n_x] = x[dofs_tx_max_t]

        # Error on u on current slab
        coords_u_slab = x[:n_dofs_scalar]
        exact_u = lambda X: exact_sol_u(X)  # noqa: E731
        err_slabs[i], norm_u_slabs[i] = compute_error_slab(space_fe, exact_u, err_type_x, err_type_t, time_fe, coords_u_slab)

        print("Current " + err_type_t + "-" + err_type_x + " error:  ",float_f(err_slabs[i]))
        print("Current " + err_type_t + "-" + err_type_x + " rel err:", float_f(err_slabs[i] / norm_u_slabs[i]))
        print("Done.\n")
    n_dofs = space_fe.n_dofs * total_n_dofs_t

    # POST-PROCESS
    if err_type_t == "linf":
        total_err = np.amax(err_slabs)
        total_norm_u = np.amax(norm_u_slabs)
    elif err_type_t == "l2":
        total_err = sqrt(np.sum(np.square(err_slabs)))
        total_norm_u = sqrt(np.sum(np.square(norm_u_slabs)))
    else:
        raise ValueError(f"Unknown error type in time: {err_type_t}")
    
    total_rel_err = total_err / total_norm_u
    print("Total error", float_f(total_err), "Total relative error", float_f(total_rel_err))
    print("error over slabls", err_slabs)

    # Plot relative error over time slabs
    times = [slab[1] for slab in time_slabs]
    rel_errs = err_slabs / norm_u_slabs
    plt.figure()
    plt.plot(times, rel_errs, marker='o')
    plt.xlabel("Time")
    plt.ylabel("Relative Error")
    plt.title("Relative Error over Time Slabs")
    plt.grid(True)
    plt.show()

    # Plot u at beginning of each time slab

    # n_dofs_tx_scalar = int(dofs_X_slabs[0].size / 2)
    # xx = space_fe.dofs  # msh_x.geometry.x[:, 0]  # msh_x.geometry.x has shape (# nodes, 3)
    # dt = t_slab_size / n_time
    # n_dofs_t = n_time+1
    # n_dofs_x = space_fe.n_dofs
    
    xx = space_fe.dofs
    
    for i, slab in enumerate(time_slabs):
        x = sol_slabs[i]
        u = x[:n_dofs_scalar]        
        n_dofs = x.size
        n_dofs_scalar = int(n_dofs / 2)
        n_t = n_dofs_scalar / n_x
        for i_t, t in enumerate(slab):
            u_t = u[i_t * n_x : (i_t + 1) * n_x]            
            plt.plot(xx, u_t, "o")

            X = cart_prod_coords(np.array([t]), xx)
            u_ex = exact_sol_u(X)
            plt.plot(xx, u_ex, ".-")

            plt.title(f"Slabs[{i}] = {np.round(slab, 2)}; time {i_t}")
            plt.ylim([-1, 1])
            plt.show()


