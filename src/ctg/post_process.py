import numpy as np
from dolfinx import fem, mesh

from ctg.FE_spaces import TimeFE


def compute_energy_tt(space_fe, sol_slabs):
    """
    Compute total, potential, and kinetic energy over time for a given solution.
    Args:
        space_fe: Finite element space object containing mass and stiffness matrices.
        sol_slabs: List of solution vectors at each time step.
        tt: Array of time points. Each point corresponds to  end of a time slab.
    Returns:
        Tuple of arrays: (total energy, potential energy, kinetic energy) at each time step.
    """

    M = space_fe.matrix["mass"]  # mass
    A = space_fe.matrix["laplace"]  # stiffness
    n_x = space_fe.n_dofs
    n_scalar = int(sol_slabs[0].size / 2)
    eenergy = np.zeros(len(sol_slabs))
    ppot = np.zeros(len(sol_slabs))  # potential energy
    kkin = np.zeros(len(sol_slabs))  # kinetic energy

    for i, X in enumerate(sol_slabs):
        u = X[0:n_x]
        v = X[n_scalar : n_scalar + n_x]
        ppot[i] = u @ A @ u
        kkin[i] = v @ M @ v
        eenergy[i] = ppot[i] + kkin[i]

    return eenergy, ppot, kkin


def inverse_DS_transform(XX, WW_fun, space_fe, time_slab, comm, order_t):
    """Apply DS transform for single time"""

    msh_t = mesh.create_interval(comm, 1, [time_slab[0], time_slab[1]])
    V_t = fem.functionspace(msh_t, ("Lagrange", order_t))
    time_fe = TimeFE(V_t)
    n_scalar = int(XX.size / 2)
    n_x = space_fe.n_dofs
    uu = XX[:n_scalar]
    vv = XX[n_scalar:]
    WW = WW_fun(time_fe.dofs)
    WW_rep = np.repeat(WW, n_x)
    return np.concatenate((uu, vv + WW_rep * uu))
