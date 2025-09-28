"""Functions for the Continuous Time Galerkin method for the space-time integration
of space-time problems, such as paraboic and hyperbolic PDEs."""

from math import sqrt, ceil, log, sqrt, pi
import numpy as np
import sys

sys.path.append("./")
import copy


def cart_prod_coords(t_coords, x_coords):
    if len(x_coords.shape) == 1:  # coordinates i wrong format (rank 1 array). assume 1d.
        x_coords = np.expand_dims(x_coords, 1)
    if len(t_coords.shape) == 1:  # t coords in wron format assume 1d
        t_coords = np.expand_dims(t_coords, 1)
        
    long_t_coords = np.kron(t_coords, np.ones((x_coords.shape[0], 1)))
    long_x_coords = np.kron(np.ones((t_coords.shape[0], 1)), x_coords)
    return np.hstack((long_t_coords, long_x_coords))


def compute_time_slabs(start_time, end_time, slab_size):
    time_slabs = [(start_time, start_time + slab_size)]  # current time interval
    while time_slabs[-1][1] < end_time - 1e-10:
        time_slabs.append((time_slabs[-1][1], time_slabs[-1][1] + slab_size))
    return time_slabs



def inverse_DS_transform(xx, WW_fun, space_fe, time_fe):
    """ Apply the inverse Doss-Sussmann transform to obtain a sample solution of the Stochastic Wave Equation (SWE) 
    from a sample solution of the Random Wave Equation (RWE), or equivalently, a solution of the Parametric Wave Equation (PWE) 
    evaluated for a random standard Gaussian vector.

    Args:
        xx (np.ndarray): Coordinates (in space time FE basis) of a sample solution of the RWE (or PWE evaluated at a random standard Gaussian vector) over 1 space-time slab. First havlf is u, second is v = partial_t u. For both u and v, coords are 1st wrt time basis, then space basis (using tensor prodcut space time FEM basis).
        WW (np.array): Samples of Brownian motion corresonding to xx at discrete times tt (same as space time basis).
        space_fe(SpaceFE): Class for space finite elements.
        time_fe(TimeFE): Class for time finite elements. It must correspond to the same time slab as xx.

    Returns:
        np.ndarray: Sample solution of the SWE obtained via the inverse Doss-Sussmann transform.
    """

    n_scalar = int(xx.size/2)
    n_x = space_fe.n_dofs
    uu = xx[:n_scalar]
    vv = xx[n_scalar:]
    WW = WW_fun(time_fe.dofs_trial)
    WW_rep = np.repeat(WW, n_x)
    return np.concatenate((uu, vv + WW_rep*uu))




