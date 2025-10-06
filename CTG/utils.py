"""Functions for the Continuous Time Galerkin method for the space-time integration
of space-time problems, such as paraboic and hyperbolic PDEs."""

import numpy as np
import sys
sys.path.append("./")


def cart_prod_coords(t_coords, x_coords):
    """
    Computes cartesian product of two coordinate arrays.
    Args:
        t_coords (np.ndarray): Column 2D array of time coordinates.
        x_coords (np.ndarray): Column 2D array of spatial coordinates.
    Returns:
        np.ndarray: 2D array where each row is a pair (t, x) from the cartesian product of t_coords and x_coords.
    """

    if len(x_coords.shape) == 1:  # coordinates i wrong format (rank 1 array). assume 1d.
        x_coords = np.expand_dims(x_coords, 1)
    if len(t_coords.shape) == 1:  # t coords in wron format assume 1d
        t_coords = np.expand_dims(t_coords, 1)
    long_t_coords = np.kron(t_coords, np.ones((x_coords.shape[0], 1)))
    long_x_coords = np.kron(np.ones((t_coords.shape[0], 1)), x_coords)
    return np.hstack((long_t_coords, long_x_coords))


def compute_time_slabs(start_time, end_time, slab_size):
    """
    Divides a time interval into consecutive slabs of a given size.
    Args:
        start_time (float): The starting time of the interval.
        end_time (float): The ending time of the interval.
        slab_size (float): The size of each time slab.
    Returns:
        list of tuple: List of (start, end) tuples for each time slab.
    """

    time_slabs = [(start_time, start_time + slab_size)]  # current time interval
    while time_slabs[-1][1] < end_time - 1e-10:
        time_slabs.append((time_slabs[-1][1], time_slabs[-1][1] + slab_size))
    return time_slabs

def inverse_DS_transform(XX, WW_fun, space_fe, time_fe):
    """Apply the inverse Doss-Sussmann transform to a sample solution of the parametric Andreson model (a parametric wave equation) to obtain a sample solution of the stochastic Anderson model (a stochastic wave equation). 

    Args:
        xx (np.ndarray): Coordinates (in space time FE basis) of a sample solution of the RWE (or PWE evaluated at a random standard Gaussian vector) over 1 space-time slab. First havlf is u, second is v = partial_t u. For both u and v, coords are 1st wrt time basis, then space basis (using tensor prodcut space time FEM basis).
        WW_fun (Callable[[np.ndarray], np.ndarray]): A callable that takes an array of times and returns the values of a (LC-expansion of the) Brownian motion at those times.
        space_fe(SpaceFE): Class for space finite elements.
        time_fe(TimeFE): Class for time finite elements corresponding to the same time slab as xx.

    Returns:
        np.ndarray: Sample solution of the SWE obtained via the inverse Doss-Sussmann transform.
    """

    n_scalar = int(XX.size/2)
    n_x = space_fe.n_dofs
    uu = XX[:n_scalar]
    vv = XX[n_scalar:]
    WW = WW_fun(time_fe.dofs)
    WW_rep = np.repeat(WW, n_x)
    return np.concatenate((uu, vv + WW_rep*uu))

def vprint(str, verbose = True):
    """
    Prints the given string if verbose is True.
    Args:
        str (str): The string to print.
        verbose (bool, optional): If True, prints the string. Defaults to True.
    Returns:
        None
    """

    if verbose:
        print(str)