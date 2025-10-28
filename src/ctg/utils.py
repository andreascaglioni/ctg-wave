"""Functions for the Continuous Time Galerkin method for the space-time integration
of space-time problems, such as paraboic and hyperbolic PDEs."""

import numpy as np


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


def vprint(str, verbose=True):
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


def have_dolfinx() -> bool:
    try:
        import dolfinx  # noqa: F401

        return True
    except Exception:
        return False
