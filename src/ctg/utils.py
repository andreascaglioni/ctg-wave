"""Small utilities used across the CTG codebase.

This module provides compact, well-tested helpers that operate on FE
coordinates and on time slabs. The functions are intentionally simple and
have predictable behaviour to ease testing and documentation.
"""

import numpy as np


def cart_prod_coords(t_coords, x_coords):
    """Return the Cartesian product of time and space coordinate arrays.

    Args:
        t_coords: Array of time coordinates (1D or (n,1) shaped).
        x_coords: Array of space coordinates (1D or (m,1) shaped).

    Returns:
        2D array with rows representing pairs (t, x) from the Cartesian
        product of the inputs.
    """

    if len(x_coords.shape) == 1:  # coordinates i wrong format (rank 1 array). assume 1d.
        x_coords = np.expand_dims(x_coords, 1)
    if len(t_coords.shape) == 1:  # t coords in wron format assume 1d
        t_coords = np.expand_dims(t_coords, 1)
    long_t_coords = np.kron(t_coords, np.ones((x_coords.shape[0], 1)))
    long_x_coords = np.kron(np.ones((t_coords.shape[0], 1)), x_coords)
    return np.hstack((long_t_coords, long_x_coords))


def compute_time_slabs(start_time, end_time, slab_size):
    """Split a time interval into consecutive slabs.

    Args:
        start_time: Start of the time interval.
        end_time: End of the time interval.
        slab_size: Desired slab size (positive float).

    Returns:
        List of ``(start, end)`` tuples covering ``[start_time, end_time]``.
    """

    time_slabs = [(start_time, start_time + slab_size)]  # current time interval
    while time_slabs[-1][1] < end_time - 1e-10:
        time_slabs.append((time_slabs[-1][1], time_slabs[-1][1] + slab_size))
    return time_slabs


def have_dolfinx() -> bool:
    try:
        import dolfinx  # noqa: F401

        return True
    except Exception:
        return False
