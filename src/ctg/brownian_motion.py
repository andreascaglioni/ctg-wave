import numpy as np
from math import ceil, log, sqrt


def param_LC_W(yy, tt, T):
    """Pythonic computation of the LC expansion of the Wiener process.

    Args:
        yy (numpy.ndarray[float]): Parameter vector for the expansion.
        tt (numpy.ndarray[float]): 1D array of discrete times in [0, T].
        T (float): Final time of approximation.

    Returns:
        numpy.ndarray[float]: 2D array. Each *ROW* is a sample path of W over tt.
    """

    # Make yy 2D array (Ny, dimy)
    yy = np.atleast_2d(yy)
    assert len(yy.shape) == 2, "param_LC_Brownian_motion: yy must be 2D"

    # Make tt 1D (Nt, )
    tt = np.atleast_1d(tt)
    tt = tt.flatten()
    assert len(tt.shape) == 1
    tol = 1e-12
    assert np.amin(tt) >= -tol and np.amax(tt) <= T + tol, "param_LC_Brownian_motion: tt not within [0,T] (with tolerance)"


    # Get # LC-levels
    L = ceil(log(yy.shape[1], 2))  # number of levels

    # Extend yy to the next power of 2 with 0
    fill = np.zeros((yy.shape[0], 2**L - yy.shape[1]))
    yy = np.column_stack((yy, fill))
    BB = LC_matrix(L, tt, T)
    W = np.matmul(yy, BB)
    return W

def LC_matrix(L: int, tt: np.ndarray, T: float) -> np.ndarray:
    """
    Generate matrix for sampling Levy-Ciesielski expansion of level L  on [0, T] on time nodes tt.
    Args:
        L (int): Number of levels in the Levy-Ciesieslky expansion. We
        tt (np.ndarray): 1D array of time points at which to evaluate the basis functions.
        T (float): Final time, used to rescale the time points.
    Returns:
        np.ndarray: A (dim_y, len(tt)) matrix where each row is a basis function evaluated at the given time points.
    """


    n_t = tt.size
    # Rescale tt (to be reverted!)
    tt = tt / T
    # Compute basis B
    BB = np.zeros((2**L, n_t))
    BB[0, :] = tt  # first basis function is the linear one
    for lev in range(1, L + 1):
        n_j = 2 ** (lev - 1)  # number of basis functions at level l
        for j in range(1, n_j + 1):
            basis_fun = 0 * tt  # basis is 0 where not assegned below
            # define increasing part basis function
            ran1 = np.where(
                (tt >= (2 * j - 2) / (2**lev)) & (tt <= (2 * j - 1) / (2**lev))
            )
            basis_fun[ran1] = tt[ran1] - (2 * j - 2) / 2**lev
            # define decreasing part basis function
            ran2 = np.where((tt >= (2 * j - 1) / (2**lev)) & (tt <= (2 * j) / (2**lev)))
            basis_fun[ran2] = -tt[ran2] + (2 * j) / 2**lev
            n_b = 2 ** (lev - 1) + j - 1  # prev. lev.s (complete) + curr. lev (partial)
            BB[n_b, :] = 2 ** ((lev - 1) / 2) * basis_fun
    # Revert time rescaling
    BB *= sqrt(T)   
    return BB