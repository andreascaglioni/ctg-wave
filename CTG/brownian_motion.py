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

    # Check input shape and make it 2D
    if yy.size == 1 or yy is int:  # 1-element array
        yy = np.array([yy], dtype=float).reshape((1, 1))
    if len(yy.shape) == 1:  # 1 parameter vector
        yy = np.array([yy], dtype=float).reshape((1, yy.size))
    assert len(yy.shape) == 2, (
        "param_LC_Brownian_motion: yy must be 2D (1 ROW per sample array)"
    )

    if tt is int:
        tt = np.array([tt])
    tt = tt.flatten()
    tol = 1e-12
    assert np.amin(tt) >= -tol and np.amax(tt) <= T + tol, (
        "param_LC_Brownian_motion: tt not within [0,T] (with tolerance)"
    )


    # Get # LC-levels
    L = ceil(log(yy.shape[1], 2))  # levels

    # Extend yy to the next power of 2
    fill = np.zeros((yy.shape[0], 2**L - yy.shape[1]))
    yy = np.column_stack((yy, fill))

    (n_y, dim_y) = yy.shape
    n_t = tt.size

    # Rescale tt (to be reverted!)
    tt = tt / T

    # Compute basis B
    BB = np.zeros((dim_y, n_t))

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

    W = np.matmul(yy, BB)

    # Revert rescaling
    W = W * sqrt(T)

    return W