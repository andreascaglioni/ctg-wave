"""Exact data for a wave equation solution:
\partial_{tt} u - \Delta u = f
u = 0 on \partial D
u = sin(pi*x) for t = 0

Everything written in first order formulation (2D vector field unknown).
"""

import numpy as np


def exact_sol(X):
    """
    Computes the exact solution for the wave equation at given space-time points.

    Args:
        X (np.ndarray): Array of shape (n, 2), where each row is a space-time point (t, x).

    Returns:
        np.ndarray: Array of shape (2, n). The first ROW is u (unknown), the second is v.
    """
    _t = X[:, 0]
    _x = X[:, 1]
    u = np.sin(np.pi * _x) * (1.0 + _t) * np.exp(-_t)
    v = np.sin(np.pi * _x) * (-_t) * np.exp(-_t)
    return np.vstack((u, v))


def exact_rhs(X):
    _t = X[:, 0]
    _x = X[:, 1]
    f1 = np.sin(np.pi * _x) * np.exp(-_t) * (1 + _t) * (-_t - 1 + np.pi**2)
    f0 = np.zeros_like(f1)
    return np.vstack((f0, f1))


def boundary_data(X):
    _t = X[:, 1]
    _x = X[:, 0]
    z = 0 * _x
    return np.vstack((z, z))


def initial_data(X):
    _t = X[:, 0]
    _x = X[:, 1]
    u0 = np.sin(np.pi * _x) + 0.0 * _t
    v0 = 0.0 * _t
    return np.vstack((u0, v0))
