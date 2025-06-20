"""Exact data for a heat equation solution:
\partial_t u - \Delta u = f
differs from exact_solution_heat by the fact that this solution is INCREASING in t

This example is designed 
"""

import numpy as np


C = 1


def exact_sol(X):  # This is not always there. Used only to compute error
    _x = X[:, 1]
    _t = X[:, 0]
    return np.sin(np.pi * _x) * (1.0 + _t) * np.exp(C * _t)


def exact_rhs(X):
    _x = X[:, 1]
    _t = X[:, 0]
    return (
        np.sin(np.pi * _x)
        * np.exp(C * _t)
        * (1 + C + np.pi**2 + (C + np.pi**2) * _t)
    )


def boundary_data(X):
    #   X is (n, d+1) array, where each ROW is a space-time point (t, x), with x in R
    _x = X[:, 1]
    _t = X[:, 1]
    return 0 * _x + 0 * _t


def initial_data(X):
    _x = X[:, 1]
    _t = X[:, 0]
    return np.sin(np.pi * _x) + 0.0 * _t
