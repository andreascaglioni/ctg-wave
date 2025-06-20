"""Exact data for a heat equation solution:
\partial_t u - \Delta u = f
"""
import numpy as np


def exact_sol(X):  # This is not always there. Used only to compute error
        _x = X[:, 1]
        _t = X[:, 0]
        return np.sin(np.pi * _x) * (1.0 + _t) * np.exp(-0.5 * _t)

def exact_rhs(X):
    _x = X[:, 1]
    _t = X[:, 0]
    return (
        np.sin(np.pi * _x)
        * np.exp(-0.5 * _t)
        * (0.5 + np.pi**2 + (np.pi**2 - 0.5) * _t)
        # * (-3./4. + np.pi**2 + (np.pi**2 + 1./4.) * _t)
    )

def boundary_data(X):
    #   X is (n, d+1) array, where each ROW is a space-time point (t, x), with x in R
      _x = X[:, 1]
      _t = X[:, 1]
      return 0*_x+0*_t

def initial_data(X):
    _x = X[:, 1]
    _t = X[:, 0]    
    return np.sin(np.pi * _x) + 0. *_t