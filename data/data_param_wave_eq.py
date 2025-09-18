"""Exact data for a 1st order parametric wave equation:

    \partial_t u &= v +  W u \\
    \partial_t v &= \Delta u +  W v + W^2 u + f.

We assume f=0, boundary and initial conditions:
    u = 0 on \partial D
    u = sin(2*pi*x) for t = 0
    \partial_t u = 0

The exact solution is not known.

NB: Initial data and boundary data must always be defined as global functions on the space-time cylinder. They must also be 0 outise of respecitvely initial time {t=0} and boundary \partial D
"""

import numpy as np


def exact_rhs(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.zeros((2, _t.size))

def exact_rhs_0(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0 * _x

def exact_rhs_1(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0 * _x

def boundary_data_u(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0 * _x

def boundary_data_v(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0 * _x

def initial_data_u(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.where(_t == 0, np.sin(2*np.pi*_x), 0.)

def initial_data_v(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0. *_t