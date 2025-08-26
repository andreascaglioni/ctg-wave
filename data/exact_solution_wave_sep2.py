"""Exact data for a wave equation:
\partial_{tt} u - \Delta u = f
u = 0 on \partial D
u = sin(2*pi*x) for t = 0
\partial_t u = 0

With exact solution

u = sin(2*pi*x) * cos(2*pi*t),
\partial_t u = sin(2*pi*x) * (-sin(2*pi*t)*2*pi),

We write separately the u and v components in the 1st roder formulation of the wave equation.

Note: below X is always understood to contain coordinates in whoe space-time domain, even for initial and boundary data.
NB: initial data and boundary data are always 0 outise of respecitvely initial time {t=0} and boundary \partial D
"""

import numpy as np


def exact_sol_u(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.sin(2*np.pi*_x) * np.cos(2*np.pi*_t)

def exact_sol_v(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.sin(2*np.pi*_x) * (-np.sin(2*np.pi*_t)*2*np.pi)

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