"""Exact data for a wave equation solution:
\partial_{tt} u - \Delta u = f
u = 0 on \partial D
u = sin(pi*x) for t = 0

We write separately the u and v components in the 1st roder formulation of the wave equation.

Note: below X is always understood to contain coordinates in whoe space-time domain, even for initial and boundary data.
NB: initial data and boundary data are always 0 outise of respecitvely initial time {t=0} and boundary \partial D
"""

import numpy as np


def exact_sol_u(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.sin(np.pi * _x) * (1.0 + _t) * np.exp(-_t)

def exact_sol_v(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.sin(np.pi * _x) * (-_t) * np.exp(-_t)

def exact_rhs(X):
    _t = X[:, 0]
    _x = X[:, 1]
    f1 = np.sin(np.pi * _x) * np.exp(-_t) * (1 + _t) * (-_t - 1 + np.pi**2)
    f0 = np.zeros_like(f1)
    return np.vstack((f0, f1))

def exact_rhs_0(X):
    _t = X[:, 0]
    _x = X[:, 1]
    f1 = np.sin(np.pi * _x) * np.exp(-_t) * (1 + _t) * (-_t - 1 + np.pi**2)
    return np.zeros_like(_x)

def exact_rhs_1(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.sin(np.pi * _x) * np.exp(-_t) * (1 + _t) * (-_t - 1 + np.pi**2)

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
    # return np.sin(np.pi * _x) * np.asarray(np.abs(_t)<1.e-10).nonzero()
    return np.where(_t == 0, np.sin(np.pi * _x), 0.)

def initial_data_v(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0. *_t