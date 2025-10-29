"""Functions needed to define a wave equation. To be imported through pydantic+yaml file"""

import numpy as np


def rhs_0(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0 * _x


def rhs_1(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0 * _x


def boundary_u(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0 * _x


def boundary_v(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0 * _x


def initial_u(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.where(_t == 0, np.sin(2 * np.pi * _x), 0.0)


def initial_v(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0.0 * _t
