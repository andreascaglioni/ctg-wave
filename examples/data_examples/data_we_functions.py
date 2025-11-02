"""Functions needed to define a wave equation. To be imported through pydantic+yaml file"""

import numpy as np
from mpi4py import MPI


def exact_sol_u(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.sin(2 * np.pi * _x) * np.cos(2 * np.pi * _t)


def exact_sol_v(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.sin(2 * np.pi * _x) * (-2 * np.pi) * np.sin(2 * np.pi * _t)


def rhs_0(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0 * _x


def rhs_1(X):
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
    return np.where(_t == 0, np.sin(2 * np.pi * _x), 0.0)


def initial_data_v(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return 0.0 * _t


def boundary_D(x):
    return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))


comm = MPI.COMM_SELF
