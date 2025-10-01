import numpy as np


def exact_sol_u(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.sin(2*np.pi*_x) * np.cos(2*np.pi*_t)

def exact_sol_v(X):
    _t = X[:, 0]
    _x = X[:, 1]
    return np.sin(2*np.pi*_x) * (-2*np.pi) * np.sin(2*np.pi*_t)

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