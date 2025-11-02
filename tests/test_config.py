import numpy as np
import sys

sys.path.insert(0, "./")
from ctg.config import load_config, _resolve_callable


def test_load_data_from_yaml(mini_yaml_path):
    cfg = load_config(mini_yaml_path)

    x = np.linspace(0.0, 1.0, 5)
    t = np.zeros_like(x)
    X = np.column_stack([t, x])  # (N, 2) with columns [t, x]

    u0 = cfg.physics.initial_data_u(X)
    v0 = cfg.physics.initial_data_v(X)
    bc_u = cfg.physics.boundary_data_u(X)
    bc_v = cfg.physics.boundary_data_v(X)
    f0 = cfg.physics.rhs_0(X)
    f1 = cfg.physics.rhs_1(X)

    assert u0.shape == x.shape
    assert np.allclose(v0, 0.0)
    assert np.allclose(bc_u, 0.0)
    assert np.allclose(bc_v, 0.0)
    assert np.allclose(f0, 0.0)
    assert np.allclose(f1, 0.0)


def test_config_resolves_callables(mini_yaml_path):
    cfg = load_config(mini_yaml_path)
    x = np.linspace(0, 1, 11)
    t = np.zeros_like(x)
    X = np.column_stack([t, x])
    for f in (
        cfg.physics.initial_data_u,
        cfg.physics.initial_data_v,
        cfg.physics.boundary_data_u,
        cfg.physics.boundary_data_v,
        cfg.physics.rhs_0,
        cfg.physics.rhs_1,
    ):
        y = f(X)
        assert y.shape == x.shape
        assert np.isfinite(y).all()


def test_resolve_callable_string_and_dict():
    f = _resolve_callable("numpy:sin")
    assert callable(f) and abs(f(0.0)) < 1e-15
