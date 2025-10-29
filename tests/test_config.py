from pathlib import Path
import numpy as np
from ctg.config import load_config
import sys

sys.path.insert(0, ".")


def test_data_from_yaml():
    cfg = load_config(Path("tests/artifacts/data_swe.yaml"))
    x = np.linspace(0.0, 1.0, 5)
    t = np.zeros_like(x)
    X = np.column_stack([t, x])  # (N, 2) with columns [t, x]

    u0 = cfg.physics.initial_data_u(X)
    v0 = cfg.physics.initial_data_v(X)
    b_u = cfg.physics.boundary_data_u(X)
    b_v = cfg.physics.boundary_data_v(X)
    f0 = cfg.physics.rhs_0(X)
    f1 = cfg.physics.rhs_1(X)

    assert u0.shape == x.shape
    assert np.allclose(v0, 0.0)
    assert np.allclose(b_u, 0.0)
    assert np.allclose(b_v, 0.0)
    assert np.allclose(f0, 0.0)
    assert np.allclose(f1, 0.0)
