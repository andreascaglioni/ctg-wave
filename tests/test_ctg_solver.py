import numpy as np
import pytest

# Import the module under test
from ctg.brownian_motion import param_LC_W
from ctg.ctg_solver import CTGSolver
from pathlib import Path
from ctg.config import load_config
from tests import have_dolfinx


@pytest.mark.skipif(not have_dolfinx(), reason="dolfinx not available")
def test_ctg_repeat_W():
    data_file = Path("tests/data_tests/data_swe_test_mini.yaml")
    cfg = load_config(data_file)

    solver = CTGSolver(cfg.numerics)

    np.random.seed(cfg.numerics.seed)
    y = np.random.standard_normal(100)

    def W_t(tt):
        return param_LC_W(y, tt, T=cfg.physics.end_time)[0]

    # first run
    sol_1, time_slabs1, stfe1, total_ndofs1 = solver.run(cfg.physics, W_t)

    # second run with the same inputs
    sol_2, time_slabs2, stfe2, total_ndofs2 = solver.run(cfg.physics, W_t)

    # basic sanity checks
    assert len(sol_1) == len(sol_2)
    assert total_ndofs1 == total_ndofs2
    assert stfe1.n_dofs == stfe2.n_dofs

    # Compare solutions in L^2(0, T, H^1(D)) for all iterations
    n_scalar = stfe1.n_dofs
    A = stfe1.matrix["L"]
    for i in range(len(sol_1)):
        d = sol_1[i][:n_scalar] - sol_2[i][:n_scalar]
        assert (A @ d) @ d < 1.0e-1


@pytest.mark.skipif(not have_dolfinx(), reason="dolfinx not available")
def test_CTG_similar_W():
    data_file = Path("tests/data_tests/data_swe_test_mini.yaml")
    cfg = load_config(data_file)

    solver = CTGSolver(cfg.numerics)

    np.random.seed(cfg.numerics.seed)
    y = np.random.standard_normal(100)

    def W_t(tt):
        return param_LC_W(y, tt, T=cfg.physics.end_time)[0]

    # first run
    sol_1, time_slabs1, stfe1, total_ndofs1 = solver.run(cfg.physics, W_t)

    # Change W2 by 1%
    y2 = y + 0.01 * np.random.standard_normal(100)

    def W2_t(tt):
        return param_LC_W(y2, tt, T=cfg.physics.end_time)[0]

    # second run with the same inputs
    sol_2, time_slabs2, stfe2, total_ndofs2 = solver.run(cfg.physics, W2_t)

    # basic sanity checks
    assert len(sol_1) == len(sol_2)
    assert total_ndofs1 == total_ndofs2
    assert stfe1.n_dofs == stfe2.n_dofs

    # Error in L^2(0, T, H^1(D)) for all iterations
    n_scalar = stfe1.n_dofs
    A = stfe1.matrix["L"]
    for i in range(len(sol_1)):
        d = sol_1[i][:n_scalar] - sol_2[i][:n_scalar]
        assert (A @ d) @ d < 1.0e-1
