"""Test the computation of error L2 or LInf in time and H1 in space. Interpolate exact solution and check that the finite element projection error is small."""

from dolfinx import fem, mesh
import pytest

from ctg.error import compute_err
from ctg.config import AppConfig, load_config
from ctg.utils import compute_time_slabs
from ctg.FE_spaces import SpaceFE, TimeFE, SpaceTimeFE


@pytest.fixture()
def space_fe(mini_yaml_deterministic_path):
    cfg: AppConfig = load_config(mini_yaml_deterministic_path)
    cfg_p = cfg.physics
    cfg_n = cfg.numerics
    comm = cfg_n.comm
    msh_x = mesh.create_unit_interval(comm, cfg_n.n_cells_space)
    V_x = fem.functionspace(msh_x, ("Lagrange", cfg_n.order_x, (1,)))
    space_fe = SpaceFE(V_x, cfg_p.boundary_D)
    return space_fe


@pytest.fixture()
def exact_u_projection(mini_yaml_deterministic_path, space_fe):
    cfg: AppConfig = load_config(mini_yaml_deterministic_path)
    cfg_p = cfg.physics
    cfg_n = cfg.numerics

    comm = cfg_n.comm

    time_slabs = compute_time_slabs(cfg_p.start_time, cfg_p.end_time, cfg_n.t_slab_size)
    msh_t = mesh.create_interval(comm, 1, [time_slabs[0][0], time_slabs[0][1]])
    V_t = fem.functionspace(msh_t, ("Lagrange", cfg_n.order_t))
    time_fe = TimeFE(V_t)

    space_time_fe = SpaceTimeFE(space_fe, time_fe)

    u_exa = cfg_p.exact_sol_u

    sol_u = []
    for slab in time_slabs:
        msh_t = mesh.create_interval(comm, 1, [slab[0], slab[1]])
        V_t = fem.functionspace(msh_t, ("Lagrange", cfg_n.order_t))
        time_fe = TimeFE(V_t)
        space_time_fe.update_time_fe(time_fe)
        sol_u.append(u_exa(space_time_fe.dofs))

    return sol_u


def test_error_l2(mini_yaml_deterministic_path, exact_u_projection, space_fe):

    cfg: AppConfig = load_config(mini_yaml_deterministic_path)
    cfg_p = cfg.physics
    cfg_n = cfg.numerics

    time_slabs = compute_time_slabs(cfg_p.start_time, cfg_p.end_time, cfg_n.t_slab_size)

    err_u_l2, _, _, _ = compute_err(
        cfg_n.comm,
        cfg_n.order_t,
        "h1",
        "l2",
        time_slabs,
        space_fe,
        exact_u_projection,
        cfg_p.exact_sol_u,
    )

    assert err_u_l2 < 3.0e-1


def test_error_linf(mini_yaml_deterministic_path, exact_u_projection, space_fe):
    cfg: AppConfig = load_config(mini_yaml_deterministic_path)
    cfg_p = cfg.physics
    cfg_n = cfg.numerics

    time_slabs = compute_time_slabs(cfg_p.start_time, cfg_p.end_time, cfg_n.t_slab_size)

    err_u_linf, _, _, _ = compute_err(
        cfg_n.comm,
        cfg_n.order_t,
        "h1",
        "linf",
        time_slabs,
        space_fe,
        exact_u_projection,
        cfg_p.exact_sol_u,
    )

    assert err_u_linf < 1.4
