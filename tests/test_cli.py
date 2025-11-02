import pytest
from typer.testing import CliRunner
import numpy as np
from ctg.cli import run
from tests import have_dolfinx

runner = CliRunner()


@pytest.mark.skipif(not have_dolfinx(), reason="dolfinx not available in this env")
def test_cli_smoke_run(mini_yaml_path):
    run(config_path=mini_yaml_path)


@pytest.mark.skipif(not have_dolfinx(), reason="dolfinx not available")
def test_cli_output(mini_yaml_path):
    results_dir = run(config_path=mini_yaml_path)
    assert results_dir.exists()
    solution_file = results_dir / "solution_data.npz"
    if solution_file.exists():
        data = np.load(solution_file)
        assert "sol_slabs" in data or len(data.files) > 0
