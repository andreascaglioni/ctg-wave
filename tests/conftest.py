from pathlib import Path
import pytest


@pytest.fixture()
def mini_yaml_path():
    return Path("tests/data_tests/data_swe_test_mini.yaml")
