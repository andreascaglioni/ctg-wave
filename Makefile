.PHONY: check-conda test example clean

# 1. Verify that a conda environment is active
check-conda:
	@python -c "import os; assert os.environ.get('CONDA_PREFIX'), 'Activate Conda environment first!'"

# 2. Run the full test suite (with coverage options from pyproject.toml)
test: check-conda
	pytest

# 3. Run a demo example. Usage: make example FILE=examples/wave1d_deterministic.yaml
example: check-conda
	ctg

# 4. Clean temporary and build artifacts
clean:
	rm -rf results __pycache__ */__pycache__ .pytest_cache
