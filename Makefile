.PHONY: check-conda test example clean

# 1. Verify that a conda environment is active
check-conda:
	@python -c "import os; assert os.environ.get('CONDA_PREFIX'), 'Activate Conda environment first!'"

# 2. Run the full test suite (with coverage options from pyproject.toml)
test: check-conda
	pytest

# 3. Run a demo example (YAML config is passed directly, no --config flag)
# Example usage:
#    make example FILE=examples/wave1d_deterministic.yaml
example: check-conda
	python -m ctg.cli $(FILE)

# 4. Clean temporary and build artifacts
clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .coverage dist build artifacts
