# CTG_wave
A Python object-oriented code for solving parametric wave equations using Continuous Time Galerkin (CTG) methods with a focus on uncertainty quantification.

## Table of Contents
- [Features](#features)
- [QuickStart](#quickstart)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features
- Continuous Time Galerkin (CTG) finite element methods for wave equations
- Space-time finite element discretization
- Parametric coefficient handling for uncertainty quantification
- Brownian motion coefficient implementation
- Efficient sparse matrix assembly for space-time operators
- Multiple examples and convergence studies
- Support for various boundary and initial conditions
- Post-processing and error analysis tools

## QuickStart
For those who have no time to lose.

```bash
mamba env create -f environment.yml
conda activate ctg-wave
pip install -e .
pre-commit install

# Run via console script with a YAML for new data, without YAML for example using data/data_swe.yaml
ctg run data/data_swe.yaml
```

## Detailed Installation

1. Clone the repository
```sh
git clone git@github.com:andreascaglioni/CTG_wave.git
```

2. Install the dependencies with Mamba
```sh
mamba env create -f environment.yml
conda activate ctg-wave
```

3. Install the package + hooks
```sh
pip install -e .
pre-commit install
```

4. Run the tiny deterministic demo (leapfrog reference)
```sh
python -m ctg.cli
```

5. Run Pytest tests (options in pyproject.toml)
```sh
pytest
```

3. Run the tests (from the project root directory):
```sh
python -m pytest tests/
```

## Usage
The program can be run from the command line with a yaml file containing the physics and numerics data:
```sh
ctg run [data/your_config.yaml]
```

If no YAML file is provided, the program will use `data/data_swe.yaml`.

Give a look to the `data/data_swe.yaml` file to understand how it is structured. Note that there is also a file `data/data_swe_funcs.py` containing the callables referenced in `data/data_swe.yaml` (e.g. initial and boundary data functions).

More examples are available in the `examples/` directory:

- `examples/SWE_CTG_example.py` - Basic parametric wave equation example
- `examples/WE_CTG_conv_dt.py` - Time step convergence study
- `examples/SWE_example_ensamble.py` - Ensemble simulation example

To generate different meshes: See example in `examples/generate_meshes_square.py`

Read the documentation for detailed API reference and mathematical background.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Andrea Scaglioni - [Get in touch on my website](https://andreascaglioni.net/contacts)
