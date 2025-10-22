# CTG_wave
A Python object-oriented code for solving parametric wave equations using Continuous Time Galerkin (CTG) methods with a focus on uncertainty quantification.

## Table of Contents
- [Features](#features)
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

## Installation
1. Clone the repository
```sh
git clone git@github.com:andreascaglioni/CTG_wave.git
```

2. Install the dependencies (requires DOLFINx and FEniCS)
```sh
pip install -r requirements.txt
```

3. Run the tests (from the project root directory):
```sh
python -m pytest tests/
```

## Usage
See the examples in the `examples/` directory:

- `PWE_CTG_example.py` - Basic parametric wave equation example
- `wave_CTG_conv_dt.py` - Time step convergence study
- `PWE_example_ensamble.py` - Ensemble simulation example

For mesh generation:
```sh
python examples/generate_meshes_square.py
```

Read the documentation for detailed API reference and mathematical background.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
Andrea Scaglioni - [Get in touch on my website](https://andreascaglioni.net/contacts)