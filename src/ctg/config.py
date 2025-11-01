"""Configuration system for CTG wave equation solver.

This module provides Pydantic-based configuration classes that support:
- Loading from YAML files
- Automatic validation of types and values
- Resolution of string paths to callable functions
- Handling of arbitrary types (MPI communicators, callables)

Example:
    Load configuration from YAML file::

        from pathlib import Path
        from ctg.config import load_config

        config = load_config(Path("config.yaml"))
        print(config.numerics.n_cells_space)  # 100

    Create configuration with string references to functions::

        config = AppConfig(
            physics={
                "initial_data_u": "data.data_functions_pwe:initial_u",
                "rhs_0": "data.data_functions_pwe:rhs_0"
            }
        )
"""

from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, field_validator
import yaml
from typing import Callable, Any, Union, Optional
from importlib import import_module
from functools import partial
import numpy as np
from mpi4py import MPI


def _resolve_callable(v: Union[str, dict, Callable]) -> Callable:
    """Resolve a callable from various input formats.

    Supports multiple input formats:
    - Direct callable: returned as-is
    - String path: "module:function" or "module.submodule.function"
    - Dict with path and params: {"path": "module:function", "params": {...}}

    Args:
        v: Input value that can be:
            - A callable object (function, lambda, etc.)
            - A string in format "module:function" or "module.function"
            - A dict with keys "path" (required) and "params" (optional)

    Returns:
        Callable: The resolved callable function. If params were provided in dict
            format, returns a functools.partial with those params applied.

    Raises:
        ValueError: If dict input doesn't include 'path' key.
        TypeError: If input is not callable, string, or dict.
        TypeError: If resolved path doesn't point to a callable object.
        ModuleNotFoundError: If the module cannot be imported.
        AttributeError: If the function name doesn't exist in the module.

    Example:
        >>> # String format
        >>> func = _resolve_callable("numpy:sin")
        >>> func(0)  # Returns 0.0

        >>> # Dict with parameters
        >>> func = _resolve_callable({
        ...     "path": "numpy:power",
        ...     "params": {"exponent": 2}
        ... })
        >>> func(3)  # Returns 9 (3^2)
    """
    # allow: callable | "mod:func"/"mod.func" | {"path":..., "params": {...}}
    if callable(v):
        return v
    if isinstance(v, str):
        path = v
        params = {}
    elif isinstance(v, dict):
        path_or_none: Optional[str] = v.get("path") or v.get("target") or v.get("func")
        if not path_or_none:
            raise ValueError("dict spec must include 'path'")
        params = v.get("params", {})
    else:
        raise TypeError("Expected callable | str | dict with 'path'")

    # split "module:function" or "module.sub.func"
    if ":" in path:
        mod, name = path.split(":", 1)
    else:
        *mods, name = path.split(".")
        mod = ".".join(mods)
    obj = getattr(import_module(mod), name)
    if not callable(obj):
        raise TypeError(f"{path} is not callable")
    return partial(obj, **params) if params else obj


class physicsCfg(BaseModel):
    """Physics configuration for wave equation problem.

    Stores callable functions defining the physics of the wave equation:
    initial conditions, boundary conditions, and right-hand side forcing terms.
    All functions should accept X with shape (n_points, 2) where columns are [t, x].

    Attributes:

        exact_sol_u: Exact solution for displacement.
        exact_sol_v: Exact solution for verlocity.
        initial_data_u: Initial condition for displacement u(x, t=0).
            Default returns zeros.
        initial_data_v: Initial condition for velocity v(x, t=0).
            Default returns zeros.
        boundary_data_u: Boundary condition for displacement u on domain boundary.
            Default returns zeros.
        boundary_data_v: Boundary condition for velocity v on domain boundary.
            Default returns zeros.
        boundary_D: Dirichlet boundary condition specification.
            Default returns zeros.
        rhs_0: Right-hand side forcing term for first equation.
            Default returns zeros.
        rhs_1: Right-hand side forcing term for second equation.
            Default returns zeros.
        start_time: Start time for the physics simulation.
            Default: 0.0.
        end_time: End time for the physics simulation.
            Default: 1.0.

    Note:
        Functions can be specified as:
        - Direct callables: lambda X: np.sin(X[:, 1])
        - String paths: "module.submodule:function_name"
        - Dicts with params: {"path": "module:func", "params": {"a": 1}}
        - exact_sol_u and exact_sol_v are optional

    Example:
        >>> physics = physicsCfg(
        ...     initial_data_u="data.data_functions_pwe:initial_u",
        ...     rhs_0=lambda X: np.zeros(X.shape[0]),
        ...     start_time=0.0,
        ...     end_time=2.0
        ... )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    exact_sol_u: Callable[..., Any] = Field(
        default=lambda X: np.zeros(X.shape[0]),
        description="Exact solution for displacement u",
    )
    exact_sol_v: Callable[..., Any] = Field(
        default=lambda X: np.zeros(X.shape[0]),
        description="Exact solution for velocity v",
    )

    initial_data_u: Callable[..., Any] = Field(
        default=lambda X: np.zeros(X.shape[0]),
        description="Initial condition for displacement u(x, t=0)",
    )
    initial_data_v: Callable[..., Any] = Field(
        default=lambda X: np.zeros(X.shape[0]),
        description="Initial condition for velocity v(x, t=0)",
    )
    boundary_data_u: Callable[..., Any] = Field(
        default=lambda X: np.zeros(X.shape[0]), description="Boundary condition for displacement u"
    )
    boundary_data_v: Callable[..., Any] = Field(
        default=lambda X: np.zeros(X.shape[0]), description="Boundary condition for velocity v"
    )
    boundary_D: Callable[..., Any] = Field(
        default=lambda X: np.zeros(X.shape[0]), description="Dirichlet boundary condition"
    )
    rhs_0: Callable[..., Any] = Field(
        default=lambda X: np.zeros(X.shape[0]),
        description="Right-hand side forcing term for first equation",
    )
    rhs_1: Callable[..., Any] = Field(
        default=lambda X: np.zeros(X.shape[0]),
        description="Right-hand side forcing term for second equation",
    )
    start_time: float = Field(default=0.0, description="Start time for the physics simulation")
    end_time: float = Field(default=1.0, gt=0.0, description="End time for the physics simulation")

    @field_validator(
        "initial_data_u",
        "initial_data_v",
        "boundary_data_u",
        "boundary_data_v",
        "boundary_D",
        "rhs_0",
        "rhs_1",
        "exact_sol_u",
        "exact_sol_v",
        mode="before",
    )
    @classmethod
    def resolve_callables(cls, v):
        """Resolve string/dict references to actual callable functions.

        Args:
            v: Input value (callable, string, or dict).

        Returns:
            Callable: The resolved callable function.
        """
        return _resolve_callable(v)


class numericsCfg(BaseModel):
    """Numerical discretization and solver configuration.

    Defines spatial and temporal discretization parameters, random seed,
    and MPI communicator for parallel execution.

    Attributes:
        seed: Random seed for reproducibility. Default: 0.
        comm: MPI communicator for parallel execution. Default: MPI.COMM_SELF.
            Can be specified as a string path like "module:attribute".
        n_cells_space: Number of spatial mesh cells. Default: 100.
        order_x: Polynomial degree for spatial finite elements. Default: 1.
        t_slab_size: Time slab size for space-time discretization. Default: 0.01.
        order_t: Polynomial degree for temporal finite elements. Default: 1.
        verbose: Print output from CTGSolver during run. Default: False.

    Example:
        >>> numerics = numericsCfg(
        ...     n_cells_space=200,
        ...     t_slab_size=0.05,
        ...     comm="data.data_pwe_functions:comm"
        ... )
    """

    # Allow non-standard types like MPI.Comm
    model_config = ConfigDict(arbitrary_types_allowed=True)

    seed: int = Field(default=0, description="Random seed for reproducibility")
    comm: MPI.Comm = Field(default=MPI.COMM_SELF, description="MPI communicator")
    n_cells_space: int = Field(default=100, ge=1, description="Number of spatial cells")
    order_x: int = Field(default=1, ge=1, description="Spatial FE polynomial degree")
    t_slab_size: float = Field(default=0.01, gt=0.0, description="Time slab size")
    order_t: int = Field(default=1, ge=1, description="Temporal FE polynomial degree")
    verbose: bool = Field(default=False, description="Print output CTGSolver during run")

    @field_validator("comm", mode="before")
    @classmethod
    def resolve_comm(cls, v):
        """Resolve MPI communicator from string path or return as-is.

        Args:
            v: Input value (MPI.Comm object or string path).

        Returns:
            MPI.Comm: The resolved MPI communicator.
        """
        if isinstance(v, str):
            # split "module:attribute"
            if ":" in v:
                mod, name = v.split(":", 1)
            else:
                *mods, name = v.split(".")
                mod = ".".join(mods)
            return getattr(import_module(mod), name)
        return v


class postCfg(BaseModel):
    """Post-processing configuration."""

    dir_save: str = Field(..., description="Directory where outputs will be saved")


class AppConfig(BaseModel):
    """Main application configuration container.

    Aggregates all configuration sections for the CTG wave equation solver:
    physics setup and numerical parameters.

    Attributes:
        physics: Physics configuration (initial/boundary conditions, forcing).
        numerics: Numerical discretization configuration.

    Example:
        >>> # Create with defaults
        >>> config = AppConfig()

        >>> # Load from YAML
        >>> config = load_config(Path("config.yaml"))

        >>> # Create programmatically
        >>> config = AppConfig(
        ...     physics={"initial_data_u": "data.funcs:my_initial_u"},
        ...     numerics={"n_cells_space": 200, "end_time": 2.0}
        ... )
    """

    physics: physicsCfg = Field(default_factory=physicsCfg, description="Physics configuration")
    numerics: numericsCfg = Field(
        default_factory=numericsCfg, description="Numerical discretization configuration"
    )
    post: postCfg = Field(default_factory=postCfg, description="Post-processing configuration")


def load_config(source: Union[Path, str]) -> AppConfig:
    """Load configuration from a YAML file path or YAML string content.

    Accepts either a Path to a YAML file or a string containing YAML content.
    Creates a validated AppConfig instance and automatically resolves string
    references to callable functions.

    Args:
        source: Either a Path object pointing to a YAML configuration file,
            or a string containing YAML content.

    Returns:
        AppConfig: Validated configuration object with all settings loaded
            and callables resolved.

    Raises:
        FileNotFoundError: If source is a Path and the file doesn't exist.
        yaml.YAMLError: If the YAML content is malformed.
        pydantic.ValidationError: If the configuration doesn't match the schema.

    Example:
        >>> from pathlib import Path
        >>> # Load from file path
        >>> config = load_config(Path("configs/wave_eq.yaml"))
        >>> print(config.numerics.n_cells_space)
        100

        >>> # Load from YAML string
        >>> yaml_str = '''
        ... physics:
        ...   initial_data_u: "data.data_functions_pwe:initial_u"
        ... numerics:
        ...   n_cells_space: 100
        ... '''
        >>> config = load_config(yaml_str)
    """
    # Determine if source is a Path or string content
    if isinstance(source, Path):
        yaml_text = source.read_text(encoding="utf-8")
    else:
        yaml_text = source

    data = yaml.safe_load(yaml_text)
    return AppConfig(**data)
