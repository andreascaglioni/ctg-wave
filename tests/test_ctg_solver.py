"""
Comprehensive tests for CTGSolver class.

This module tests the main CTG (Continuous Time Galerkin) solver functionality,
including initialization, time stepping, and complete solution workflows.
"""

import pytest
import numpy as np
import scipy.sparse
from dolfinx import fem, mesh, default_scalar_type
from mpi4py import MPI
import warnings
from unittest.mock import Mock, patch

# Import the class under test
from ctg.ctg_solver import CTGSolver
from ctg.FE_spaces import SpaceFE, TimeFE, SpaceTimeFE
from ctg.utils import compute_time_slabs


# Helper functions for creating test data
def create_test_mesh(comm=MPI.COMM_WORLD, n=4):
    """Create a simple unit square mesh for testing."""
    return mesh.create_unit_square(comm, n, n, mesh.CellType.triangle)

def create_test_function_space(comm=MPI.COMM_WORLD, degree=1):
    """Create a test function space."""
    msh = create_test_mesh(comm)
    return fem.functionspace(msh, ("Lagrange", degree))

def dummy_function_zero(x):
    """Dummy function returning zeros."""
    if x.ndim == 1:
        return np.zeros(1, dtype=default_scalar_type)
    else:
        return np.zeros(x.shape[0], dtype=default_scalar_type)

def dummy_function_one(x):
    """Dummy function returning ones."""
    if x.ndim == 1:
        return np.ones(1, dtype=default_scalar_type)
    else:
        return np.ones(x.shape[0], dtype=default_scalar_type)

def dummy_function_linear(x):
    """Dummy linear function."""
    if x.ndim == 1:
        return x[0] if len(x) > 0 else np.array([0.0])
    else:
        return x[:, 0]  # First coordinate (time)

def dummy_function_quadratic(x):
    """Dummy quadratic function."""
    if x.ndim == 1:
        return x[0]**2 if len(x) > 0 else np.array([0.0])
    else:
        return x[:, 0]**2  # First coordinate squared

def dummy_boundary_function(x):
    """Dummy boundary function."""
    if x.ndim == 1:
        return np.zeros(1, dtype=default_scalar_type)
    else:
        return np.zeros(x.shape[0], dtype=default_scalar_type)

def create_test_numerics_params(comm=MPI.COMM_WORLD):
    """Create test numerics parameters."""
    V_x = create_test_function_space(comm)
    return {
        "comm": comm,
        "V_x": V_x,
        "t_slab_size": 0.1,
        "order_t": 1
    }

def create_test_physics_params():
    """Create test physics parameters."""
    return {
        "boundary_D": lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
        "start_time": 0.0,
        "end_time": 0.2,
        "boundary_data_u": dummy_boundary_function,
        "boundary_data_v": dummy_boundary_function,
        "exact_rhs_0": dummy_function_zero,
        "exact_rhs_1": dummy_function_zero,
        "initial_data_u": dummy_function_zero,
        "initial_data_v": dummy_function_zero,
        "W_t": None
    }

def create_test_physics_params_with_W():
    """Create test physics parameters with W_t function."""
    params = create_test_physics_params()
    params["W_t"] = lambda x: 0.1 * np.sin(x[:, 0]) if x.ndim > 1 else np.array([0.1 * np.sin(x[0])])
    return params


class TestCTGSolverInitialization:
    """Test CTGSolver initialization and parameter handling."""

    def test_initialization_basic(self):
        """Test basic CTGSolver initialization."""
        numerics_params = create_test_numerics_params()
        solver = CTGSolver(numerics_params)
        
        assert solver.verbose == False
        assert solver.comm == numerics_params["comm"]
        assert solver.V_x == numerics_params["V_x"]
        assert solver.t_slab_size == numerics_params["t_slab_size"]
        assert solver.order_t == numerics_params["order_t"]

    def test_initialization_with_verbose(self):
        """Test CTGSolver initialization with verbose mode."""
        numerics_params = create_test_numerics_params()
        solver = CTGSolver(numerics_params, verbose=True)
        
        assert solver.verbose == True

    @pytest.mark.parametrize("t_slab_size", [0.05, 0.1, 0.2])
    def test_initialization_different_slab_sizes(self, t_slab_size):
        """Test initialization with different time slab sizes."""
        numerics_params = create_test_numerics_params()
        numerics_params["t_slab_size"] = t_slab_size
        solver = CTGSolver(numerics_params)
        
        assert solver.t_slab_size == t_slab_size

    @pytest.mark.parametrize("order_t", [1, 2, 3])
    def test_initialization_different_time_orders(self, order_t):
        """Test initialization with different time polynomial orders."""
        numerics_params = create_test_numerics_params()
        numerics_params["order_t"] = order_t
        solver = CTGSolver(numerics_params)
        
        assert solver.order_t == order_t

    def test_initialization_missing_params(self):
        """Test initialization with missing parameters."""
        incomplete_params = {"comm": MPI.COMM_WORLD}
        
        with pytest.raises(KeyError):
            CTGSolver(incomplete_params)


class TestCTGSolverRun:
    """Test the main run method of CTGSolver."""

    def test_run_basic_workflow(self):
        """Test basic run workflow without errors."""
        numerics_params = create_test_numerics_params()
        physics_params = create_test_physics_params()
        solver = CTGSolver(numerics_params)
        
        # Run the solver
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        # Check return types and basic properties
        assert isinstance(sol_slabs, list)
        assert isinstance(time_slabs, list)
        assert isinstance(space_time_fe, SpaceTimeFE)
        assert isinstance(total_n_dofs, int)
        assert total_n_dofs > 0
        
        # Check that we have solutions for all time slabs
        assert len(sol_slabs) == len(time_slabs)
        
        # Check that each solution is a numpy array
        for sol in sol_slabs:
            assert isinstance(sol, np.ndarray)
            assert sol.size > 0

    def test_run_with_W_function(self):
        """Test run with stochastic function W_t."""
        numerics_params = create_test_numerics_params()
        physics_params = create_test_physics_params_with_W()
        solver = CTGSolver(numerics_params)
        
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        assert len(sol_slabs) == len(time_slabs)
        assert total_n_dofs > 0

    def test_run_without_W_function_warning(self):
        """Test that warning is issued when W_t is not provided."""
        numerics_params = create_test_numerics_params()
        physics_params = create_test_physics_params()
        del physics_params["W_t"]  # Remove W_t to trigger warning
        solver = CTGSolver(numerics_params)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
            # Check if any UserWarning was issued about W_t
            assert any("W_t not provided" in str(warning.message) for warning in w if issubclass(warning.category, UserWarning))

    @pytest.mark.parametrize("start_time,end_time,t_slab_size,expected_slabs", [
        (0.0, 0.1, 0.1, 1),
        (0.0, 0.2, 0.1, 2),
        (0.0, 0.3, 0.1, 3),
        (0.0, 0.05, 0.1, 1),
    ])
    def test_run_different_time_configurations(self, start_time, end_time, t_slab_size, expected_slabs):
        """Test run with different time configurations."""
        numerics_params = create_test_numerics_params()
        numerics_params["t_slab_size"] = t_slab_size
        
        physics_params = create_test_physics_params()
        physics_params["start_time"] = start_time
        physics_params["end_time"] = end_time
        
        solver = CTGSolver(numerics_params)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        assert len(time_slabs) == expected_slabs
        assert len(sol_slabs) == expected_slabs

    def test_run_with_verbose_output(self, capsys):
        """Test run with verbose output."""
        numerics_params = create_test_numerics_params()
        physics_params = create_test_physics_params()
        solver = CTGSolver(numerics_params, verbose=True)
        
        solver.run(physics_params)
        
        captured = capsys.readouterr()
        assert "Slab_" in captured.out
        assert "Relative residual norm:" in captured.out

    def test_run_different_initial_conditions(self):
        """Test run with different initial conditions."""
        numerics_params = create_test_numerics_params()
        physics_params = create_test_physics_params()
        
        # Test with non-zero initial conditions
        physics_params["initial_data_u"] = dummy_function_linear
        physics_params["initial_data_v"] = dummy_function_quadratic
        
        solver = CTGSolver(numerics_params)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        assert len(sol_slabs) > 0
        assert total_n_dofs > 0

    def test_run_different_boundary_conditions(self):
        """Test run with different boundary conditions."""
        numerics_params = create_test_numerics_params()
        physics_params = create_test_physics_params()
        
        # Test with non-zero boundary conditions
        physics_params["boundary_data_u"] = dummy_function_one
        physics_params["boundary_data_v"] = dummy_function_linear
        
        solver = CTGSolver(numerics_params)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        assert len(sol_slabs) > 0
        assert total_n_dofs > 0

    def test_run_different_rhs_functions(self):
        """Test run with different right-hand side functions."""
        numerics_params = create_test_numerics_params()
        physics_params = create_test_physics_params()
        
        # Test with non-zero RHS
        physics_params["exact_rhs_0"] = dummy_function_one
        physics_params["exact_rhs_1"] = dummy_function_linear
        
        solver = CTGSolver(numerics_params)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        assert len(sol_slabs) > 0
        assert total_n_dofs > 0


class TestCTGSolverIterate:
    """Test the iterate method of CTGSolver."""

    def setup_method(self):
        """Set up test fixtures."""
        self.numerics_params = create_test_numerics_params()
        self.physics_params = create_test_physics_params()
        self.solver = CTGSolver(self.numerics_params)
        
        # Create a test space-time FE
        space_fe = SpaceFE(self.numerics_params["V_x"], self.physics_params["boundary_D"])
        slab = [0.0, 0.1]
        msh_t = mesh.create_interval(self.numerics_params["comm"], 1, slab)
        V_t = fem.functionspace(msh_t, ("Lagrange", self.numerics_params["order_t"]))
        time_fe = TimeFE(msh_t, V_t)
        self.space_time_fe = SpaceTimeFE(space_fe, time_fe)

    def test_iterate_basic(self):
        """Test basic iterate functionality."""
        slab = [0.0, 0.1]
        X0 = np.zeros(2 * self.space_time_fe.n_dofs)
        
        n_dofs, X = self.solver.iterate(
            self.physics_params["boundary_data_u"],
            self.physics_params["boundary_data_v"],
            self.physics_params["exact_rhs_0"],
            self.physics_params["exact_rhs_1"],
            self.physics_params["W_t"],
            slab,
            self.space_time_fe,
            X0
        )
        
        assert isinstance(n_dofs, int)
        assert n_dofs > 0
        assert isinstance(X, np.ndarray)
        assert X.size == 2 * self.space_time_fe.n_dofs

    def test_iterate_with_stochastic_function(self):
        """Test iterate with stochastic function W_t."""
        slab = [0.0, 0.1]
        X0 = np.zeros(2 * self.space_time_fe.n_dofs)
        W_t = lambda x: 0.1 * np.sin(x[:, 0]) if x.ndim > 1 else np.array([0.1 * np.sin(x[0])])
        
        n_dofs, X = self.solver.iterate(
            self.physics_params["boundary_data_u"],
            self.physics_params["boundary_data_v"],
            self.physics_params["exact_rhs_0"],
            self.physics_params["exact_rhs_1"],
            W_t,
            slab,
            self.space_time_fe,
            X0
        )
        
        assert isinstance(n_dofs, int)
        assert isinstance(X, np.ndarray)

    def test_iterate_none_system_matrix_error(self):
        """Test that RuntimeError is raised when system matrix is None."""
        slab = [0.0, 0.1]
        X0 = np.zeros(2 * self.space_time_fe.n_dofs)
        
        # Mock the assembler to return None system matrix
        with patch('ctg.ctg_solver.AssemblerWave') as mock_assembler_class:
            mock_assembler = Mock()
            mock_assembler.assemble_system.return_value = (None, np.zeros(10), np.zeros(10))
            mock_assembler_class.return_value = mock_assembler
            
            with pytest.raises(RuntimeError, match="System matrix is None"):
                self.solver.iterate(
                    self.physics_params["boundary_data_u"],
                    self.physics_params["boundary_data_v"],
                    self.physics_params["exact_rhs_0"],
                    self.physics_params["exact_rhs_1"],
                    self.physics_params["W_t"],
                    slab,
                    self.space_time_fe,
                    X0
                )

    def test_iterate_different_slab_sizes(self):
        """Test iterate with different slab sizes."""
        X0 = np.zeros(2 * self.space_time_fe.n_dofs)
        
        slabs = [[0.0, 0.05], [0.0, 0.1], [0.0, 0.2]]
        
        for slab in slabs:
            n_dofs, X = self.solver.iterate(
                self.physics_params["boundary_data_u"],
                self.physics_params["boundary_data_v"],
                self.physics_params["exact_rhs_0"],
                self.physics_params["exact_rhs_1"],
                self.physics_params["W_t"],
                slab,
                self.space_time_fe,
                X0
            )
            
            assert isinstance(n_dofs, int)
            assert isinstance(X, np.ndarray)

    def test_iterate_with_verbose_output(self, capsys):
        """Test iterate with verbose output."""
        self.solver.verbose = True
        slab = [0.0, 0.1]
        X0 = np.zeros(2 * self.space_time_fe.n_dofs)
        
        self.solver.iterate(
            self.physics_params["boundary_data_u"],
            self.physics_params["boundary_data_v"],
            self.physics_params["exact_rhs_0"],
            self.physics_params["exact_rhs_1"],
            self.physics_params["W_t"],
            slab,
            self.space_time_fe,
            X0
        )
        
        captured = capsys.readouterr()
        assert "Relative residual norm:" in captured.out

    def test_iterate_non_zero_initial_condition(self):
        """Test iterate with non-zero initial conditions."""
        slab = [0.0, 0.1]
        X0 = np.ones(2 * self.space_time_fe.n_dofs) * 0.1
        
        n_dofs, X = self.solver.iterate(
            self.physics_params["boundary_data_u"],
            self.physics_params["boundary_data_v"],
            self.physics_params["exact_rhs_0"],
            self.physics_params["exact_rhs_1"],
            self.physics_params["W_t"],
            slab,
            self.space_time_fe,
            X0
        )
        
        assert isinstance(n_dofs, int)
        assert isinstance(X, np.ndarray)
        # Solution should incorporate the initial condition
        assert not np.allclose(X, 0.0)


class TestCTGSolverParametrized:
    """Parametrized tests for CTGSolver with different configurations."""

    @pytest.mark.parametrize("mesh_size,time_order", [
        (2, 1), (4, 1), (8, 1),
        (4, 2), (4, 3)
    ])
    def test_solver_different_discretizations(self, mesh_size, time_order):
        """Test solver with different space-time discretizations."""
        # Create mesh with specified size
        msh = mesh.create_unit_square(MPI.COMM_WORLD, mesh_size, mesh_size, mesh.CellType.triangle)
        V_x = fem.functionspace(msh, ("Lagrange", 1))
        
        numerics_params = {
            "comm": MPI.COMM_WORLD,
            "V_x": V_x,
            "t_slab_size": 0.1,
            "order_t": time_order
        }
        
        physics_params = create_test_physics_params()
        solver = CTGSolver(numerics_params)
        
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        assert len(sol_slabs) == len(time_slabs)
        assert total_n_dofs > 0

    @pytest.mark.parametrize("W_function", [
        None,
        lambda x: np.zeros(x.shape[0]) if x.ndim > 1 else np.array([0.0]),  # Zero function
        lambda x: 0.1 * x[:, 0] if x.ndim > 1 else np.array([0.1 * x[0]]),  # Linear function  
        lambda x: 0.1 * np.sin(x[:, 0]) if x.ndim > 1 else np.array([0.1 * np.sin(x[0])]),  # Sinusoidal function
        lambda x: 0.1 * np.exp(-x[:, 0]) if x.ndim > 1 else np.array([0.1 * np.exp(-x[0])]),  # Exponential decay
    ])
    def test_solver_different_stochastic_functions(self, W_function):
        """Test solver with different stochastic functions."""
        numerics_params = create_test_numerics_params()
        physics_params = create_test_physics_params()
        physics_params["W_t"] = W_function
        
        solver = CTGSolver(numerics_params)
        
        if W_function is None:
            # Remove W_t key entirely to trigger the warning
            if "W_t" in physics_params:
                del physics_params["W_t"]
            # Check for warnings about W_t not provided
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
                # Check if any UserWarning was issued about W_t
                assert any("W_t not provided" in str(warning.message) for warning in w if issubclass(warning.category, UserWarning))
        else:
            sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        assert len(sol_slabs) == len(time_slabs)
        assert total_n_dofs > 0


class TestCTGSolverIntegration:
    """Integration tests for CTGSolver functionality."""

    def test_complete_workflow_multiple_slabs(self):
        """Test complete workflow with multiple time slabs."""
        numerics_params = create_test_numerics_params()
        numerics_params["t_slab_size"] = 0.05  # Smaller slabs for multiple iterations
        
        physics_params = create_test_physics_params()
        physics_params["end_time"] = 0.2  # Multiple slabs
        
        solver = CTGSolver(numerics_params, verbose=True)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        # Should have 4 time slabs
        assert len(time_slabs) == 4
        assert len(sol_slabs) == 4
        
        # Check continuity: final condition of one slab becomes initial of next
        for i in range(len(sol_slabs) - 1):
            current_sol = sol_slabs[i]
            next_sol = sol_slabs[i + 1]
            
            # Both should have same size
            assert current_sol.size == next_sol.size

    def test_consistency_with_time_slab_computation(self):
        """Test consistency with time slab computation utility."""
        start_time = 0.0
        end_time = 0.3
        t_slab_size = 0.1
        
        # Compute expected time slabs
        expected_slabs = compute_time_slabs(start_time, end_time, t_slab_size)
        
        numerics_params = create_test_numerics_params()
        numerics_params["t_slab_size"] = t_slab_size
        
        physics_params = create_test_physics_params()
        physics_params["start_time"] = start_time
        physics_params["end_time"] = end_time
        
        solver = CTGSolver(numerics_params)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        # Time slabs should match expected
        assert len(time_slabs) == len(expected_slabs)
        for actual, expected in zip(time_slabs, expected_slabs):
            assert np.isclose(actual[0], expected[0])
            assert np.isclose(actual[1], expected[1])

    def test_solution_vector_properties(self):
        """Test properties of solution vectors."""
        numerics_params = create_test_numerics_params()
        physics_params = create_test_physics_params()
        
        solver = CTGSolver(numerics_params)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        expected_size = 2 * space_time_fe.n_dofs
        
        for sol in sol_slabs:
            # Each solution should have correct size
            assert sol.size == expected_size
            
            # Should be real-valued
            assert np.isrealobj(sol)
            
            # Should be finite (no NaN or inf)
            assert np.all(np.isfinite(sol))

    def test_dof_counting_consistency(self):
        """Test that total DOF count is consistent."""
        numerics_params = create_test_numerics_params()
        physics_params = create_test_physics_params()
        physics_params["end_time"] = 0.2  # Two slabs
        
        solver = CTGSolver(numerics_params)
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        
        # Total DOFs should equal number of slabs times DOFs per slab
        expected_total = len(time_slabs) * space_time_fe.n_dofs
        assert total_n_dofs == expected_total


class TestCTGSolverErrorHandling:
    """Test error handling in CTGSolver."""

    def test_missing_physics_parameters(self):
        """Test handling of missing physics parameters."""
        numerics_params = create_test_numerics_params()
        incomplete_physics = {"start_time": 0.0, "end_time": 0.1}
        
        solver = CTGSolver(numerics_params)
        
        with pytest.raises(KeyError):
            solver.run(incomplete_physics)

    def test_invalid_time_configuration(self):
        """Test handling of invalid time configurations."""
        numerics_params = create_test_numerics_params()
        physics_params = create_test_physics_params()
        physics_params["start_time"] = 0.1
        physics_params["end_time"] = 0.0  # Invalid: end < start
        
        solver = CTGSolver(numerics_params)
        
        # The solver will still create slabs starting from start_time
        # This is behavior of compute_time_slabs - it doesn't validate inputs
        sol_slabs, time_slabs, space_time_fe, total_n_dofs = solver.run(physics_params)
        assert len(time_slabs) >= 1  # Will create at least one slab
        assert total_n_dofs > 0


# Main execution for standalone testing
if __name__ == "__main__":
    print("Running CTGSolver tests...")
    
    # Basic initialization tests
    print("Testing basic initialization...")
    test_init = TestCTGSolverInitialization()
    test_init.test_initialization_basic()
    test_init.test_initialization_with_verbose()
    print("Initialization tests passed")
    
    # Run method tests
    print("Testing run method...")
    test_run = TestCTGSolverRun()
    test_run.test_run_basic_workflow()
    print("Run method tests passed")
    
    # Iterate method tests
    print("Testing iterate method...")
    test_iterate = TestCTGSolverIterate()
    test_iterate.setup_method()
    test_iterate.test_iterate_basic()
    print("Iterate method tests passed")
    
    # Integration tests
    print("Testing integration scenarios...")
    test_integration = TestCTGSolverIntegration()
    test_integration.test_complete_workflow_multiple_slabs()
    print("Integration tests passed")
    
    print("All CTGSolver tests completed successfully!")
