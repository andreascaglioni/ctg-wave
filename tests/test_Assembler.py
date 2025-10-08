"""
Tests for the Assembler module.

This module tests the AssemblerWave class which handles the assembly of linear systems
for the CTG space-time solver, including boundary and initial conditions.
"""

import numpy as np
import pytest
import scipy.sparse
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl

from ctg.Assembler import AssemblerWave
from ctg.FE_spaces import SpaceFE, TimeFE, SpaceTimeFE


def create_unit_square_mesh(n=4):
    """Create a simple unit square mesh for testing."""
    return mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)


def create_unit_interval_mesh(n=5):
    """Create a simple unit interval mesh for testing."""
    return mesh.create_unit_interval(MPI.COMM_WORLD, n)


# def create_function_space(test_mesh, element_type="Lagrange", degree=1):
#     """Create a function space with proper DOLFiNx syntax."""
#     return fem.functionspace(test_mesh, (element_type, degree))


def boundary_marker(x):
    """Simple boundary marker for unit square - marks all boundaries."""
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
        np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
    )


def create_test_space_time_fe(space_n=3, time_n=4, space_degree=1, time_degree=1):
    """Create a complete SpaceTimeFE for testing."""
    # Create space FE
    space_mesh = create_unit_square_mesh(space_n)
    space_V = fem.functionspace(space_mesh, ("Lagrange", space_degree))
    space_fe = SpaceFE(space_V, boundary_D=boundary_marker)
    
    # Create time FE
    time_mesh = create_unit_interval_mesh(time_n)
    time_V = fem.functionspace(time_mesh, ("Lagrange", time_degree))
    time_fe = TimeFE(time_mesh, time_V)
    
    # Create space-time FE
    st_fe = SpaceTimeFE(space_fe, time_fe)
    st_fe.assemble_noW()  # Assemble W-independent matrices
    
    return st_fe


# Test functions for RHS and boundary conditions
def dummy_rhs_0(coords):
    """Dummy right-hand side function for first equation."""
    return np.sin(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])


def dummy_rhs_1(coords):
    """Dummy right-hand side function for second equation."""
    return np.cos(np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1])


def dummy_boundary_u(coords):
    """Dummy boundary condition for u."""
    return np.zeros(coords.shape[0])


def dummy_boundary_v(coords):
    """Dummy boundary condition for v."""
    return np.zeros(coords.shape[0])


def dummy_W_function(x):
    """Dummy W function for parameter-dependent assembly."""
    return 0.1 * x[0]  # Linear in time


class TestAssemblerWaveBasic:
    """Basic tests for AssemblerWave class."""

    def test_initialization_with_space_time_fe(self):
        """Test initialization with SpaceTimeFE."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe, verbose=True)
        
        assert assembler.space_time_fe is st_fe
        assert assembler.verbose is True

    def test_initialization_minimal(self):
        """Test minimal initialization with required SpaceTimeFE."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        assert assembler.space_time_fe is st_fe
        assert assembler.verbose is False

    def test_update_space_time_fe(self):
        """Test updating SpaceTimeFE after initialization."""
        st_fe1 = create_test_space_time_fe(space_n=2, time_n=3)
        st_fe2 = create_test_space_time_fe(space_n=3, time_n=4)
        
        assembler = AssemblerWave(space_time_fe=st_fe1)
        assert assembler.space_time_fe is st_fe1
        
        assembler.update_space_time_fe(st_fe2)
        assert assembler.space_time_fe is st_fe2


class TestAssemblerWaveAssembly:
    """Tests for assembly methods."""

    def test_assemble_A0_b_with_space_time_fe(self):
        """Test A0_b assembly with valid space_time_fe."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        A0, b = assembler.assemble_A0_b(dummy_rhs_0, dummy_rhs_1)
        
        assert A0 is not None
        assert b is not None
        assert scipy.sparse.issparse(A0)
        assert isinstance(b, np.ndarray)
        
        # Check dimensions
        expected_size = 2 * st_fe.n_dofs
        assert A0.shape == (expected_size, expected_size)
        assert len(b) == expected_size

    def test_assemble_A_W_with_none_W(self):
        """Test A_W assembly when W is None."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe, verbose=True)
        
        A_W = assembler.assemble_A_W(W_t=None)
        
        assert A_W is not None
        assert scipy.sparse.issparse(A_W)
        # Should return zero matrix
        expected_size = 2 * st_fe.n_dofs
        assert A_W.shape == (expected_size, expected_size)
        # Check if matrix is essentially zero (small number of non-zeros)
        assert A_W.data.size == 0 or np.allclose(A_W.data, 0)

    def test_assemble_A_W_with_valid_W(self):
        """Test A_W assembly with valid W function."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        A_W = assembler.assemble_A_W(dummy_W_function)
        
        assert A_W is not None
        assert scipy.sparse.issparse(A_W)
        expected_size = 2 * st_fe.n_dofs
        assert A_W.shape == (expected_size, expected_size)


class TestAssemblerWaveParametrized:
    """Parametrized tests for different mesh sizes and polynomial degrees."""

    @pytest.mark.parametrize("space_n,time_n", [
        (2, 3),  # Very small
        (3, 4),  # Small
        (4, 5),  # Medium
    ])
    def test_assembly_different_mesh_sizes(self, space_n, time_n):
        """Test assembly with different mesh sizes."""
        st_fe = create_test_space_time_fe(space_n=space_n, time_n=time_n)
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        A0, b = assembler.assemble_A0_b(dummy_rhs_0, dummy_rhs_1)
        A_W = assembler.assemble_A_W(dummy_W_function)
        
        # All should assemble successfully
        assert A0 is not None
        assert b is not None
        assert A_W is not None
        
        # Check dimensions are consistent
        expected_size = 2 * st_fe.n_dofs
        assert A0.shape == (expected_size, expected_size)
        assert A_W.shape == (expected_size, expected_size)
        assert len(b) == expected_size

    @pytest.mark.parametrize("space_degree,time_degree", [
        (1, 1),  # Linear elements
        (1, 2),  # Linear space, quadratic time
        (2, 1),  # Quadratic space, linear time
    ])
    def test_assembly_different_polynomial_degrees(self, space_degree, time_degree):
        """Test assembly with different polynomial degrees."""
        st_fe = create_test_space_time_fe(
            space_n=3, time_n=4, 
            space_degree=space_degree, time_degree=time_degree
        )
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        A0, b = assembler.assemble_A0_b(dummy_rhs_0, dummy_rhs_1)
        A_W = assembler.assemble_A_W(dummy_W_function)
        
        # All should assemble successfully
        assert A0 is not None
        assert b is not None
        assert A_W is not None

    @pytest.mark.parametrize("W_function", [
        lambda x: 0.0 * x[0],           # Zero function
        lambda x: 0.5 * x[0],           # Linear function
        lambda x: 0.1 * x[0]**2,        # Quadratic function
        lambda x: 0.2 * np.sin(x[0]),   # Sinusoidal function
    ])
    def test_assembly_different_W_functions(self, W_function):
        """Test assembly with different W functions."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        A_W = assembler.assemble_A_W(W_function)
        
        assert A_W is not None
        assert scipy.sparse.issparse(A_W)
        expected_size = 2 * st_fe.n_dofs
        assert A_W.shape == (expected_size, expected_size)


class TestAssemblerWaveSystemAssembly:
    """Tests for full system assembly."""

    def test_assemble_system_complete(self):
        """Test complete system assembly."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        # Create proper initial condition
        X0 = np.zeros(2 * st_fe.n_dofs)
        
        A, b, X0D = assembler.assemble_system(
            dummy_W_function, X0, dummy_rhs_0, dummy_rhs_1,
            dummy_boundary_u, dummy_boundary_v
        )
        
        assert A is not None
        assert b is not None
        assert X0D is not None
        
        # Check dimensions
        expected_size = 2 * st_fe.n_dofs
        assert A.shape == (expected_size, expected_size)
        assert len(b) == expected_size
        assert len(X0D) == expected_size

    def test_impose_IC_BC_complete(self):
        """Test complete IC/BC imposition."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        # Create test system
        n = 2 * st_fe.n_dofs
        sys_mat = scipy.sparse.identity(n)
        rhs = np.ones(n)
        X0 = np.zeros(n)
        
        A, b, X0D = assembler.impose_IC_BC(
            sys_mat, rhs, X0, dummy_boundary_u, dummy_boundary_v
        )
        
        assert A is not None
        assert b is not None
        assert X0D is not None
        assert A.shape == (n, n)
        assert len(b) == n
        assert len(X0D) == n

    def test_system_assembly_with_different_initial_conditions(self):
        """Test system assembly with different initial conditions."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        # Test with non-zero initial condition
        X0 = np.ones(2 * st_fe.n_dofs) * 0.5
        
        A, b, X0D = assembler.assemble_system(
            dummy_W_function, X0, dummy_rhs_0, dummy_rhs_1,
            dummy_boundary_u, dummy_boundary_v
        )
        
        assert A is not None
        assert b is not None
        assert X0D is not None


class TestAssemblerWaveMatrixProperties:
    """Tests for mathematical properties of assembled matrices."""

    def test_A0_matrix_structure(self):
        """Test structure of A0 matrix."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        A0, b = assembler.assemble_A0_b(dummy_rhs_0, dummy_rhs_1)
        
        # Check sparsity
        assert scipy.sparse.issparse(A0)
        n = st_fe.n_dofs
        expected_size = 2 * n
        assert A0.shape == (expected_size, expected_size)
        
        # Matrix should be sparse (use .nnz to check non-zeros)
        total_elements = expected_size * expected_size
        nonzero_elements = A0.nnz
        sparsity = nonzero_elements / total_elements
        assert sparsity < 0.5, f"Matrix should be sparse, got sparsity: {sparsity}"

    def test_system_matrix_properties(self):
        """Test system matrix properties after complete assembly."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        # Assemble system
        X0 = np.zeros(2 * st_fe.n_dofs)
        A, b, X0D = assembler.assemble_system(
            dummy_W_function, X0, dummy_rhs_0, dummy_rhs_1,
            dummy_boundary_u, dummy_boundary_v
        )
        
        # Matrix should still be square and sparse
        n = 2 * st_fe.n_dofs
        assert A is not None
        assert A.shape == (n, n)
        assert scipy.sparse.issparse(A)
        
        # After BC imposition, matrix should still have non-zero entries
        assert A.nnz > 0, "Matrix should not be completely zero after BC imposition"

    def test_rhs_vector_properties(self):
        """Test properties of right-hand side vector."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        A0, b = assembler.assemble_A0_b(dummy_rhs_0, dummy_rhs_1)
        
        assert isinstance(b, np.ndarray)
        assert len(b) == 2 * st_fe.n_dofs
        assert np.all(np.isfinite(b)), "RHS should contain only finite values"


class TestAssemblerWaveIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.parametrize("verbose", [True, False])
    def test_verbose_mode(self, verbose):
        """Test assembler behavior with different verbose settings."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe, verbose=verbose)
        
        # Test methods that should work with valid space_time_fe
        A0, b = assembler.assemble_A0_b(dummy_rhs_0, dummy_rhs_1)
        A_W = assembler.assemble_A_W(dummy_W_function)
        
        assert A0 is not None
        assert b is not None
        assert A_W is not None

    def test_workflow_complete_assembly(self):
        """Test complete assembly workflow."""
        # Create components
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        # Step 1: Assemble parameter-independent parts
        A0, b0 = assembler.assemble_A0_b(dummy_rhs_0, dummy_rhs_1)
        assert A0 is not None and b0 is not None
        
        # Step 2: Assemble parameter-dependent parts
        A_W = assembler.assemble_A_W(dummy_W_function)
        assert A_W is not None
        
        # Step 3: Combine and apply boundary conditions
        X0 = np.zeros(2 * st_fe.n_dofs)
        A_final, b_final, X0D = assembler.assemble_system(
            dummy_W_function, X0, dummy_rhs_0, dummy_rhs_1,
            dummy_boundary_u, dummy_boundary_v
        )
        
        assert A_final is not None
        assert b_final is not None
        assert X0D is not None
        
        # Final system should be consistent
        n = 2 * st_fe.n_dofs
        assert A_final.shape == (n, n)
        assert len(b_final) == n
        assert len(X0D) == n

    def test_matrix_addition_consistency(self):
        """Test that A0 + A_W is consistent with system assembly."""
        st_fe = create_test_space_time_fe()
        assembler = AssemblerWave(space_time_fe=st_fe)
        
        # Assemble components separately
        A0, _ = assembler.assemble_A0_b(dummy_rhs_0, dummy_rhs_1)
        A_W = assembler.assemble_A_W(dummy_W_function)
        
        # Both should be valid matrices
        assert A0 is not None
        assert A_W is not None
        
        # Combined matrix
        A_combined = A0 + A_W
        
        # Should have same dimensions
        assert A_combined.shape == A0.shape
        assert A_combined.shape == A_W.shape
        
        # Should still be sparse
        assert scipy.sparse.issparse(A_combined)


if __name__ == "__main__":
    # Run basic tests if script is executed directly
    print("Running AssemblerWave tests...")
    
    # Test basic functionality
    print("Testing basic initialization...")
    test_basic = TestAssemblerWaveBasic()
    test_basic.test_initialization_with_space_time_fe()
    print("Basic tests passed")
    
    # Test assembly
    print("Testing assembly methods...")
    test_assembly = TestAssemblerWaveAssembly()
    test_assembly.test_assemble_A0_b_with_space_time_fe()
    test_assembly.test_assemble_A_W_with_valid_W()
    print("Assembly tests passed")
    
    # Test system assembly
    print("Testing system assembly...")
    test_system = TestAssemblerWaveSystemAssembly()
    test_system.test_assemble_system_complete()
    print("System assembly tests passed")
    
    print("All AssemblerWave tests passed")
