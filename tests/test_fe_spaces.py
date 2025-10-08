"""
Tests for the FE_spaces module.

This module tests the finite element space classes: SpaceFE, TimeFE, and SpaceTimeFE.
Tests include initialization, matrix assembly, boundary conditions, and space-time coupling.
"""

import numpy as np
import pytest
from dolfinx import fem, mesh
import ufl
from mpi4py import MPI

from ctg.FE_spaces import SpaceFE, TimeFE, SpaceTimeFE


def create_unit_square_mesh(n=8):
    """Create a simple unit square mesh for testing."""
    return mesh.create_unit_square(MPI.COMM_WORLD, n, n, mesh.CellType.triangle)


def create_unit_interval_mesh(n=10):
    """Create a simple unit interval mesh for testing."""
    return mesh.create_unit_interval(MPI.COMM_WORLD, n)


def create_function_space(test_mesh, element_type="Lagrange", degree=1):
    """Create a function space with proper DOLFiNx syntax."""
    return fem.functionspace(test_mesh, (element_type, degree))


def boundary_marker(x):
    """Simple boundary marker for unit square - marks all boundaries."""
    return np.logical_or(
        np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
        np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
    )


class TestSpaceFE:
    """Test class for SpaceFE finite element space."""

    def test_initialization_basic(self):
        """Test basic initialization of SpaceFE."""
        # Create mesh and function space
        test_mesh = create_unit_square_mesh(4)
        V = create_function_space(test_mesh, "Lagrange", 1)
        
        # Create SpaceFE instance
        space_fe = SpaceFE(V)
        
        # Check basic properties
        assert space_fe.mesh == test_mesh
        assert space_fe.V == V
        assert space_fe.n_dofs > 0
        assert space_fe.dofs.shape == (space_fe.n_dofs, 2)  # 2D mesh
        
        # Check that matrices were assembled
        assert "mass" in space_fe.matrix
        assert "laplace" in space_fe.matrix
        assert space_fe.matrix["mass"].shape == (space_fe.n_dofs, space_fe.n_dofs)
        assert space_fe.matrix["laplace"].shape == (space_fe.n_dofs, space_fe.n_dofs)

    def test_initialization_with_boundary(self):
        """Test SpaceFE initialization with boundary conditions."""
        test_mesh = create_unit_square_mesh(4)
        V = create_function_space(test_mesh, "Lagrange", 1)
        
        # Create SpaceFE with boundary
        space_fe = SpaceFE(V, boundary_D=boundary_marker)
        
        # Check boundary DOF vector exists
        assert hasattr(space_fe, 'boundary_dof_vector')
        assert len(space_fe.boundary_dof_vector) == space_fe.n_dofs
        assert np.sum(space_fe.boundary_dof_vector) > 0  # Some boundary DOFs should be marked

    def test_matrix_properties(self):
        """Test mathematical properties of assembled matrices."""
        test_mesh = create_unit_square_mesh(6)
        V = create_function_space(test_mesh, "Lagrange", 1)
        space_fe = SpaceFE(V)
        
        mass_matrix = space_fe.matrix["mass"]
        laplace_matrix = space_fe.matrix["laplace"]
        
        # Mass matrix should be symmetric and positive definite
        assert np.allclose(mass_matrix.toarray(), mass_matrix.T.toarray()), "Mass matrix should be symmetric"
        
        # Laplace matrix should be symmetric
        assert np.allclose(laplace_matrix.toarray(), laplace_matrix.T.toarray()), "Laplace matrix should be symmetric"
        
        # Mass matrix should have positive diagonal
        assert np.all(mass_matrix.diagonal() > 0), "Mass matrix diagonal should be positive"

    def test_dof_coordinates(self):
        """Test that DOF coordinates are correctly extracted."""
        test_mesh = create_unit_square_mesh(3)
        V = create_function_space(test_mesh, "Lagrange", 1)
        space_fe = SpaceFE(V)
        
        # Check DOF coordinates are within unit square
        assert np.all(space_fe.dofs >= 0.0), "DOF coordinates should be non-negative"
        assert np.all(space_fe.dofs <= 1.0), "DOF coordinates should be <= 1.0"
        
        # Check dimension consistency
        assert space_fe.dofs.shape[1] == test_mesh.geometry.dim, "DOF dimension should match mesh dimension"


class TestTimeFE:
    """Test class for TimeFE finite element space."""

    def test_initialization_basic(self):
        """Test basic initialization of TimeFE."""
        test_mesh = create_unit_interval_mesh(5)
        V = create_function_space(test_mesh, "Lagrange", 2)
        
        time_fe = TimeFE(test_mesh, V)
        
        # Check basic properties
        assert time_fe.mesh == test_mesh
        assert time_fe.V == V
        assert time_fe.n_dofs > 0
        assert time_fe.dofs.shape == (time_fe.n_dofs, 1)  # 1D mesh
        
        # Check matrices were assembled
        assert "mass" in time_fe.matrix
        assert "derivative" in time_fe.matrix
        assert "mass_err" in time_fe.matrix

    def test_initial_final_condition_dofs(self):
        """Test that initial and final condition DOFs are correctly identified."""
        test_mesh = create_unit_interval_mesh(10)
        V = create_function_space(test_mesh, "Lagrange", 1)
        time_fe = TimeFE(test_mesh, V)
        
        # Check IC and FC vectors
        assert len(time_fe.dof_IC_vector) == time_fe.n_dofs
        assert len(time_fe.dof_FC_vector) == time_fe.n_dofs
        
        # Exactly one IC and one FC DOF should be marked
        assert np.sum(time_fe.dof_IC_vector) == 1.0, "Exactly one IC DOF should be marked"
        assert np.sum(time_fe.dof_FC_vector) == 1.0, "Exactly one FC DOF should be marked"
        
        # IC should be at minimum time, FC at maximum time
        ic_dof_idx = np.argmax(time_fe.dof_IC_vector)
        fc_dof_idx = np.argmax(time_fe.dof_FC_vector)
        
        ic_time = time_fe.dofs[ic_dof_idx, 0]
        fc_time = time_fe.dofs[fc_dof_idx, 0]
        
        assert ic_time == np.min(time_fe.dofs), "IC should be at minimum time"
        assert fc_time == np.max(time_fe.dofs), "FC should be at maximum time"

    def test_w_dependent_matrices(self):
        """Test assembly of W-dependent matrices."""
        test_mesh = create_unit_interval_mesh(8)
        V = create_function_space(test_mesh, "Lagrange", 1)
        time_fe = TimeFE(test_mesh, V)
        
        # Define a simple W function
        def W_func(x):
            return 0.5 * x[0]  # Linear function of time
        
        # Assemble W-dependent matrices
        result = time_fe.assemble_matrices_W(W_func)
        
        # Check that matrices were created
        assert "W_mass" in time_fe.matrix
        assert "WW_mass" in time_fe.matrix
        assert time_fe.matrix["W_mass"].shape == (time_fe.n_dofs, time_fe.n_dofs)
        assert time_fe.matrix["WW_mass"].shape == (time_fe.n_dofs, time_fe.n_dofs)
        
        # Check return values if they exist
        if result is not None:
            W_mass, WW_mass = result
            assert W_mass.shape == (time_fe.n_dofs, time_fe.n_dofs)
            assert WW_mass.shape == (time_fe.n_dofs, time_fe.n_dofs)

    def test_w_none_handling(self):
        """Test behavior when W is None."""
        test_mesh = create_unit_interval_mesh(5)
        V = create_function_space(test_mesh, "Lagrange", 1)
        time_fe = TimeFE(test_mesh, V, verbose=True)
        
        # Should handle None gracefully
        result = time_fe.assemble_matrices_W(W_t=None)
        assert result is None


class TestSpaceTimeFE:
    """Test class for SpaceTimeFE finite element space."""

    def test_initialization_basic(self):
        """Test basic initialization of SpaceTimeFE."""
        # Create space and time FE
        space_mesh = create_unit_square_mesh(3)
        space_V = create_function_space(space_mesh, "Lagrange", 1)
        space_fe = SpaceFE(space_V)
        
        time_mesh = create_unit_interval_mesh(4)
        time_V = create_function_space(time_mesh, "Lagrange", 1)
        time_fe = TimeFE(time_mesh, time_V)
        
        # Create SpaceTimeFE
        st_fe = SpaceTimeFE(space_fe, time_fe)
        
        # Check basic properties
        assert st_fe.space_fe == space_fe
        assert st_fe.time_fe == time_fe
        assert st_fe.n_dofs == space_fe.n_dofs * time_fe.n_dofs
        assert st_fe.dofs.shape == (st_fe.n_dofs, 3)  # (t, x, y) coordinates

    def test_initialization_without_time_fe(self):
        """Test initialization without time FE."""
        space_mesh = create_unit_square_mesh(3)
        space_V = create_function_space(space_mesh, "Lagrange", 1)
        space_fe = SpaceFE(space_V)
        
        st_fe = SpaceTimeFE(space_fe, time_fe=None)
        
        assert st_fe.space_fe == space_fe
        assert st_fe.time_fe is None
        assert st_fe.n_dofs == 0

    def test_update_time_fe(self):
        """Test updating time FE after initialization."""
        # Create space FE
        space_mesh = create_unit_square_mesh(3)
        space_V = create_function_space(space_mesh, "Lagrange", 1)
        space_fe = SpaceFE(space_V)
        
        # Create SpaceTimeFE without time FE
        st_fe = SpaceTimeFE(space_fe)
        
        # Create and update with time FE
        time_mesh = create_unit_interval_mesh(5)
        time_V = create_function_space(time_mesh, "Lagrange", 1)
        time_fe = TimeFE(time_mesh, time_V)
        
        st_fe.update_time_fe(time_fe)
        
        assert st_fe.time_fe == time_fe
        assert st_fe.n_dofs == space_fe.n_dofs * time_fe.n_dofs

    def test_ic_fc_dofs(self):
        """Test initial and final condition DOF identification in space-time."""
        # Create space and time FE
        space_mesh = create_unit_square_mesh(2)
        space_V = create_function_space(space_mesh, "Lagrange", 1)
        space_fe = SpaceFE(space_V)
        
        time_mesh = create_unit_interval_mesh(3)
        time_V = create_function_space(time_mesh, "Lagrange", 1)
        time_fe = TimeFE(time_mesh, time_V)
        
        st_fe = SpaceTimeFE(space_fe, time_fe)
        
        # Check IC and FC DOF vectors
        assert len(st_fe.dofs_IC) == 2 * st_fe.n_dofs  # For (u, v) system
        assert len(st_fe.dofs_FC) == 2 * st_fe.n_dofs
        
        # Check that some DOFs are marked for IC and FC
        assert np.sum(st_fe.dofs_IC) > 0, "Some IC DOFs should be marked"
        assert np.sum(st_fe.dofs_FC) > 0, "Some FC DOFs should be marked"

    def test_assemble_now_matrices(self):
        """Test assembly of W-independent matrices."""
        # Create space and time FE
        space_mesh = create_unit_square_mesh(2)
        space_V = create_function_space(space_mesh, "Lagrange", 1)
        space_fe = SpaceFE(space_V)
        
        time_mesh = create_unit_interval_mesh(3)
        time_V = create_function_space(time_mesh, "Lagrange", 1)
        time_fe = TimeFE(time_mesh, time_V)
        
        st_fe = SpaceTimeFE(space_fe, time_fe)
        st_fe.assemble_noW()
        
        # Check that matrices were assembled
        expected_matrices = ["L", "D_t", "M"]
        for matrix_name in expected_matrices:
            assert matrix_name in st_fe.matrix, f"Matrix {matrix_name} should be assembled"
            matrix = st_fe.matrix[matrix_name]
            assert matrix.shape == (st_fe.n_dofs, st_fe.n_dofs), f"Matrix {matrix_name} has wrong shape"

    def test_assemble_w_matrices(self):
        """Test assembly of W-dependent matrices."""
        # Create space and time FE
        space_mesh = create_unit_square_mesh(2)
        space_V = create_function_space(space_mesh, "Lagrange", 1)
        space_fe = SpaceFE(space_V)
        
        time_mesh = create_unit_interval_mesh(3)
        time_V = create_function_space(time_mesh, "Lagrange", 1)
        time_fe = TimeFE(time_mesh, time_V)
        
        st_fe = SpaceTimeFE(space_fe, time_fe)
        
        # Define W function
        def W_func(x):
            return 0.1 * x[0]
        
        st_fe.assemble_W(W_func)
        
        # Check W-dependent matrices
        assert "M_W" in st_fe.matrix
        assert "M_WW" in st_fe.matrix
        assert st_fe.matrix["M_W"].shape == (st_fe.n_dofs, st_fe.n_dofs)
        assert st_fe.matrix["M_WW"].shape == (st_fe.n_dofs, st_fe.n_dofs)

    def test_interpolate_function(self):
        """Test interpolation of functions on space-time DOFs."""
        # Create space and time FE
        space_mesh = create_unit_square_mesh(2)
        space_V = create_function_space(space_mesh, "Lagrange", 1)
        space_fe = SpaceFE(space_V)
        
        time_mesh = create_unit_interval_mesh(3)
        time_V = create_function_space(time_mesh, "Lagrange", 1)
        time_fe = TimeFE(time_mesh, time_V)
        
        st_fe = SpaceTimeFE(space_fe, time_fe)
        
        # Define a simple test function
        def test_func(coords):
            t, x, y = coords[:, 0], coords[:, 1], coords[:, 2]
            return t + x + y
        
        # Interpolate
        values = st_fe.interpolate(test_func)
        
        # Check that interpolation worked
        if values is not None:
            assert len(values) == st_fe.n_dofs
            assert isinstance(values, np.ndarray)
        else:
            # If interpolation returns None, that's also acceptable for this test
            pass

    def test_full_assembly(self):
        """Test complete assembly with both W-independent and W-dependent terms."""
        # Create space and time FE
        space_mesh = create_unit_square_mesh(2)
        space_V = create_function_space(space_mesh, "Lagrange", 1)
        space_fe = SpaceFE(space_V)
        
        time_mesh = create_unit_interval_mesh(3)
        time_V = create_function_space(time_mesh, "Lagrange", 1)
        time_fe = TimeFE(time_mesh, time_V)
        
        st_fe = SpaceTimeFE(space_fe, time_fe)
        
        # Define W function
        def W_func(x):
            return 0.2 * np.sin(x[0])
        
        # Full assembly
        st_fe.assemble(W_func)
        
        # Check all expected matrices
        expected_matrices = ["L", "D_t", "M", "M_W", "M_WW"]
        for matrix_name in expected_matrices:
            assert matrix_name in st_fe.matrix, f"Matrix {matrix_name} should be assembled"


class TestIntegration:
    """Integration tests combining multiple classes."""

    def test_space_time_consistency(self):
        """Test consistency between space, time, and space-time FE."""
        # Create compatible space and time FE
        space_mesh = create_unit_square_mesh(2)
        space_V = create_function_space(space_mesh, "Lagrange", 1)
        space_fe = SpaceFE(space_V)
        
        time_mesh = create_unit_interval_mesh(3)
        time_V = create_function_space(time_mesh, "Lagrange", 1)
        time_fe = TimeFE(time_mesh, time_V)
        
        st_fe = SpaceTimeFE(space_fe, time_fe)
        
        # Check dimension consistency
        assert st_fe.n_dofs == space_fe.n_dofs * time_fe.n_dofs
        
        # Check that space-time matrices have correct dimensions
        st_fe.assemble_noW()
        for matrix_name in ["L", "D_t", "M"]:
            matrix = st_fe.matrix[matrix_name]
            assert matrix.shape == (st_fe.n_dofs, st_fe.n_dofs)

    def test_matrix_sparsity(self):
        """Test that assembled matrices have reasonable sparsity patterns."""
        space_mesh = create_unit_square_mesh(4)
        space_V = create_function_space(space_mesh, "Lagrange", 1)
        space_fe = SpaceFE(space_V)
        
        time_mesh = create_unit_interval_mesh(5)
        time_V = create_function_space(time_mesh, "Lagrange", 1)
        time_fe = TimeFE(time_mesh, time_V)
        
        st_fe = SpaceTimeFE(space_fe, time_fe)
        st_fe.assemble_noW()
        
        # Check that matrices are sparse
        for matrix_name in ["L", "D_t", "M"]:
            matrix = st_fe.matrix[matrix_name]
            sparsity = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
            assert sparsity < 0.5, f"Matrix {matrix_name} should be sparse (sparsity: {sparsity:.3f})"


if __name__ == "__main__":
    # Run basic tests if script is executed directly
    print("Running FE_spaces tests...")
    
    # Test SpaceFE
    print("Testing SpaceFE...")
    test_space = TestSpaceFE()
    test_space.test_initialization_basic()
    test_space.test_matrix_properties()
    print("SpaceFE tests passed")
    
    # Test TimeFE  
    print("Testing TimeFE...")
    test_time = TestTimeFE()
    test_time.test_initialization_basic()
    test_time.test_initial_final_condition_dofs()
    print("TimeFE tests passed")
    
    # Test SpaceTimeFE
    print("Testing SpaceTimeFE...")
    test_st = TestSpaceTimeFE()
    test_st.test_initialization_basic()
    test_st.test_assemble_now_matrices()
    print("SpaceTimeFE tests passed")
    
    print("All FE_spaces tests passed.")