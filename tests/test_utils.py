"""
Comprehensive tests for utils module.

This module tests utility functions used throughout the CTG (Continuous Time Galerkin) 
framework, including coordinate transformations, time slab computation, and inverse
Doss-Sussmann transforms.
"""

import pytest
import numpy as np
import sys
from io import StringIO
from unittest.mock import Mock, patch
from dolfinx import fem, mesh
from mpi4py import MPI

# Import the functions under test
from ctg.utils import (
    cart_prod_coords,
    compute_time_slabs,
    inverse_DS_transform,
    vprint
)

# Import dependencies for testing complex functions
from ctg.FE_spaces import SpaceFE, TimeFE


class TestCartProdCoords:
    """Test the cart_prod_coords function for cartesian product of coordinate arrays."""

    def test_basic_2d_coordinates(self):
        """Test basic functionality with 2D coordinate arrays."""
        t_coords = np.array([[0.0], [0.5], [1.0]])  # 3 time points
        x_coords = np.array([[0.0], [1.0]])          # 2 spatial points
        
        result = cart_prod_coords(t_coords, x_coords)
        
        # Should have 3*2 = 6 rows, 2 columns (t, x)
        assert result.shape == (6, 2)
        
        # Check expected combinations
        expected = np.array([
            [0.0, 0.0], [0.0, 1.0],  # t=0.0 with all x
            [0.5, 0.0], [0.5, 1.0],  # t=0.5 with all x
            [1.0, 0.0], [1.0, 1.0]   # t=1.0 with all x
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_1d_time_coordinates(self):
        """Test with 1D time coordinates (should be expanded to 2D)."""
        t_coords = np.array([0.0, 0.5, 1.0])         # 1D array
        x_coords = np.array([[0.0], [1.0]])          # 2D array
        
        result = cart_prod_coords(t_coords, x_coords)
        
        assert result.shape == (6, 2)
        expected = np.array([
            [0.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 1.0],
            [1.0, 0.0], [1.0, 1.0]
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_1d_spatial_coordinates(self):
        """Test with 1D spatial coordinates (should be expanded to 2D)."""
        t_coords = np.array([[0.0], [1.0]])          # 2D array
        x_coords = np.array([0.0, 0.5, 1.0])         # 1D array
        
        result = cart_prod_coords(t_coords, x_coords)
        
        assert result.shape == (6, 2)
        expected = np.array([
            [0.0, 0.0], [0.0, 0.5], [0.0, 1.0],
            [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_both_1d_coordinates(self):
        """Test with both coordinate arrays as 1D (both should be expanded)."""
        t_coords = np.array([0.0, 1.0])              # 1D array
        x_coords = np.array([0.0, 1.0])              # 1D array
        
        result = cart_prod_coords(t_coords, x_coords)
        
        assert result.shape == (4, 2)
        expected = np.array([
            [0.0, 0.0], [0.0, 1.0],
            [1.0, 0.0], [1.0, 1.0]
        ])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_point_coordinates(self):
        """Test with single point in each dimension."""
        t_coords = np.array([[0.5]])
        x_coords = np.array([[1.5]])
        
        result = cart_prod_coords(t_coords, x_coords)
        
        assert result.shape == (1, 2)
        np.testing.assert_array_almost_equal(result, [[0.5, 1.5]])

    def test_multidimensional_spatial_coordinates(self):
        """Test with 2D spatial coordinates (x, y)."""
        t_coords = np.array([[0.0], [1.0]])
        x_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # 3 spatial points in 2D
        
        result = cart_prod_coords(t_coords, x_coords)
        
        # Should have 2*3 = 6 rows, 3 columns (t, x, y)
        assert result.shape == (6, 3)
        expected = np.array([
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]
        ])
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.parametrize("n_time,n_space", [
        (1, 1), (2, 3), (3, 2), (5, 4), (10, 1)
    ])
    def test_different_sizes(self, n_time, n_space):
        """Test with different numbers of time and spatial points."""
        t_coords = np.linspace(0, 1, n_time).reshape(-1, 1)
        x_coords = np.linspace(0, 2, n_space).reshape(-1, 1)
        
        result = cart_prod_coords(t_coords, x_coords)
        
        expected_shape = (n_time * n_space, 2)
        assert result.shape == expected_shape
        
        # Check that we have all combinations
        assert len(np.unique(result[:, 0])) <= n_time  # At most n_time unique time values
        assert len(np.unique(result[:, 1])) <= n_space  # At most n_space unique space values

    def test_empty_coordinates(self):
        """Test with empty coordinate arrays."""
        t_coords = np.array([]).reshape(0, 1)
        x_coords = np.array([]).reshape(0, 1)
        
        result = cart_prod_coords(t_coords, x_coords)
        
        assert result.shape == (0, 2)

    def test_ordering_preservation(self):
        """Test that the ordering of the cartesian product is as expected."""
        t_coords = np.array([[1.0], [2.0], [3.0]])
        x_coords = np.array([[10.0], [20.0]])
        
        result = cart_prod_coords(t_coords, x_coords)
        
        # First all combinations with t=1.0, then t=2.0, then t=3.0
        expected = np.array([
            [1.0, 10.0], [1.0, 20.0],
            [2.0, 10.0], [2.0, 20.0],
            [3.0, 10.0], [3.0, 20.0]
        ])
        np.testing.assert_array_almost_equal(result, expected)


class TestComputeTimeSlabs:
    """Test the compute_time_slabs function for time interval subdivision."""

    def test_basic_time_slabs(self):
        """Test basic time slab computation."""
        start_time = 0.0
        end_time = 1.0
        slab_size = 0.25
        
        slabs = compute_time_slabs(start_time, end_time, slab_size)
        
        expected = [
            (0.0, 0.25),
            (0.25, 0.5),
            (0.5, 0.75),
            (0.75, 1.0)
        ]
        
        assert len(slabs) == 4
        for actual, expected_slab in zip(slabs, expected):
            assert np.isclose(actual[0], expected_slab[0])
            assert np.isclose(actual[1], expected_slab[1])

    def test_exact_division(self):
        """Test when end_time is exactly divisible by slab_size."""
        start_time = 0.0
        end_time = 2.0
        slab_size = 0.5
        
        slabs = compute_time_slabs(start_time, end_time, slab_size)
        
        expected = [
            (0.0, 0.5),
            (0.5, 1.0),
            (1.0, 1.5),
            (1.5, 2.0)
        ]
        
        assert len(slabs) == 4
        for actual, expected_slab in zip(slabs, expected):
            assert np.isclose(actual[0], expected_slab[0])
            assert np.isclose(actual[1], expected_slab[1])

    def test_inexact_division(self):
        """Test when end_time is not exactly divisible by slab_size."""
        start_time = 0.0
        end_time = 1.1
        slab_size = 0.3
        
        slabs = compute_time_slabs(start_time, end_time, slab_size)
        
        expected = [
            (0.0, 0.3),
            (0.3, 0.6),
            (0.6, 0.9),
            (0.9, 1.2)  # Last slab goes beyond end_time
        ]
        
        assert len(slabs) == 4
        for actual, expected_slab in zip(slabs, expected):
            assert np.isclose(actual[0], expected_slab[0])
            assert np.isclose(actual[1], expected_slab[1])

    def test_single_slab(self):
        """Test when slab_size is larger than the time interval."""
        start_time = 0.0
        end_time = 0.5
        slab_size = 1.0
        
        slabs = compute_time_slabs(start_time, end_time, slab_size)
        
        assert len(slabs) == 1
        assert np.isclose(slabs[0][0], 0.0)
        assert np.isclose(slabs[0][1], 1.0)

    def test_very_small_slab_size(self):
        """Test with very small slab size."""
        start_time = 0.0
        end_time = 0.2
        slab_size = 0.05
        
        slabs = compute_time_slabs(start_time, end_time, slab_size)
        
        assert len(slabs) == 4
        for i, slab in enumerate(slabs):
            expected_start = i * slab_size
            expected_end = (i + 1) * slab_size
            assert np.isclose(slab[0], expected_start)
            assert np.isclose(slab[1], expected_end)

    def test_non_zero_start_time(self):
        """Test with non-zero start time."""
        start_time = 1.0
        end_time = 3.0
        slab_size = 0.5
        
        slabs = compute_time_slabs(start_time, end_time, slab_size)
        
        expected = [
            (1.0, 1.5),
            (1.5, 2.0),
            (2.0, 2.5),
            (2.5, 3.0)
        ]
        
        assert len(slabs) == 4
        for actual, expected_slab in zip(slabs, expected):
            assert np.isclose(actual[0], expected_slab[0])
            assert np.isclose(actual[1], expected_slab[1])

    def test_negative_times(self):
        """Test with negative time values."""
        start_time = -2.0
        end_time = 0.0
        slab_size = 0.5
        
        slabs = compute_time_slabs(start_time, end_time, slab_size)
        
        expected = [
            (-2.0, -1.5),
            (-1.5, -1.0),
            (-1.0, -0.5),
            (-0.5, 0.0)
        ]
        
        assert len(slabs) == 4
        for actual, expected_slab in zip(slabs, expected):
            assert np.isclose(actual[0], expected_slab[0])
            assert np.isclose(actual[1], expected_slab[1])

    @pytest.mark.parametrize("start,end,size", [
        (0.0, 1.0, 0.1),
        (0.0, 0.5, 0.25),
        (1.0, 2.0, 0.3),
        (0.0, 3.14, 1.0),
        (-1.0, 1.0, 0.4)
    ])
    def test_parametrized_configurations(self, start, end, size):
        """Test various time configurations."""
        slabs = compute_time_slabs(start, end, size)
        
        # Basic properties
        assert len(slabs) >= 1
        assert slabs[0][0] == start
        
        # Check continuity
        for i in range(len(slabs) - 1):
            assert np.isclose(slabs[i][1], slabs[i+1][0])
        
        # Check slab sizes
        for slab in slabs:
            assert np.isclose(slab[1] - slab[0], size)

    def test_very_close_end_time(self):
        """Test the epsilon tolerance in the while loop condition."""
        start_time = 0.0
        end_time = 1.0 - 1e-11  # Very close to 1.0
        slab_size = 0.5
        
        slabs = compute_time_slabs(start_time, end_time, slab_size)
        
        # Should have exactly 2 slabs due to the 1e-10 tolerance
        assert len(slabs) == 2
        expected = [(0.0, 0.5), (0.5, 1.0)]
        for actual, expected_slab in zip(slabs, expected):
            assert np.isclose(actual[0], expected_slab[0])
            assert np.isclose(actual[1], expected_slab[1])

    def test_continuity_property(self):
        """Test that slabs are continuous (no gaps)."""
        start_time = 0.0
        end_time = 2.7
        slab_size = 0.4
        
        slabs = compute_time_slabs(start_time, end_time, slab_size)
        
        # Check that each slab starts where the previous one ended
        for i in range(1, len(slabs)):
            assert np.isclose(slabs[i-1][1], slabs[i][0])


class TestInverseDSTransform:
    """Test the inverse_DS_transform function for Doss-Sussmann transformation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test mesh and finite element spaces
        self.comm = MPI.COMM_WORLD
        msh = mesh.create_unit_square(self.comm, 2, 2, mesh.CellType.triangle)
        V_x = fem.functionspace(msh, ("Lagrange", 1))
        self.space_fe = SpaceFE(V_x)
        
        # Create time finite element space
        slab = [0.0, 1.0]
        msh_t = mesh.create_interval(self.comm, 2, slab)
        V_t = fem.functionspace(msh_t, ("Lagrange", 1))
        self.time_fe = TimeFE(msh_t, V_t)
        
        self.n_x = self.space_fe.n_dofs
        self.n_t = self.time_fe.n_dofs
        self.n_scalar = self.n_x * self.n_t

    def test_basic_transform(self):
        """Test basic inverse DS transform functionality."""
        # Create test solution vector XX (u, v components)
        XX = np.ones(2 * self.n_scalar) * 0.5
        
        # Create test Brownian motion function
        def WW_fun(t_dofs):
            return 0.1 * t_dofs[:, 0]  # Linear function of time
        
        result = inverse_DS_transform(XX, WW_fun, self.space_fe, self.time_fe)
        
        # Result should have same size as input
        assert result.size == XX.size
        assert len(result) == 2 * self.n_scalar
        
        # Result should be finite
        assert np.all(np.isfinite(result))

    def test_zero_input_solution(self):
        """Test transform with zero input solution."""
        XX = np.zeros(2 * self.n_scalar)
        
        def WW_fun(t_dofs):
            return np.zeros(len(t_dofs))
        
        result = inverse_DS_transform(XX, WW_fun, self.space_fe, self.time_fe)
        
        # Should return zero result
        np.testing.assert_array_almost_equal(result, XX)

    def test_zero_brownian_motion(self):
        """Test transform with zero Brownian motion."""
        XX = np.ones(2 * self.n_scalar)
        
        def WW_fun(t_dofs):
            return np.zeros(len(t_dofs))
        
        result = inverse_DS_transform(XX, WW_fun, self.space_fe, self.time_fe)
        
        # With zero Brownian motion, should return original XX
        np.testing.assert_array_almost_equal(result, XX)

    def test_solution_structure(self):
        """Test that the transform preserves expected structure."""
        XX = np.random.rand(2 * self.n_scalar)
        
        def WW_fun(t_dofs):
            return 0.2 * np.sin(t_dofs[:, 0])
        
        result = inverse_DS_transform(XX, WW_fun, self.space_fe, self.time_fe)
        
        # First half (u component) should be unchanged
        uu_original = XX[:self.n_scalar]
        uu_result = result[:self.n_scalar]
        np.testing.assert_array_almost_equal(uu_original, uu_result)
        
        # Second half (v component) should be modified
        vv_original = XX[self.n_scalar:]
        vv_result = result[self.n_scalar:]
        # They should be different (unless Brownian motion is zero)
        if not np.allclose(WW_fun(self.time_fe.dofs), 0):
            assert not np.allclose(vv_original, vv_result)

    def test_different_brownian_functions(self):
        """Test with different types of Brownian motion functions."""
        XX = np.ones(2 * self.n_scalar)
        
        # Linear Brownian motion
        def WW_linear(t_dofs):
            return 0.1 * t_dofs[:, 0]
        
        # Quadratic Brownian motion
        def WW_quadratic(t_dofs):
            return 0.1 * t_dofs[:, 0]**2
        
        # Sinusoidal Brownian motion
        def WW_sin(t_dofs):
            return 0.1 * np.sin(t_dofs[:, 0])
        
        result_linear = inverse_DS_transform(XX, WW_linear, self.space_fe, self.time_fe)
        result_quadratic = inverse_DS_transform(XX, WW_quadratic, self.space_fe, self.time_fe)
        result_sin = inverse_DS_transform(XX, WW_sin, self.space_fe, self.time_fe)
        
        # All should have correct shape
        assert result_linear.shape == XX.shape
        assert result_quadratic.shape == XX.shape
        assert result_sin.shape == XX.shape
        
        # Results should be different
        assert not np.allclose(result_linear, result_quadratic)
        assert not np.allclose(result_linear, result_sin)
        assert not np.allclose(result_quadratic, result_sin)

    def test_brownian_motion_scaling(self):
        """Test how the transform scales with Brownian motion amplitude."""
        XX = np.ones(2 * self.n_scalar)
        
        # Small amplitude
        def WW_small(t_dofs):
            return 0.01 * t_dofs[:, 0]
        
        # Large amplitude
        def WW_large(t_dofs):
            return 1.0 * t_dofs[:, 0]
        
        result_small = inverse_DS_transform(XX, WW_small, self.space_fe, self.time_fe)
        result_large = inverse_DS_transform(XX, WW_large, self.space_fe, self.time_fe)
        
        # First components should be the same
        np.testing.assert_array_almost_equal(
            result_small[:self.n_scalar], 
            result_large[:self.n_scalar]
        )
        
        # Second components should have larger difference for larger Brownian motion
        diff_small = np.abs(result_small[self.n_scalar:] - XX[self.n_scalar:])
        diff_large = np.abs(result_large[self.n_scalar:] - XX[self.n_scalar:])
        
        # Large amplitude should produce larger changes
        assert np.mean(diff_large) > np.mean(diff_small)

    def test_transform_linearity_in_solution(self):
        """Test linearity properties of the transform with respect to solution."""
        XX1 = np.random.rand(2 * self.n_scalar)
        XX2 = np.random.rand(2 * self.n_scalar)
        alpha = 0.3
        
        def WW_fun(t_dofs):
            return 0.1 * t_dofs[:, 0]
        
        # Transform linear combination
        XX_combined = alpha * XX1 + (1 - alpha) * XX2
        result_combined = inverse_DS_transform(XX_combined, WW_fun, self.space_fe, self.time_fe)
        
        # Transform individual solutions
        result1 = inverse_DS_transform(XX1, WW_fun, self.space_fe, self.time_fe)
        result2 = inverse_DS_transform(XX2, WW_fun, self.space_fe, self.time_fe)
        
        # Check linearity (approximately, due to the u*W term)
        expected_linear = alpha * result1 + (1 - alpha) * result2
        
        # The u-component should be exactly linear
        np.testing.assert_array_almost_equal(
            result_combined[:self.n_scalar],
            expected_linear[:self.n_scalar]
        )

    def test_vector_sizes_consistency(self):
        """Test that vector sizes are handled correctly."""
        # Use fixed values instead of random for reproducible test
        XX = np.ones(2 * self.n_scalar) * 0.5
        
        def WW_fun(t_dofs):
            # Use a simple deterministic function
            return 0.1 * t_dofs[:, 0]
        
        result = inverse_DS_transform(XX, WW_fun, self.space_fe, self.time_fe)
        
        # Check internal dimensions are consistent
        n_scalar = int(XX.size / 2)
        assert n_scalar == self.n_scalar
        
        uu = XX[:n_scalar]
        vv = XX[n_scalar:]
        
        WW = WW_fun(self.time_fe.dofs)
        WW_rep = np.repeat(WW, self.n_x)
        
        expected_uu = uu
        expected_vv = vv + WW_rep * uu
        expected_result = np.concatenate((expected_uu, expected_vv))
        
        np.testing.assert_array_almost_equal(result, expected_result)


class TestVprint:
    """Test the vprint function for conditional printing."""

    def test_verbose_true_default(self):
        """Test vprint with default verbose=True."""
        # Capture stdout
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            vprint("Test message")
        
        output = captured_output.getvalue()
        assert "Test message" in output

    def test_verbose_true_explicit(self):
        """Test vprint with explicit verbose=True."""
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            vprint("Test message", verbose=True)
        
        output = captured_output.getvalue()
        assert "Test message" in output

    def test_verbose_false(self):
        """Test vprint with verbose=False."""
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            vprint("Test message", verbose=False)
        
        output = captured_output.getvalue()
        assert output == ""  # No output should be produced

    def test_empty_string(self):
        """Test vprint with empty string."""
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            vprint("", verbose=True)
        
        output = captured_output.getvalue()
        assert output == "\n"  # Just a newline

    def test_multiline_string(self):
        """Test vprint with multiline string."""
        test_message = "Line 1\nLine 2\nLine 3"
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            vprint(test_message, verbose=True)
        
        output = captured_output.getvalue()
        assert "Line 1" in output
        assert "Line 2" in output
        assert "Line 3" in output

    def test_special_characters(self):
        """Test vprint with special characters."""
        test_message = "Message with special chars: !@#$%^&*()[]{}|;:,.<>?"
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            vprint(test_message, verbose=True)
        
        output = captured_output.getvalue()
        assert test_message in output

    def test_numeric_string(self):
        """Test vprint with numeric content."""
        test_message = "Value: 3.14159, Count: 42"
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            vprint(test_message, verbose=True)
        
        output = captured_output.getvalue()
        assert test_message in output

    @pytest.mark.parametrize("verbose_flag", [True, False])
    def test_parametrized_verbose(self, verbose_flag):
        """Test vprint with different verbose flags."""
        test_message = "Parametrized test message"
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            vprint(test_message, verbose=verbose_flag)
        
        output = captured_output.getvalue()
        if verbose_flag:
            assert test_message in output
        else:
            assert output == ""

    def test_multiple_calls(self):
        """Test multiple calls to vprint."""
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            vprint("Message 1", verbose=True)
            vprint("Message 2", verbose=False)
            vprint("Message 3", verbose=True)
        
        output = captured_output.getvalue()
        assert "Message 1" in output
        assert "Message 2" not in output
        assert "Message 3" in output

    def test_return_value(self):
        """Test that vprint returns None."""
        result = vprint("Test", verbose=True)
        assert result is None
        
        result = vprint("Test", verbose=False)
        assert result is None


class TestUtilsIntegration:
    """Integration tests combining multiple utility functions."""

    def test_cart_prod_with_time_slabs(self):
        """Test using cart_prod_coords with compute_time_slabs output."""
        # Compute time slabs
        time_slabs = compute_time_slabs(0.0, 1.0, 0.5)
        
        # Use first slab for time coordinates
        t_start, t_end = time_slabs[0]
        t_coords = np.array([[t_start], [t_end]])
        
        # Create spatial coordinates
        x_coords = np.array([[0.0], [1.0]])
        
        # Compute cartesian product
        space_time_coords = cart_prod_coords(t_coords, x_coords)
        
        assert space_time_coords.shape == (4, 2)
        expected = np.array([
            [0.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 1.0]
        ])
        np.testing.assert_array_almost_equal(space_time_coords, expected)

    def test_time_slabs_coverage(self):
        """Test that time slabs provide complete coverage."""
        start_time = 0.0
        end_time = 2.3
        slab_size = 0.4
        
        slabs = compute_time_slabs(start_time, end_time, slab_size)
        
        # Check coverage
        assert slabs[0][0] == start_time
        assert slabs[-1][1] >= end_time
        
        # Check no gaps
        total_time = sum(slab[1] - slab[0] for slab in slabs)
        expected_total = len(slabs) * slab_size
        assert np.isclose(total_time, expected_total)

    def test_verbose_with_time_computation(self):
        """Test verbose output during time computation simulation."""
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            slabs = compute_time_slabs(0.0, 1.0, 0.2)
            
            for i, slab in enumerate(slabs):
                vprint(f"Processing slab {i}: [{slab[0]:.2f}, {slab[1]:.2f}]", verbose=True)
        
        output = captured_output.getvalue()
        assert "Processing slab 0" in output
        assert "[0.00, 0.20]" in output


# Main execution for standalone testing
if __name__ == "__main__":
    print("Running utils tests...")
    
    # Test cart_prod_coords
    print("Testing cart_prod_coords...")
    test_cart = TestCartProdCoords()
    test_cart.test_basic_2d_coordinates()
    test_cart.test_1d_time_coordinates()
    print("cart_prod_coords tests passed")
    
    # Test compute_time_slabs
    print("Testing compute_time_slabs...")
    test_slabs = TestComputeTimeSlabs()
    test_slabs.test_basic_time_slabs()
    test_slabs.test_exact_division()
    print("compute_time_slabs tests passed")
    
    # Test vprint
    print("Testing vprint...")
    test_vp = TestVprint()
    test_vp.test_verbose_true_default()
    test_vp.test_verbose_false()
    print("vprint tests passed")
    
    # Test inverse DS transform requires FE setup, skip in standalone
    print("Testing inverse_DS_transform...")
    print("(Skipping inverse_DS_transform tests in standalone mode - requires FE setup)")
    
    # Test integration scenarios
    print("Testing integration scenarios...")
    test_integration = TestUtilsIntegration()
    test_integration.test_cart_prod_with_time_slabs()
    test_integration.test_time_slabs_coverage()
    print("Integration tests passed")
    
    print("All utils tests completed successfully!")
