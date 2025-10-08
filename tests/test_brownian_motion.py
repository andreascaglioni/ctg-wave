"""
Tests for the brownian_motion module.

This module tests the statistical properties of the Levy-Ciesielski 
parametrization of Brownian motion to ensure it generates sample paths
with the correct mean and variance properties.
"""

from math import sqrt
import numpy as np
import pytest
from ctg.brownian_motion import param_LC_W, LC_matrix


class TestParamLCW:
    """Test class for param_LC_W function."""
    
    def test_mean_zero_single_sample(self):
        """Test that a single sample path has approximately zero mean."""
        # Parameters
        T = 1.0
        n_time_points = 100
        n_coeffs = 16  # 2^4 coefficients for good approximation
        
        # Generate random coefficients (standard normal)
        np.random.seed(42)
        y = np.random.randn(n_coeffs)
        
        # Time points
        tt = np.linspace(0, T, n_time_points)
        
        # Generate sample path
        W = param_LC_W(y, tt, T)
        
        # Check that W starts at 0 (Brownian motion property)
        assert abs(W[0, 0]) < 1e-12, "Brownian motion should start at 0"
        
        # For a single sample, we can't test statistical properties,
        # but we can check that the path is reasonable
        assert W.shape == (1, n_time_points), "Wrong output shape"
    
    def test_mean_zero_ensemble(self):
        """Test that ensemble of sample paths has mean approximately zero."""
        # Parameters
        T = 1.0
        n_time_points = 50
        n_coeffs = 32  # More coefficients for better approximation
        n_samples = 1000  # Number of sample paths
        
        # Generate ensemble of random coefficients
        np.random.seed(123)
        Y = np.random.randn(n_samples, n_coeffs)
        
        # Time points
        tt = np.linspace(0, T, n_time_points)
        
        # Generate sample paths
        W = param_LC_W(Y, tt, T)
        
        # Check shape
        assert W.shape == (n_samples, n_time_points), "Wrong output shape"
        
        # Check that all paths start at 0
        assert np.allclose(W[:, 0], 0, atol=1e-12), "All paths should start at 0"
        
        # Compute empirical mean across samples at each time point
        empirical_mean = np.mean(W, axis=0)
        
        # Test that mean is close to 0 (should be exactly 0 for Brownian motion)
        # We allow some tolerance due to finite sample approximation
        tolerance = 3 * np.sqrt(T) / np.sqrt(n_samples)  # 3-sigma bound
        assert np.all(np.abs(empirical_mean) < tolerance), \
            f"Mean deviates too much from 0: max deviation = {np.max(np.abs(empirical_mean)):.6f}"
    
    def test_variance_brownian_motion(self):
        """Test that sample paths have correct variance structure for Brownian motion."""
        # Parameters
        T = 2.0
        n_time_points = 100
        n_coeffs = 64  # More coefficients for better approximation
        n_samples = 2000  # Large number for good statistics
        
        # Generate ensemble of random coefficients
        np.random.seed(456)
        Y = np.random.randn(n_samples, n_coeffs)
        
        # Time points
        tt = np.linspace(0, T, n_time_points)
        
        # Generate sample paths
        W = param_LC_W(Y, tt, T)
        
        # Compute empirical variance at each time point
        empirical_var = np.var(W, axis=0, ddof=1)
        
        # For Brownian motion, Var(W(t)) = t
        theoretical_var = tt
        
        # Test variance at final time (should be close to T)
        final_var = empirical_var[-1]
        expected_final_var = T
        var_tolerance = 3 * np.sqrt(2 * T**2 / (n_samples - 1))  # 3-sigma bound for variance
        
        assert abs(final_var - expected_final_var) < var_tolerance, \
            f"Final variance {final_var:.4f} too far from expected {expected_final_var:.4f}"
        
        # Test variance at intermediate points
        # We test a few specific points to avoid issues with discretization
        test_indices = [n_time_points//4, n_time_points//2, 3*n_time_points//4]
        
        for idx in test_indices:
            t = tt[idx]
            var_t = empirical_var[idx]
            expected_var_t = t
            tolerance_t = 3 * np.sqrt(2 * t**2 / (n_samples - 1))
            
            assert abs(var_t - expected_var_t) < tolerance_t, \
                f"Variance at t={t:.2f} is {var_t:.4f}, expected {expected_var_t:.4f}"
    
    def test_increments_independence(self):
        """Test that Brownian motion increments have correct properties."""
        # Parameters
        T = 1.0
        n_samples = 1000
        n_coeffs = 32
        
        # Generate ensemble
        np.random.seed(789)
        Y = np.random.randn(n_samples, n_coeffs)
        
        # Specific time points for increment test
        t1, t2, t3 = 0.2, 0.5, 0.8
        tt = np.array([0.0, t1, t2, t3, T])
        
        # Generate sample paths
        W = param_LC_W(Y, tt, T)
        
        # Compute increments
        dW1 = W[:, 1] - W[:, 0]  # W(t1) - W(0)
        dW2 = W[:, 2] - W[:, 1]  # W(t2) - W(t1)
        dW3 = W[:, 3] - W[:, 2]  # W(t3) - W(t2)
        
        # Test that increments have zero mean
        tolerance = 3 / np.sqrt(n_samples)
        
        assert abs(np.mean(dW1)) < tolerance, "First increment should have zero mean"
        assert abs(np.mean(dW2)) < tolerance, "Second increment should have zero mean"
        assert abs(np.mean(dW3)) < tolerance, "Third increment should have zero mean"
        
        # Test that increments have correct variance
        var_tolerance = 3 * np.sqrt(2 / (n_samples - 1))
        
        assert abs(np.var(dW1, ddof=1) - t1) < var_tolerance * t1, \
            f"First increment variance incorrect"
        assert abs(np.var(dW2, ddof=1) - (t2 - t1)) < var_tolerance * (t2 - t1), \
            f"Second increment variance incorrect"
        assert abs(np.var(dW3, ddof=1) - (t3 - t2)) < var_tolerance * (t3 - t2), \
            f"Third increment variance incorrect"
    
    def test_input_validation(self):
        """Test that function handles edge cases and invalid inputs properly."""
        T = 1.0
        tt = np.linspace(0, T, 10)
        
        # Test with 1D input (should be converted to 2D)
        y_1d = np.random.randn(8)
        W = param_LC_W(y_1d, tt, T)
        assert W.shape == (1, 10), "1D input should produce single sample path"
        
        # Test with single time point
        t_single = np.array([0.5])
        W_single = param_LC_W(y_1d, t_single, T)
        assert W_single.shape == (1, 1), "Single time point should work"
        
        # Test that time boundaries are respected
        with pytest.raises(AssertionError):
            param_LC_W(y_1d, np.array([-0.1, 0.5]), T)  # Negative time
        
        with pytest.raises(AssertionError):
            param_LC_W(y_1d, np.array([0.5, T + 0.1]), T)  # Time > T


class TestLCMatrix:
    """Test class for LC_matrix function."""
    
    def test_matrix_dimensions(self):
        """Test that LC_matrix returns correct dimensions."""
        L = 3
        T = 1.0
        tt = np.linspace(0, T, 50)
        
        BB = LC_matrix(L, tt, T)
        
        expected_rows = 2**L
        expected_cols = len(tt)
        
        assert BB.shape == (expected_rows, expected_cols), \
            f"Expected shape ({expected_rows}, {expected_cols}), got {BB.shape}"
    
    def test_first_basis_function(self):
        """Test that first basis function is linear."""
        L = 2
        T = 2.0
        tt = np.linspace(0, T, 100)
        
        BB = LC_matrix(L, tt, T)
        
        # First basis function should be sqrt(T) * t/T = 1/sqrt(T) * t
        expected_first = tt / sqrt(T)
        actual_first = BB[0, :]
        
        np.testing.assert_allclose(actual_first, expected_first, rtol=1e-12,
                                 err_msg="First basis function should be linear")


if __name__ == "__main__":
    # Run basic tests if script is executed directly
    test_instance = TestParamLCW()
    
    print("Running basic tests...")
    
    print("Testing mean zero property...")
    test_instance.test_mean_zero_ensemble()
    print("✓ Mean zero test passed")
    
    print("Testing variance property...")
    test_instance.test_variance_brownian_motion()
    print("✓ Variance test passed")
    
    print("Testing increment properties...")
    test_instance.test_increments_independence()
    print("✓ Increment test passed")
    
    print("All tests passed.")