"""
Comprehensive tests for Zeckendorf-recursive optimization algorithms.

This test suite validates the mathematical foundations and algorithmic implementations
of the Fibonacci-constrained optimization framework.
"""

import numpy as np

from src.algorithms.fibonacci_optimizer import FibonacciConstrainedGradientDescent
from src.constraints.no11_constraint import No11ConstraintSystem
from src.core.golden_ratio import GoldenRatioOptimizer
from src.core.zeckendorf import ZeckendorfRepresentation


class TestZeckendorfRepresentation:
    """Test Zeckendorf representation mathematical operations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.zeck = ZeckendorfRepresentation(max_n=100)

    def test_fibonacci_sequence_generation(self):
        """Test Fibonacci sequence is generated correctly."""
        fib_seq = self.zeck.fibonacci_sequence

        # Check first few Fibonacci numbers
        assert fib_seq[0] == 1  # F_2
        assert fib_seq[1] == 2  # F_3
        assert fib_seq[2] == 3  # F_4
        assert fib_seq[3] == 5  # F_5
        assert fib_seq[4] == 8  # F_6

        # Check Fibonacci recurrence relation
        for i in range(2, len(fib_seq)):
            assert fib_seq[i] == fib_seq[i-1] + fib_seq[i-2]

    def test_zeckendorf_representation_validity(self):
        """Test Zeckendorf representation produces valid results."""
        test_cases = [
            (1, [1]),
            (2, [2]),
            (3, [3]),
            (4, [1, 3]),
            (5, [5]),
            (6, [1, 5]),
            (7, [2, 5]),
            (8, [8]),
            (9, [1, 8]),
            (10, [2, 8])
        ]

        for n, expected in test_cases:
            result = self.zeck.to_zeckendorf(n)
            assert result == expected, f"Failed for n={n}: got {result}, expected {expected}"

            # Verify sum equals original number
            assert sum(result) == n

            # Verify no consecutive Fibonacci numbers
            assert self.zeck._validate_no_consecutive(result)

    def test_zeckendorf_roundtrip(self):
        """Test conversion to and from Zeckendorf representation."""
        for n in range(1, 51):
            zeck_repr = self.zeck.to_zeckendorf(n)
            reconstructed = self.zeck.from_zeckendorf(zeck_repr)
            assert reconstructed == n, f"Roundtrip failed for n={n}"

    def test_entropy_density_properties(self):
        """Test entropy density calculation."""
        # Entropy density should be positive
        for n in range(1, 21):
            entropy = self.zeck.get_entropy_density(n)
            assert entropy > 0, f"Entropy should be positive for n={n}"

    def test_bounded_growth_factor(self):
        """Test bounded growth factor calculation."""
        # Growth factor should be positive and bounded
        for n in range(1, 21):
            growth_factor = self.zeck.bounded_growth_factor(n)
            assert growth_factor > 0, f"Growth factor should be positive for n={n}"
            assert growth_factor < 10, f"Growth factor should be bounded for n={n}"


class TestGoldenRatioOptimizer:
    """Test golden ratio optimization algorithms."""

    def setup_method(self):
        """Setup test fixtures."""
        self.golden = GoldenRatioOptimizer()

    def test_golden_ratio_constants(self):
        """Test golden ratio constants are correct."""
        phi = self.golden.phi

        # Test fundamental golden ratio property: φ² = φ + 1
        assert abs(phi**2 - (phi + 1)) < 1e-10

        # Test inverse golden ratio
        assert abs(self.golden.inv_phi - 1/phi) < 1e-10

        # Test approximate value
        assert abs(phi - 1.618033988749) < 1e-10

    def test_golden_section_search(self):
        """Test golden section search on simple functions."""
        # Test on parabola: f(x) = (x-2)^2, minimum at x=2
        def parabola(x):
            return (x - 2)**2

        optimal_x, optimal_f = self.golden.golden_section_search(
            parabola, 0, 4, tolerance=1e-6
        )

        assert abs(optimal_x - 2.0) < 1e-5, f"Expected x≈2, got {optimal_x}"
        assert abs(optimal_f) < 1e-10, f"Expected f≈0, got {optimal_f}"

    def test_fibonacci_step_sizes(self):
        """Test Fibonacci-based step size generation."""
        step_sizes = self.golden.fibonacci_step_sizes(10)

        assert len(step_sizes) == 10
        assert all(0 < step < 1 for step in step_sizes)

        # Step sizes should generally decrease (with some oscillation)
        assert step_sizes[-1] < step_sizes[0]

    def test_phi_spiral_search(self):
        """Test golden spiral search pattern."""
        # Test on simple 2D parabola: f(x,y) = x² + y²
        def paraboloid(point):
            x, y = point
            return x**2 + y**2

        optimal_point, optimal_value = self.golden.phi_spiral_search(
            paraboloid, center=(0, 0), radius=2.0, n_points=13, iterations=3
        )

        # Should find point near origin
        x_opt, y_opt = optimal_point
        assert abs(x_opt) < 0.5, f"Expected x≈0, got {x_opt}"
        assert abs(y_opt) < 0.5, f"Expected y≈0, got {y_opt}"
        assert optimal_value < 0.25, f"Expected small objective value, got {optimal_value}"


class TestNo11ConstraintSystem:
    """Test No-11 constraint system."""

    def setup_method(self):
        """Setup test fixtures."""
        self.constraints = No11ConstraintSystem(max_n=100)

    def test_valid_representation_detection(self):
        """Test detection of valid No-11 representations."""
        # Valid representations (no consecutive Fibonacci numbers)
        valid_cases = [
            [1],           # Single number
            [1, 3],        # Non-consecutive
            [1, 5, 13],    # All non-consecutive
            [2, 8, 21]     # Non-consecutive
        ]

        for case in valid_cases:
            assert self.constraints.is_valid_representation(case), \
                f"Should be valid: {case}"

    def test_invalid_representation_detection(self):
        """Test detection of invalid No-11 representations."""
        # Invalid representations (consecutive Fibonacci numbers)
        invalid_cases = [
            [1, 2],        # 1 and 2 are consecutive
            [2, 3],        # 2 and 3 are consecutive
            [3, 5],        # 3 and 5 are consecutive
            [5, 8, 13]     # 5 and 8 are consecutive
        ]

        for case in invalid_cases:
            assert not self.constraints.is_valid_representation(case), \
                f"Should be invalid: {case}"

    def test_projection_to_valid(self):
        """Test projection of invalid representations to valid ones."""
        # Test F_n + F_{n+1} = F_{n+2} combination rule
        invalid_repr = [2, 3]  # Should become [5]
        projected = self.constraints.project_to_valid(invalid_repr)

        assert projected == [5], f"Expected [5], got {projected}"
        assert sum(projected) == sum(invalid_repr), "Sum should be preserved"
        assert self.constraints.is_valid_representation(projected), "Result should be valid"

    def test_constraint_penalty(self):
        """Test constraint penalty calculation."""
        # Valid representation should have zero penalty
        valid_repr = [1, 3, 8]
        assert self.constraints.constraint_penalty(valid_repr) == 0.0

        # Invalid representation should have positive penalty
        invalid_repr = [2, 3, 8]
        penalty = self.constraints.constraint_penalty(invalid_repr)
        assert penalty > 0, f"Expected positive penalty, got {penalty}"

    def test_entropy_with_constraints(self):
        """Test entropy calculation with constraint considerations."""
        # Valid representation should have finite entropy
        valid_repr = [1, 3, 8]
        entropy = self.constraints.entropy_with_constraints(valid_repr)
        assert entropy > 0 and np.isfinite(entropy)

        # Invalid representation should have -infinity entropy
        invalid_repr = [2, 3]
        entropy_invalid = self.constraints.entropy_with_constraints(invalid_repr)
        assert entropy_invalid == -float('inf')


class TestFibonacciConstrainedOptimizer:
    """Test Fibonacci-constrained optimization algorithms."""

    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = FibonacciConstrainedGradientDescent(max_param_value=50)

    def test_parameter_encoding_decoding(self):
        """Test parameter encoding/decoding roundtrip."""
        test_params = np.array([0.5, 1.2, 0.8])

        # Encode to Zeckendorf
        encoded = self.optimizer.encode_parameters(test_params)

        # Should be list of Fibonacci representations
        assert isinstance(encoded, list)
        assert len(encoded) == len(test_params)

        # Each should be valid Zeckendorf representation
        for fib_list in encoded:
            assert self.optimizer.constraints.is_valid_representation(fib_list)

        # Decode back
        decoded = self.optimizer.decode_parameters(encoded)

        # Should have same dimensions
        assert decoded.shape == test_params.shape
        assert len(decoded) == len(test_params)

    def test_constraint_penalty_calculation(self):
        """Test constraint penalty for encoded parameters."""
        # Valid encoded parameters
        valid_encoded = [[1, 3], [5], [2, 8]]
        penalty = self.optimizer.constraint_penalty(valid_encoded)
        assert penalty == 0.0, "Valid representations should have zero penalty"

        # Invalid encoded parameters
        invalid_encoded = [[2, 3], [5], [8, 13]]  # First has consecutive terms
        penalty_invalid = self.optimizer.constraint_penalty(invalid_encoded)
        assert penalty_invalid > 0, "Invalid representations should have positive penalty"

    def test_simple_optimization(self):
        """Test optimization on simple quadratic function."""
        # Simple quadratic: f(x) = sum((x - 1)²)
        def objective(x):
            return np.sum((x - 1)**2)

        def gradient(x):
            return 2 * (x - 1)

        # Start from point away from optimum
        x0 = np.array([0.0, 2.0])

        optimal_x, history = self.optimizer.optimize(
            objective=objective,
            gradient=gradient,
            x0=x0,
            max_iterations=50,
            tolerance=1e-4
        )

        # Should converge toward [1, 1] (within encoding limitations)
        assert len(optimal_x) == 2, "Should preserve dimensionality"
        assert len(history) > 0, "Should record convergence history"

        # Final objective should be better than initial
        final_obj = objective(optimal_x)
        initial_obj = objective(x0)
        assert final_obj < initial_obj, "Should improve objective function"


def test_integration_example():
    """Integration test using all components together."""
    # Simple 2D optimization problem
    def rosenbrock_2d(x):
        """Simplified 2D Rosenbrock function."""
        return (1 - x[0])**2 + 10 * (x[1] - x[0]**2)**2

    def rosenbrock_grad(x):
        """Gradient of simplified Rosenbrock."""
        dx0 = -2 * (1 - x[0]) - 40 * x[0] * (x[1] - x[0]**2)
        dx1 = 20 * (x[1] - x[0]**2)
        return np.array([dx0, dx1])

    # Test with Fibonacci-constrained optimizer
    optimizer = FibonacciConstrainedGradientDescent(max_param_value=100)

    x0 = np.array([0.5, 0.5])
    optimal_x, history = optimizer.optimize(
        objective=rosenbrock_2d,
        gradient=rosenbrock_grad,
        x0=x0,
        max_iterations=100,
        tolerance=1e-4
    )

    # Verify optimization made progress
    initial_obj = rosenbrock_2d(x0)
    final_obj = rosenbrock_2d(optimal_x)

    assert final_obj < initial_obj, "Optimization should improve objective"
    assert len(history) > 0, "Should record convergence history"

    # Check that constraints were respected throughout
    final_encoded = optimizer.encode_parameters(optimal_x)
    for fib_list in final_encoded:
        assert optimizer.constraints.is_valid_representation(fib_list), \
            "Final parameters should satisfy constraints"


if __name__ == "__main__":
    # Run tests directly for development

    print("Running Zeckendorf-Recursive Optimization Tests")
    print("=" * 50)

    # Run test classes
    test_classes = [
        TestZeckendorfRepresentation,
        TestGoldenRatioOptimizer,
        TestNo11ConstraintSystem,
        TestFibonacciConstrainedOptimizer
    ]

    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}...")
        instance = test_class()
        instance.setup_method()

        # Run test methods
        methods = [method for method in dir(instance) if method.startswith('test_')]
        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")

    # Run integration test
    print("\nTesting integration...")
    try:
        test_integration_example()
        print("  ✓ test_integration_example")
    except Exception as e:
        print(f"  ✗ test_integration_example: {e}")

    print("\nTest suite completed!")
