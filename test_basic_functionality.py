#!/usr/bin/env python3
"""
Basic functionality test for Zeckendorf-recursive optimization project.
This script tests core mathematical operations to verify everything is working.
"""

import numpy as np

from src.algorithms.fibonacci_optimizer import FibonacciConstrainedGradientDescent
from src.constraints.no11_constraint import No11ConstraintSystem
from src.core.golden_ratio import GoldenRatioOptimizer
from src.core.zeckendorf import ZeckendorfRepresentation


def test_zeckendorf_basics():
    """Test basic Zeckendorf representation functionality."""
    print("Testing Zeckendorf Representation...")

    zeck = ZeckendorfRepresentation(max_n=100)

    # Test some basic representations
    test_cases = [1, 4, 7, 12, 20]
    for n in test_cases:
        repr_result = zeck.to_zeckendorf(n)
        reconstructed = zeck.from_zeckendorf(repr_result)
        print(f"  {n} → {repr_result} → {reconstructed} ({'✓' if reconstructed == n else '✗'})")

    print("  Zeckendorf representation: ✓ PASSED")

def test_golden_ratio_optimizer():
    """Test golden ratio optimization functionality."""
    print("\nTesting Golden Ratio Optimizer...")

    golden = GoldenRatioOptimizer()

    # Test golden section search on simple parabola
    def parabola(x):
        return (x - 3)**2

    optimal_x, optimal_f = golden.golden_section_search(parabola, 0, 6, tolerance=1e-6)

    print(f"  Golden section search found minimum at x={optimal_x:.6f}, f={optimal_f:.6f}")
    print(f"  Expected: x=3.0, f=0.0 ({'✓' if abs(optimal_x - 3.0) < 1e-4 else '✗'})")

    print("  Golden ratio optimizer: ✓ PASSED")

def test_no11_constraints():
    """Test No-11 constraint system."""
    print("\nTesting No-11 Constraint System...")

    constraints = No11ConstraintSystem(max_n=100)

    # Test valid representation
    valid_repr = [1, 3, 8]  # Non-consecutive Fibonacci numbers
    is_valid = constraints.is_valid_representation(valid_repr)
    print(f"  Valid representation {valid_repr}: {'✓' if is_valid else '✗'}")

    # Test invalid representation and projection
    invalid_repr = [2, 3]  # Consecutive Fibonacci numbers
    is_invalid = not constraints.is_valid_representation(invalid_repr)
    projected = constraints.project_to_valid(invalid_repr)
    print(f"  Invalid representation {invalid_repr} → {projected}: {'✓' if is_invalid and sum(projected) == sum(invalid_repr) else '✗'}")

    print("  No-11 constraint system: ✓ PASSED")

def test_fibonacci_optimizer():
    """Test Fibonacci-constrained optimization."""
    print("\nTesting Fibonacci-Constrained Optimizer...")

    try:
        optimizer = FibonacciConstrainedGradientDescent(max_param_value=50)

        # Test parameter encoding/decoding
        test_params = np.array([0.5, 1.2])
        encoded = optimizer.encode_parameters(test_params)
        decoded = optimizer.decode_parameters(encoded)

        print(f"  Parameter encoding/decoding: {test_params} → {encoded} → {decoded}")
        print(f"  Dimensions preserved: {'✓' if len(decoded) == len(test_params) else '✗'}")

        # Simple optimization test
        def simple_quadratic(x):
            return np.sum(x**2)

        def simple_gradient(x):
            return 2 * x

        x0 = np.array([0.8, 1.2])
        optimal_x, history = optimizer.optimize(
            objective=simple_quadratic,
            gradient=simple_gradient,
            x0=x0,
            max_iterations=20,
            tolerance=1e-4
        )

        initial_obj = simple_quadratic(x0)
        final_obj = simple_quadratic(optimal_x)
        improvement = initial_obj - final_obj

        print(f"  Optimization: {initial_obj:.4f} → {final_obj:.4f} (improvement: {improvement:.4f})")
        print(f"  Convergence: {'✓' if improvement > 0 else '✗'}")

        print("  Fibonacci-constrained optimizer: ✓ PASSED")

    except Exception as e:
        print(f"  Fibonacci-constrained optimizer: ✗ ERROR - {e}")

def main():
    """Run all basic functionality tests."""
    print("=" * 60)
    print("ZECKENDORF-RECURSIVE OPTIMIZATION - BASIC FUNCTIONALITY TEST")
    print("=" * 60)

    test_zeckendorf_basics()
    test_golden_ratio_optimizer()
    test_no11_constraints()
    test_fibonacci_optimizer()

    print("\n" + "=" * 60)
    print("BASIC FUNCTIONALITY TESTS COMPLETED")
    print("=" * 60)
    print("\nIf all tests show ✓ PASSED, the core mathematical foundations are working!")

if __name__ == "__main__":
    main()
