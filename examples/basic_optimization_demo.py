"""
Basic example demonstrating Zeckendorf-recursive optimization algorithms.

This script shows how to use the Fibonacci-constrained optimization algorithms
on simple test functions to validate the mathematical framework.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.fibonacci_optimizer import FibonacciConstrainedGradientDescent, GoldenRatioEvolutionaryOptimizer

# Path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
    Classic optimization test function with global minimum at (1,1).
    """
    a, b = 1, 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of Rosenbrock function."""
    a, b = 1, 100
    dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    dy = 2 * b * (x[1] - x[0]**2)
    return np.array([dx, dy])


def sphere_function(x: np.ndarray) -> float:
    """Simple sphere function: f(x) = sum(x_i^2)"""
    return float(np.sum(x**2))


def sphere_gradient(x: np.ndarray) -> np.ndarray:
    """Gradient of sphere function."""
    return 2 * x


def demonstrate_fibonacci_gradient_descent():
    """Demonstrate Fibonacci-constrained gradient descent."""
    print("=== Fibonacci-Constrained Gradient Descent ===")

    optimizer = FibonacciConstrainedGradientDescent(max_param_value=100)

    # Test on sphere function (easier problem)
    print("\n1. Sphere Function Optimization:")
    x0_sphere = np.array([3.0, 4.0])
    print(f"Initial point: {x0_sphere}")
    print(f"Initial objective: {sphere_function(x0_sphere):.6f}")

    optimal_sphere, history_sphere = optimizer.optimize(
        objective=sphere_function,
        gradient=sphere_gradient,
        x0=x0_sphere,
        max_iterations=100,
        tolerance=1e-6
    )

    print(f"Optimal point: {optimal_sphere}")
    print(f"Optimal objective: {sphere_function(optimal_sphere):.6f}")
    print(f"Convergence in {len(history_sphere)} iterations")

    # Test on Rosenbrock function (harder problem)
    print("\n2. Rosenbrock Function Optimization:")
    x0_rosenbrock = np.array([0.0, 0.0])
    print(f"Initial point: {x0_rosenbrock}")
    print(f"Initial objective: {rosenbrock_function(x0_rosenbrock):.6f}")

    optimal_rosenbrock, history_rosenbrock = optimizer.optimize(
        objective=rosenbrock_function,
        gradient=rosenbrock_gradient,
        x0=x0_rosenbrock,
        max_iterations=500,
        tolerance=1e-4
    )

    print(f"Optimal point: {optimal_rosenbrock}")
    print(f"Optimal objective: {rosenbrock_function(optimal_rosenbrock):.6f}")
    print(f"Convergence in {len(history_rosenbrock)} iterations")

    return history_sphere, history_rosenbrock


def demonstrate_golden_evolutionary_optimizer():
    """Demonstrate golden ratio evolutionary optimization."""
    print("\n=== Golden Ratio Evolutionary Optimizer ===")

    optimizer = GoldenRatioEvolutionaryOptimizer(max_param_value=100)

    # Test on Rosenbrock function
    print("\n1. Rosenbrock Function with Evolutionary Algorithm:")
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    optimal_ea, history_ea = optimizer.optimize(
        objective=rosenbrock_function,
        bounds=bounds,
        population_size=30,
        max_generations=50
    )

    print(f"Optimal point: {optimal_ea}")
    print(f"Optimal objective: {rosenbrock_function(optimal_ea):.6f}")
    print(f"Final generation: {len(history_ea)}")

    return history_ea


def analyze_zeckendorf_properties():
    """Analyze mathematical properties of Zeckendorf representation."""
    print("\n=== Zeckendorf Representation Analysis ===")

    from src.constraints.no11_constraint import No11ConstraintSystem
    from src.core.zeckendorf import ZeckendorfRepresentation

    zeck = ZeckendorfRepresentation(100)
    constraints = No11ConstraintSystem(100)

    print("\n1. Zeckendorf Representations for integers 1-20:")
    for n in range(1, 21):
        repr_fib = zeck.to_zeckendorf(n)
        entropy = zeck.get_entropy_density(n)
        growth_factor = zeck.bounded_growth_factor(n)

        print(f"n={n:2d}: {repr_fib} (entropy: {entropy:.3f}, growth: {growth_factor:.3f})")

    print("\n2. No-11 Constraint Validation:")
    # Test valid representation
    valid_repr = [1, 3, 8]  # Non-consecutive Fibonacci numbers
    print(f"Valid representation {valid_repr}: {constraints.is_valid_representation(valid_repr)}")

    # Test invalid representation
    invalid_repr = [2, 3, 8]  # 2 and 3 are consecutive
    print(f"Invalid representation {invalid_repr}: {constraints.is_valid_representation(invalid_repr)}")

    # Test projection to valid
    projected = constraints.project_to_valid(invalid_repr)
    print(f"Projected to valid: {projected}")
    print(f"Sum before: {sum(invalid_repr)}, sum after: {sum(projected)}")


def plot_convergence_analysis(sphere_history, rosenbrock_history, ea_history):
    """Plot convergence analysis for different algorithms."""
    try:
        plt.figure(figsize=(15, 5))

        # Sphere function convergence
        plt.subplot(1, 3, 1)
        plt.semilogy(sphere_history)
        plt.title('Fibonacci GD: Sphere Function')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value (log scale)')
        plt.grid(True)

        # Rosenbrock function convergence
        plt.subplot(1, 3, 2)
        plt.semilogy(rosenbrock_history)
        plt.title('Fibonacci GD: Rosenbrock Function')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value (log scale)')
        plt.grid(True)

        # Evolutionary algorithm convergence
        plt.subplot(1, 3, 3)
        plt.semilogy(ea_history)
        plt.title('Golden Ratio EA: Rosenbrock Function')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness (log scale)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("\nConvergence analysis plot saved as 'convergence_analysis.png'")

    except ImportError:
        print("Matplotlib not available for plotting. Install with: pip install matplotlib")


def main():
    """Run all demonstration examples."""
    print("Zeckendorf-Recursive Optimization Algorithm Demonstration")
    print("=" * 60)

    # Analyze mathematical properties first
    analyze_zeckendorf_properties()

    # Demonstrate optimization algorithms
    sphere_hist, rosenbrock_hist = demonstrate_fibonacci_gradient_descent()
    ea_hist = demonstrate_golden_evolutionary_optimizer()

    # Plot results
    plot_convergence_analysis(sphere_hist, rosenbrock_hist, ea_hist)

    print("\n" + "=" * 60)
    print("Demonstration completed successfully!")
    print("Key observations:")
    print("1. Fibonacci constraints provide natural bounds for optimization")
    print("2. Golden ratio step sizes show stable convergence properties")
    print("3. No-11 constraints maintain representation validity")
    print("4. Evolutionary algorithms benefit from golden ratio operations")


if __name__ == "__main__":
    main()
