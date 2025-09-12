"""
Comprehensive benchmarking suite for Zeckendorf-recursive optimization algorithms.

This script evaluates the performance of Fibonacci-constrained optimization algorithms
against standard test functions and compares with traditional optimization methods.
"""

import json
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.algorithms.fibonacci_optimizer import FibonacciConstrainedGradientDescent, GoldenRatioEvolutionaryOptimizer


# Test functions for optimization benchmarking
class BenchmarkFunctions:
    """Collection of standard optimization test functions."""

    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Sphere function: f(x) = Σx_i²"""
        return float(np.sum(x**2))

    @staticmethod
    def sphere_gradient(x: np.ndarray) -> np.ndarray:
        """Gradient of sphere function."""
        return 2 * x

    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²"""
        if len(x) != 2:
            raise ValueError("Rosenbrock function requires 2D input")
        a, b = 1, 100
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

    @staticmethod
    def rosenbrock_gradient(x: np.ndarray) -> np.ndarray:
        """Gradient of Rosenbrock function."""
        if len(x) != 2:
            raise ValueError("Rosenbrock gradient requires 2D input")
        a, b = 1, 100
        dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
        dy = 2 * b * (x[1] - x[0]**2)
        return np.array([dx, dy])

    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin function: f(x) = An + Σ[x_i² - A*cos(2πx_i)]"""
        A = 10
        n = len(x)
        return float(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))

    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Ackley function: multimodal test function"""
        a, b, c = 20, 0.2, 2*np.pi
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.e

    @staticmethod
    def beale(x: np.ndarray) -> float:
        """Beale function: f(x,y) = ..."""
        if len(x) != 2:
            raise ValueError("Beale function requires 2D input")
        x1, x2 = x[0], x[1]
        term1 = (1.5 - x1 + x1*x2)**2
        term2 = (2.25 - x1 + x1*x2**2)**2
        term3 = (2.625 - x1 + x1*x2**3)**2
        return term1 + term2 + term3


class OptimizationBenchmark:
    """Benchmark suite for optimization algorithms."""

    def __init__(self):
        """Initialize benchmark suite."""
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': []
        }

        # Test function configurations
        self.test_functions = {
            'sphere_2d': {
                'function': BenchmarkFunctions.sphere,
                'gradient': BenchmarkFunctions.sphere_gradient,
                'bounds': [(-5, 5), (-5, 5)],
                'optimum': [0, 0],
                'initial': [3, 4],
                'tolerance': 1e-6
            },
            'sphere_5d': {
                'function': BenchmarkFunctions.sphere,
                'gradient': BenchmarkFunctions.sphere_gradient,
                'bounds': [(-3, 3)] * 5,
                'optimum': [0] * 5,
                'initial': [2, -1, 1.5, -2, 0.5],
                'tolerance': 1e-4
            },
            'rosenbrock': {
                'function': BenchmarkFunctions.rosenbrock,
                'gradient': BenchmarkFunctions.rosenbrock_gradient,
                'bounds': [(-2, 2), (-1, 3)],
                'optimum': [1, 1],
                'initial': [0, 0],
                'tolerance': 1e-3
            },
            'rastrigin_2d': {
                'function': BenchmarkFunctions.rastrigin,
                'gradient': None,  # No analytical gradient
                'bounds': [(-5.12, 5.12), (-5.12, 5.12)],
                'optimum': [0, 0],
                'initial': [3, -2],
                'tolerance': 1e-2
            },
            'ackley_2d': {
                'function': BenchmarkFunctions.ackley,
                'gradient': None,  # No analytical gradient
                'bounds': [(-5, 5), (-5, 5)],
                'optimum': [0, 0],
                'initial': [2, -3],
                'tolerance': 1e-2
            }
        }

    def run_fibonacci_gradient_descent(self,
                                     function_name: str,
                                     max_iterations: int = 500) -> Dict:
        """Benchmark Fibonacci-constrained gradient descent."""
        config = self.test_functions[function_name]

        if config['gradient'] is None:
            return {'error': 'No gradient available for this function'}

        optimizer = FibonacciConstrainedGradientDescent(max_param_value=100)

        start_time = time.time()

        try:
            optimal_x, history = optimizer.optimize(
                objective=config['function'],
                gradient=config['gradient'],
                x0=np.array(config['initial']),
                max_iterations=max_iterations,
                tolerance=config['tolerance']
            )

            end_time = time.time()

            # Calculate performance metrics
            final_objective = config['function'](optimal_x)
            initial_objective = config['function'](np.array(config['initial']))
            optimum_distance = np.linalg.norm(
                optimal_x - np.array(config['optimum'])
            )

            return {
                'algorithm': 'FibonacciGradientDescent',
                'function': function_name,
                'success': True,
                'runtime_seconds': end_time - start_time,
                'iterations': len(history),
                'final_objective': float(final_objective),
                'initial_objective': float(initial_objective),
                'improvement': float(initial_objective - final_objective),
                'optimum_distance': float(optimum_distance),
                'final_point': optimal_x.tolist(),
                'convergence_history': history[:50]  # Limit history for storage
            }

        except Exception as e:
            return {
                'algorithm': 'FibonacciGradientDescent',
                'function': function_name,
                'success': False,
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }

    def run_golden_evolutionary(self,
                              function_name: str,
                              population_size: int = 30,
                              max_generations: int = 50) -> Dict:
        """Benchmark golden ratio evolutionary optimizer."""
        config = self.test_functions[function_name]

        optimizer = GoldenRatioEvolutionaryOptimizer(max_param_value=100)

        start_time = time.time()

        try:
            optimal_x, history = optimizer.optimize(
                objective=config['function'],
                bounds=config['bounds'],
                population_size=population_size,
                max_generations=max_generations
            )

            end_time = time.time()

            # Calculate performance metrics
            final_objective = config['function'](optimal_x)
            initial_objective = config['function'](np.array(config['initial']))
            optimum_distance = np.linalg.norm(
                optimal_x - np.array(config['optimum'])
            )

            return {
                'algorithm': 'GoldenRatioEvolutionary',
                'function': function_name,
                'success': True,
                'runtime_seconds': end_time - start_time,
                'generations': len(history),
                'population_size': population_size,
                'final_objective': float(final_objective),
                'initial_objective': float(initial_objective),
                'improvement': float(initial_objective - final_objective),
                'optimum_distance': float(optimum_distance),
                'final_point': optimal_x.tolist(),
                'convergence_history': history[:50]  # Limit history for storage
            }

        except Exception as e:
            return {
                'algorithm': 'GoldenRatioEvolutionary',
                'function': function_name,
                'success': False,
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }

    def run_comprehensive_benchmark(self) -> Dict:
        """Run comprehensive benchmark on all test functions."""
        print("Starting comprehensive optimization benchmark...")
        print("=" * 60)

        all_results = []

        for function_name in self.test_functions.keys():
            print(f"\nBenchmarking function: {function_name}")

            # Test Fibonacci Gradient Descent (if gradient available)
            if self.test_functions[function_name]['gradient'] is not None:
                print("  Running Fibonacci Gradient Descent...")
                fgd_result = self.run_fibonacci_gradient_descent(function_name)
                all_results.append(fgd_result)

                if fgd_result['success']:
                    print(f"    ✓ Converged in {fgd_result['iterations']} iterations")
                    print(f"    ✓ Final objective: {fgd_result['final_objective']:.6f}")
                    print(f"    ✓ Runtime: {fgd_result['runtime_seconds']:.3f}s")
                else:
                    print(f"    ✗ Failed: {fgd_result.get('error', 'Unknown error')}")

            # Test Golden Ratio Evolutionary
            print("  Running Golden Ratio Evolutionary...")
            gre_result = self.run_golden_evolutionary(function_name)
            all_results.append(gre_result)

            if gre_result['success']:
                print(f"    ✓ Converged in {gre_result['generations']} generations")
                print(f"    ✓ Final objective: {gre_result['final_objective']:.6f}")
                print(f"    ✓ Runtime: {gre_result['runtime_seconds']:.3f}s")
            else:
                print(f"    ✗ Failed: {gre_result.get('error', 'Unknown error')}")

        # Compile summary statistics
        successful_runs = [r for r in all_results if r['success']]

        summary = {
            'total_runs': len(all_results),
            'successful_runs': len(successful_runs),
            'success_rate': len(successful_runs) / len(all_results) if all_results else 0,
            'average_runtime': np.mean([r['runtime_seconds'] for r in successful_runs]) if successful_runs else 0,
            'results': all_results
        }

        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Total runs: {summary['total_runs']}")
        print(f"Successful runs: {summary['successful_runs']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Average runtime: {summary['average_runtime']:.3f}s")

        return summary

    def save_results(self, results: Dict, filename: Optional[str] = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filename}")


def main():
    """Run the benchmark suite."""
    benchmark = OptimizationBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    benchmark.save_results(results)


if __name__ == "__main__":
    main()
