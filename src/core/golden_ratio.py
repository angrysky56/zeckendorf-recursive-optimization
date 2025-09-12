"""
Golden Ratio Optimization Utilities

This module implements mathematical operations and optimization algorithms based on
the golden ratio (φ = (1+√5)/2), which appears naturally in Fibonacci sequences
and provides optimal convergence properties for certain optimization problems.

Mathematical Foundation:
- Golden ratio φ ≈ 1.618033988749...
- φ² = φ + 1 (fundamental property)
- Optimal convergence rate in golden section search
- Natural connection to Fibonacci sequences: F_n ≈ φⁿ/√5
"""

import math
from typing import Callable, List, Optional, Tuple

import numpy as np


class GoldenRatioOptimizer:
    """
    Implements optimization algorithms based on golden ratio principles.

    The golden ratio provides provably optimal convergence rates for unimodal
    optimization problems and appears naturally in Fibonacci-based algorithms.
    """

    def __init__(self):
        """Initialize golden ratio constants."""
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        self.inv_phi = 1 / self.phi         # 1/φ ≈ 0.618
        self.sqrt5 = math.sqrt(5)

    def golden_section_search(self,
                            f: Callable[[float], float],
                            a: float,
                            b: float,
                            tolerance: float = 1e-6,
                            max_iterations: int = 100) -> Tuple[float, float]:
        """
        Golden section search for unimodal function optimization.

        Provably optimal algorithm for finding minimum of unimodal function
        with golden ratio convergence rate.

        Args:
            f: Unimodal objective function to minimize
            a: Left boundary of search interval
            b: Right boundary of search interval
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations

        Returns:
            Tuple of (optimal_x, optimal_f_value)
        """
        # Initialize golden section points
        c = a + (b - a) * (1 - self.inv_phi)  # Left golden section
        d = a + (b - a) * self.inv_phi         # Right golden section

        fc = f(c)
        fd = f(d)

        for _ in range(max_iterations):
            if abs(b - a) < tolerance:
                break

            if fc < fd:
                # Minimum in left section
                b = d
                d = c
                fd = fc
                c = a + (b - a) * (1 - self.inv_phi)
                fc = f(c)
            else:
                # Minimum in right section
                a = c
                c = d
                fc = fd
                d = a + (b - a) * self.inv_phi
                fd = f(d)

        optimal_x = (a + b) / 2
        return optimal_x, f(optimal_x)

    def fibonacci_step_sizes(self, n_steps: int) -> List[float]:
        """
        Generate Fibonacci-based step sizes for optimization.

        Creates step size schedule based on Fibonacci ratios,
        providing natural decay with golden ratio convergence.

        Args:
            n_steps: Number of step sizes to generate

        Returns:
            List of step sizes with Fibonacci-based decay
        """
        fib_ratios = []
        f1, f2 = 1, 1

        for i in range(n_steps):
            if i <= 1:
                ratio = 1.0 / (i + 1)
            else:
                f1, f2 = f2, f1 + f2
                ratio = f1 / f2  # Fibonacci ratio approaches 1/φ

            fib_ratios.append(ratio)

        return fib_ratios

    def golden_gradient_descent(self,
                              gradient_fn: Callable[[np.ndarray], np.ndarray],
                              x0: np.ndarray,
                              max_iterations: int = 1000,
                              tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
        """
        Gradient descent with golden ratio step size adaptation.

        Uses golden ratio principles to adapt step sizes during optimization,
        combining gradient information with Fibonacci-based convergence.

        Args:
            gradient_fn: Function computing gradient at point x
            x0: Initial point
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Tuple of (optimal_x, convergence_history)
        """
        x = x0.copy()
        history = []

        # Initial step size based on golden ratio
        step_size = 1.0 / self.phi

        for iteration in range(max_iterations):
            grad = gradient_fn(x)
            grad_norm = np.linalg.norm(grad)

            if grad_norm < tolerance:
                break

            # Golden ratio step size adaptation
            if iteration > 0:
                # Adapt step size using Fibonacci sequence ratio
                fib_factor = self._fibonacci_ratio(iteration)
                step_size *= fib_factor

            # Update with golden ratio step
            x_new = x - step_size * grad
            x = x_new

            history.append(grad_norm)

        return x, history

    def _fibonacci_ratio(self, n: int) -> float:
        """
        Calculate nth Fibonacci ratio for step size adaptation.

        Args:
            n: Iteration number

        Returns:
            Fibonacci-based adaptation factor
        """
        if n <= 1:
            return 1.0

        # Use iterative calculation to avoid overflow for large n
        if n > 50:  # For large n, ratio approaches 1/φ
            return self.inv_phi
        
        # For moderate n, use iterative Fibonacci calculation
        fib_prev, fib_curr = 1, 1
        for _ in range(2, n + 1):
            fib_prev, fib_curr = fib_curr, fib_prev + fib_curr
            
        # Calculate F_(n-1) 
        if n == 2:
            fib_prev_step = 1
        else:
            fib_prev_step = fib_prev

        if fib_curr == 0:
            return 1.0

        ratio = fib_prev_step / fib_curr

        # Ensure ratio is in reasonable range for step size
        return max(0.1, min(1.0, ratio))

    def phi_spiral_search(self,
                         f: Callable[[Tuple[float, float]], float],
                         center: Tuple[float, float] = (0.0, 0.0),
                         radius: float = 1.0,
                         n_points: int = 21,
                         iterations: int = 5) -> Tuple[Tuple[float, float], float]:
        """
        Optimize using golden ratio spiral search pattern.

        Searches function space using golden angle spiral, which provides
        optimal coverage of circular search regions.

        Args:
            f: Objective function taking (x, y) tuple
            center: Center of spiral search
            radius: Initial search radius
            n_points: Points per spiral turn
            iterations: Number of spiral iterations

        Returns:
            Tuple of (optimal_point, optimal_value)
        """
        golden_angle = 2 * math.pi * (1 - 1/self.phi)  # ≈ 2.4 radians

        best_point = center
        best_value = f(center)

        for iteration in range(iterations):
            current_radius = radius / (1 + iteration * self.inv_phi)

            for i in range(n_points):
                angle = i * golden_angle
                r = current_radius * math.sqrt(i / n_points)

                x = center[0] + r * math.cos(angle)
                y = center[1] + r * math.sin(angle)

                point = (x, y)
                value = f(point)

                if value < best_value:
                    best_value = value
                    best_point = point

        return best_point, best_value

    def fibonacci_cooling_schedule(self,
                                 initial_temp: float,
                                 n_steps: int) -> List[float]:
        """
        Generate Fibonacci-based cooling schedule for simulated annealing.

        Creates temperature schedule using Fibonacci ratios, providing
        natural cooling with golden ratio decay properties.

        Args:
            initial_temp: Initial temperature
            n_steps: Number of cooling steps

        Returns:
            List of temperatures with Fibonacci-based decay
        """
        temperatures = []

        for i in range(n_steps):
            if i == 0:
                temp = initial_temp
            else:
                # Fibonacci-based cooling: T_n = T_0 * F_n/F_{n+k}
                fib_ratio = self._fibonacci_ratio(i + 1)
                temp = initial_temp * fib_ratio

            temperatures.append(temp)

        return temperatures
