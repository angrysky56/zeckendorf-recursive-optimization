"""
Optimization algorithms using Fibonacci constraints and golden ratio principles.

This module implements practical optimization algorithms:
- Fibonacci-constrained gradient descent
- Golden ratio evolutionary algorithms  
- Zeckendorf space optimization methods
"""

from .fibonacci_optimizer import (
    FibonacciConstrainedGradientDescent,
    GoldenRatioEvolutionaryOptimizer,
    FibonacciConstrainedOptimizer
)

__all__ = [
    "FibonacciConstrainedGradientDescent",
    "GoldenRatioEvolutionaryOptimizer", 
    "FibonacciConstrainedOptimizer"
]
