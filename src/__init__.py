"""
Zeckendorf-Recursive Optimization Package

A research implementation of optimization algorithms based on Zeckendorf representation,
Fibonacci constraints, and golden ratio principles.

This package provides:
- Core mathematical operations for Zeckendorf representation
- Golden ratio optimization utilities
- No-11 constraint systems for bounded optimization
- Fibonacci-constrained optimization algorithms

Mathematical Foundation:
The package implements practical applications of mathematical concepts from
Zeckendorf-Hilbert theory, focusing on the testable and algorithmically
useful aspects while avoiding speculative applications.
"""

from .core.zeckendorf import ZeckendorfRepresentation
from .core.golden_ratio import GoldenRatioOptimizer
from .constraints.no11_constraint import No11ConstraintSystem
from .algorithms.fibonacci_optimizer import (
    FibonacciConstrainedGradientDescent,
    GoldenRatioEvolutionaryOptimizer
)

__version__ = "0.1.0"
__author__ = "Research Team"

__all__ = [
    "ZeckendorfRepresentation",
    "GoldenRatioOptimizer", 
    "No11ConstraintSystem",
    "FibonacciConstrainedGradientDescent",
    "GoldenRatioEvolutionaryOptimizer"
]

# Package-level configuration
DEFAULT_MAX_VALUE = 1000
GOLDEN_RATIO = (1 + 5**0.5) / 2
