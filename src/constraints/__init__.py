"""
Constraint systems for Zeckendorf-based optimization.

This module implements constraint handling systems:
- No-11 constraints for preventing consecutive Fibonacci terms
- Constraint projection and penalty methods
"""

from .no11_constraint import No11ConstraintSystem

__all__ = ["No11ConstraintSystem"]
