"""
No-11 Constraint System for Zeckendorf Optimization

This module implements constraint systems based on the "No-11" principle,
which prevents consecutive Fibonacci numbers in Zeckendorf representations.
This constraint provides natural bounds and sparsity for optimization algorithms.

Mathematical Foundation:
- No-11 Constraint: No consecutive Fibonacci numbers in representation
- Creates natural sparsity similar to L1 regularization
- Provides bounded growth and convergence guarantees
- Enables efficient constraint handling in optimization
"""

from typing import Callable, List, Optional, Set, Tuple

import numpy as np

from ..core.zeckendorf import ZeckendorfRepresentation


class No11ConstraintSystem:
    """
    Implements No-11 constraint system for Zeckendorf-based optimization.

    The No-11 constraint prevents consecutive Fibonacci numbers in representations,
    creating natural sparsity and bounds that are beneficial for optimization algorithms.
    """

    def __init__(self, max_n: int = 1000):
        """
        Initialize constraint system with Zeckendorf representation.

        Args:
            max_n: Maximum integer for Zeckendorf representations
        """
        self.zeck = ZeckendorfRepresentation(max_n)
        self.max_n = max_n

    def is_valid_representation(self, fib_numbers: List[int]) -> bool:
        """
        Check if Fibonacci number list satisfies No-11 constraint.

        Args:
            fib_numbers: List of Fibonacci numbers to validate

        Returns:
            True if valid (no consecutive Fibonacci numbers)
        """
        return self.zeck._validate_no_consecutive(fib_numbers)

    def project_to_valid(self, fib_numbers: List[int]) -> List[int]:
        """
        Project invalid representation to nearest valid No-11 representation.

        When consecutive Fibonacci numbers are present, this method projects
        to the nearest valid representation by combining consecutive terms.

        Args:
            fib_numbers: Potentially invalid Fibonacci representation

        Returns:
            Valid Fibonacci representation satisfying No-11 constraint
        """
        if self.is_valid_representation(fib_numbers):
            return fib_numbers.copy()

        # Sort and remove duplicates
        sorted_fibs = sorted(set(fib_numbers))
        result = []
        i = 0

        while i < len(sorted_fibs):
            current_fib = sorted_fibs[i]

            # Check if next Fibonacci number is consecutive
            if (i + 1 < len(sorted_fibs) and
                self._are_consecutive(current_fib, sorted_fibs[i + 1])):

                # Combine consecutive Fibonacci numbers using F_n + F_{n+1} = F_{n+2}
                combined_fib = self._combine_consecutive(current_fib, sorted_fibs[i + 1])
                result.append(combined_fib)
                i += 2  # Skip both combined numbers
            else:
                result.append(current_fib)
                i += 1

        # Recursively project in case combination created new consecutive pairs
        if not self.is_valid_representation(result):
            return self.project_to_valid(result)

        return result

    def _are_consecutive(self, fib1: int, fib2: int) -> bool:
        """
        Check if two Fibonacci numbers are consecutive in sequence.

        Args:
            fib1: First Fibonacci number (assumed smaller)
            fib2: Second Fibonacci number

        Returns:
            True if consecutive in Fibonacci sequence
        """
        try:
            idx1 = self.zeck.get_fibonacci_index(fib1)
            idx2 = self.zeck.get_fibonacci_index(fib2)
            return idx2 - idx1 == 1
        except ValueError:
            return False

    def _combine_consecutive(self, fib1: int, fib2: int) -> int:
        """
        Combine two consecutive Fibonacci numbers using F_n + F_{n+1} = F_{n+2}.

        Args:
            fib1: First Fibonacci number
            fib2: Second consecutive Fibonacci number

        Returns:
            Combined Fibonacci number
        """
        return fib1 + fib2

    def constraint_penalty(self, fib_numbers: List[int]) -> float:
        """
        Calculate penalty for violating No-11 constraint.

        Returns 0 for valid representations, positive penalty for violations.
        Can be used in penalized optimization methods.

        Args:
            fib_numbers: Fibonacci representation to evaluate

        Returns:
            Penalty value (0 if valid, positive if invalid)
        """
        if self.is_valid_representation(fib_numbers):
            return 0.0

        penalty = 0.0
        sorted_fibs = sorted(set(fib_numbers))

        for i in range(len(sorted_fibs) - 1):
            if self._are_consecutive(sorted_fibs[i], sorted_fibs[i + 1]):
                # Penalty proportional to magnitude of consecutive terms
                penalty += (sorted_fibs[i] + sorted_fibs[i + 1]) / self.max_n

        return penalty

    def generate_valid_neighborhood(self,
                                  fib_numbers: List[int],
                                  radius: int = 2) -> List[List[int]]:
        """
        Generate neighborhood of valid Zeckendorf representations.

        Creates nearby representations by small modifications while
        maintaining No-11 constraint validity.

        Args:
            fib_numbers: Current valid representation
            radius: Size of neighborhood to generate

        Returns:
            List of valid neighboring representations
        """
        if not self.is_valid_representation(fib_numbers):
            fib_numbers = self.project_to_valid(fib_numbers)

        neighbors = []
        fib_set = set(fib_numbers)

        # Generate neighbors by adding/removing single Fibonacci numbers
        for fib in self.zeck.fibonacci_sequence:
            if fib > self.max_n:
                break

            # Try adding this Fibonacci number
            if fib not in fib_set:
                candidate = fib_numbers + [fib]
                projected = self.project_to_valid(candidate)
                if projected not in neighbors:
                    neighbors.append(projected)

            # Try removing this Fibonacci number
            if fib in fib_set:
                candidate = [f for f in fib_numbers if f != fib]
                if candidate:  # Don't create empty representations
                    projected = self.project_to_valid(candidate)
                    if projected not in neighbors:
                        neighbors.append(projected)

        return neighbors[:radius]

    def constraint_gradient(self,
                          fib_numbers: List[int],
                          epsilon: float = 1e-6) -> np.ndarray:
        """
        Compute gradient of constraint penalty function.

        Useful for gradient-based optimization with No-11 constraints.

        Args:
            fib_numbers: Current representation
            epsilon: Finite difference step size

        Returns:
            Gradient vector of constraint penalty
        """
        base_penalty = self.constraint_penalty(fib_numbers)
        gradient = np.zeros(len(self.zeck.fibonacci_sequence))

        for i, fib in enumerate(self.zeck.fibonacci_sequence):
            if fib > self.max_n:
                break

            # Forward difference approximation
            perturbed = fib_numbers.copy()
            if fib in fib_numbers:
                # Small perturbation by removing and adding nearby value
                perturbed.remove(fib)
                if i > 0:
                    perturbed.append(self.zeck.fibonacci_sequence[i-1])
            else:
                perturbed.append(fib)

            perturbed_penalty = self.constraint_penalty(perturbed)
            gradient[i] = (perturbed_penalty - base_penalty) / epsilon

        return gradient

    def entropy_with_constraints(self, fib_numbers: List[int]) -> float:
        """
        Calculate entropy of representation considering No-11 constraints.

        Measures information content while accounting for constraint restrictions,
        providing information-theoretic measure of representation efficiency.

        Args:
            fib_numbers: Fibonacci representation

        Returns:
            Entropy value adjusted for constraint restrictions
        """
        if not self.is_valid_representation(fib_numbers):
            # Penalize invalid representations
            return -float('inf')

        # Base entropy from Zeckendorf representation
        n = sum(fib_numbers)
        base_entropy = self.zeck.get_entropy_density(n)

        # Constraint bonus: valid representations get entropy bonus
        constraint_bonus = 1.0 / (1 + self.constraint_penalty(fib_numbers))

        return base_entropy * constraint_bonus

    def optimal_constrained_representation(self, n: int) -> List[int]:
        """
        Find optimal Zeckendorf representation for n under No-11 constraints.

        Since Zeckendorf representation is unique and automatically satisfies
        No-11 constraints, this returns the standard representation.

        Args:
            n: Positive integer to represent

        Returns:
            Optimal valid Fibonacci representation
        """
        return self.zeck.to_zeckendorf(n)
