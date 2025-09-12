"""
Core Zeckendorf Representation Implementation

This module implements the fundamental mathematical operations for Zeckendorf representation,
which expresses every positive integer as a unique sum of non-consecutive Fibonacci numbers.

Mathematical Foundation:
- Zeckendorf Theorem: Every positive integer n has a unique representation as
  sum of non-consecutive Fibonacci numbers
- No-11 Constraint: No two consecutive Fibonacci numbers in representation
- Provides natural bounds and sparsity for optimization algorithms
"""

import math
from typing import Dict, List, Tuple


class ZeckendorfRepresentation:
    """
    Implements Zeckendorf representation and associated mathematical operations.

    The Zeckendorf representation provides a unique way to express positive integers
    using non-consecutive Fibonacci numbers, which creates natural bounds and
    sparsity properties useful for optimization algorithms.
    """

    def __init__(self, max_n: int = 1000):
        """
        Initialize with precomputed Fibonacci sequence up to max_n.

        Args:
            max_n: Maximum integer to support in Zeckendorf representation
        """
        self.max_n = max_n
        self.fibonacci_sequence = self._generate_fibonacci(max_n)
        self.phi = (1 + math.sqrt(5)) / 2  # Golden ratio

    def _generate_fibonacci(self, max_n: int) -> List[int]:
        """
        Generate Fibonacci sequence up to max_n.

        Args:
            max_n: Maximum value to generate

        Returns:
            List of Fibonacci numbers up to max_n
        """
        fib = [1, 2]  # Start with F_2=1, F_3=2 (standard indexing)

        while fib[-1] <= max_n:
            next_fib = fib[-1] + fib[-2]
            if next_fib > max_n:
                break
            fib.append(next_fib)

        return fib

    def to_zeckendorf(self, n: int) -> List[int]:
        """
        Convert positive integer n to Zeckendorf representation.

        Args:
            n: Positive integer to convert

        Returns:
            List of non-consecutive Fibonacci numbers that sum to n

        Raises:
            ValueError: If n <= 0 or n > max_n
        """
        if n <= 0:
            raise ValueError("Zeckendorf representation only defined for positive integers")
        if n > self.max_n:
            raise ValueError(f"Integer {n} exceeds maximum supported value {self.max_n}")

        result = []
        remaining = n

        # Greedy algorithm: use largest Fibonacci number <= remaining
        for fib in reversed(self.fibonacci_sequence):
            if fib <= remaining:
                result.append(fib)
                remaining -= fib
                if remaining == 0:
                    break

        return sorted(result)

    def from_zeckendorf(self, fib_numbers: List[int]) -> int:
        """
        Convert Zeckendorf representation back to integer.

        Args:
            fib_numbers: List of Fibonacci numbers in Zeckendorf representation

        Returns:
            Integer sum of the Fibonacci numbers

        Raises:
            ValueError: If representation violates No-11 constraint
        """
        if not self._validate_no_consecutive(fib_numbers):
            raise ValueError("Invalid Zeckendorf representation: consecutive Fibonacci numbers")

        return sum(fib_numbers)

    def _validate_no_consecutive(self, fib_numbers: List[int]) -> bool:
        """
        Validate that Fibonacci numbers satisfy No-11 constraint (no consecutive).

        Args:
            fib_numbers: List of Fibonacci numbers to validate

        Returns:
            True if valid (no consecutive), False otherwise
        """
        if len(fib_numbers) <= 1:
            return True

        sorted_fibs = sorted(fib_numbers)

        for i in range(len(sorted_fibs) - 1):
            # Check if current and next are consecutive in Fibonacci sequence
            curr_idx = self.fibonacci_sequence.index(sorted_fibs[i])
            next_idx = self.fibonacci_sequence.index(sorted_fibs[i + 1])

            if next_idx - curr_idx == 1:  # Consecutive in sequence
                return False

        return True

    def get_entropy_density(self, n: int) -> float:
        """
        Calculate entropy density of Zeckendorf representation for n.

        The entropy density measures information content relative to binary representation.
        Higher entropy density indicates more efficient encoding.

        Args:
            n: Positive integer

        Returns:
            Entropy density (bits per representation element)
        """
        zeck_repr = self.to_zeckendorf(n)
        binary_length = len(bin(n)) - 2  # Remove '0b' prefix
        zeck_length = len(zeck_repr)

        if zeck_length == 0:
            return 0.0

        return binary_length / zeck_length

    def bounded_growth_factor(self, n: int) -> float:
        """
        Calculate bounded growth factor for Zeckendorf representation.

        This measures how the representation length grows relative to golden ratio,
        providing bounds for optimization algorithm complexity.

        Args:
            n: Positive integer

        Returns:
            Growth factor relative to log_phi(n)
        """
        zeck_repr = self.to_zeckendorf(n)
        
        # Handle edge case for n=1 where log(n)=0
        if n == 1:
            return 1.0  # For n=1, theoretical and actual lengths are both 1
        
        theoretical_length = math.log(n) / math.log(self.phi)
        actual_length = len(zeck_repr)

        # Additional safety check
        if theoretical_length == 0:
            return 1.0
            
        return actual_length / theoretical_length

    def get_fibonacci_index(self, fib_num: int) -> int:
        """
        Get index of Fibonacci number in sequence.

        Args:
            fib_num: Fibonacci number

        Returns:
            Index in Fibonacci sequence

        Raises:
            ValueError: If not a valid Fibonacci number
        """
        try:
            return self.fibonacci_sequence.index(fib_num)
        except ValueError:
            raise ValueError(f"{fib_num} is not a Fibonacci number in sequence")
