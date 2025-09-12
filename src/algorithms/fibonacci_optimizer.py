"""
Fibonacci-Constrained Optimization Algorithms

This module implements optimization algorithms that leverage Zeckendorf representations
and golden ratio principles to create bounded, efficient optimization procedures.

Key Algorithms:
- FibonacciConstrainedGradientDescent: Gradient descent with Zeckendorf parameter encoding
- GoldenRatioEvolutionaryOptimizer: EA using golden ratio mutation and crossover
- ZeckendorfSpaceOptimizer: Direct optimization in Zeckendorf representation space
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..constraints.no11_constraint import No11ConstraintSystem
from ..core.golden_ratio import GoldenRatioOptimizer
from ..core.zeckendorf import ZeckendorfRepresentation


class FibonacciConstrainedOptimizer:
    """
    Base class for optimization algorithms using Fibonacci constraints.

    Provides common functionality for Zeckendorf-based optimization including
    parameter encoding, constraint handling, and convergence analysis.
    """

    def __init__(self, max_param_value: int = 1000):
        """
        Initialize with Zeckendorf and constraint systems.

        Args:
            max_param_value: Maximum parameter value for Zeckendorf encoding
        """
        self.zeck = ZeckendorfRepresentation(max_param_value)
        self.golden = GoldenRatioOptimizer()
        self.constraints = No11ConstraintSystem(max_param_value)
        self.max_param_value = max_param_value

    def encode_parameters(self, params: np.ndarray) -> List[List[int]]:
        """
        Encode parameter vector using Zeckendorf representation.

        Args:
            params: Parameter vector to encode

        Returns:
            List of Zeckendorf representations for each parameter
        """
        # Normalize parameters to positive integers
        normalized = np.abs(params)
        max_val = np.max(normalized)
        
        # Handle edge case where all parameters are zero
        if max_val == 0:
            scaled = np.ones_like(normalized, dtype=int)  # Default to 1 for all zero params
        else:
            scaled = (normalized * self.max_param_value / max_val).astype(int)
        
        scaled = np.clip(scaled, 1, self.max_param_value)

        return [self.zeck.to_zeckendorf(int(p)) for p in scaled]

    def decode_parameters(self, encoded_params: List[List[int]]) -> np.ndarray:
        """
        Decode Zeckendorf representations back to parameter vector.

        Args:
            encoded_params: List of Zeckendorf representations

        Returns:
            Decoded parameter vector
        """
        decoded = np.array([sum(fib_list) for fib_list in encoded_params])
        return decoded / self.max_param_value

    def constraint_penalty(self, encoded_params: List[List[int]]) -> float:
        """
        Calculate total constraint penalty for encoded parameters.

        Args:
            encoded_params: List of Zeckendorf representations

        Returns:
            Total penalty for constraint violations
        """
        total_penalty = 0.0
        for fib_list in encoded_params:
            total_penalty += self.constraints.constraint_penalty(fib_list)
        return total_penalty

    def _project_to_valid_parameters(self, params: np.ndarray) -> np.ndarray:
        """
        Project parameters to valid Zeckendorf representation space.

        Args:
            params: Parameter vector to project

        Returns:
            Valid parameter vector satisfying constraints
        """
        # Encode to Zeckendorf
        encoded = self.encode_parameters(params)

        # Project each representation to valid form
        valid_encoded = [self.constraints.project_to_valid(fib_list)
                         for fib_list in encoded]

        # Decode back to parameters
        return self.decode_parameters(valid_encoded)


class FibonacciConstrainedGradientDescent(FibonacciConstrainedOptimizer):
    """
    Gradient descent optimization with Zeckendorf parameter encoding.

    Uses Fibonacci constraints to bound parameter space and golden ratio
    step sizes for improved convergence properties.
    """

    def optimize(self,
                objective: Callable[[np.ndarray], float],
                gradient: Callable[[np.ndarray], np.ndarray],
                x0: np.ndarray,
                max_iterations: int = 1000,
                tolerance: float = 1e-6,
                penalty_weight: float = 1.0) -> Tuple[np.ndarray, List[float]]:
        """
        Perform Fibonacci-constrained gradient descent optimization.

        Args:
            objective: Objective function to minimize
            gradient: Gradient function
            x0: Initial parameter vector
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            penalty_weight: Weight for constraint penalties

        Returns:
            Tuple of (optimal_parameters, convergence_history)
        """
        x = x0.copy()
        history = []

        # Initialize with golden ratio step size
        step_size = 1.0 / self.golden.phi

        for iteration in range(max_iterations):
            # Encode current parameters
            encoded_x = self.encode_parameters(x)

            # Calculate penalized gradient
            grad = gradient(x)
            penalty_grad = self._constraint_gradient(encoded_x)

            penalized_grad = grad + penalty_weight * penalty_grad
            grad_norm = np.linalg.norm(penalized_grad)

            if grad_norm < tolerance:
                break

            # Fibonacci step size adaptation
            if iteration > 0:
                fib_ratio = self.golden._fibonacci_ratio(iteration)
                step_size *= fib_ratio

            # Update parameters
            x_new = x - step_size * penalized_grad

            # Project to valid Zeckendorf representation
            x = self._project_to_valid_parameters(x_new)

            # Record objective value
            obj_value = objective(x) + penalty_weight * self.constraint_penalty(encoded_x)
            history.append(obj_value)

        return x, history

    # Method moved to base class FibonacciConstrainedOptimizer

    def _constraint_gradient(self, encoded_params: List[List[int]]) -> np.ndarray:
        """
        Compute the gradient of the constraint penalty with respect to parameters.

        Args:
            encoded_params: List of Zeckendorf representations

        Returns:
            Gradient vector of constraint penalties
        """
        # Simple heuristic: penalty gradient is proportional to the penalty for each parameter
        penalty_grad = []
        for fib_list in encoded_params:
            penalty = self.constraints.constraint_penalty(fib_list)
            # Assign penalty to each parameter (could be improved for more accuracy)
            penalty_grad.append(penalty)
        return np.array(penalty_grad)

    def _project_to_valid_parameters(self, params: np.ndarray) -> np.ndarray:
        """
        Project parameters to valid Zeckendorf representation space.

        Args:
            params: Parameter vector to project

        Returns:
            Valid parameter vector satisfying constraints
        """
        # Encode to Zeckendorf
        encoded = self.encode_parameters(params)

        # Project each representation to valid form
        valid_encoded = [self.constraints.project_to_valid(fib_list)
                        for fib_list in encoded]

        # Decode back to parameters
        return self.decode_parameters(valid_encoded)


class GoldenRatioEvolutionaryOptimizer(FibonacciConstrainedOptimizer):
    """
    Evolutionary algorithm using golden ratio principles and Fibonacci constraints.

    Population-based optimization with golden ratio mutation rates,
    Fibonacci crossover operations, and constraint-aware selection.
    """

    def optimize(self,
                objective: Callable[[np.ndarray], float],
                bounds: List[Tuple[float, float]],
                population_size: int = 50,
                max_generations: int = 100,
                mutation_rate: Optional[float] = None) -> Tuple[np.ndarray, List[float]]:
        """
        Evolutionary optimization with golden ratio operations.

        Args:
            objective: Objective function to minimize
            bounds: Parameter bounds for each dimension
            population_size: Size of population
            max_generations: Maximum generations
            mutation_rate: Mutation rate (defaults to 1/φ)

        Returns:
            Tuple of (best_parameters, convergence_history)
        """
        if mutation_rate is None:
            mutation_rate = 1.0 / self.golden.phi

        n_dims = len(bounds)

        # Initialize population
        population = self._initialize_population(population_size, bounds)
        history = []

        for generation in range(max_generations):
            # Evaluate fitness with constraint penalties
            fitness = self._evaluate_population(population, objective)

            # Record best fitness
            best_idx = np.argmin(fitness)
            history.append(fitness[best_idx])

            # Golden ratio selection
            selected = self._golden_selection(population, fitness)

            # Fibonacci crossover
            offspring = self._fibonacci_crossover(selected, bounds)

            # Golden ratio mutation
            mutated = self._golden_mutation(offspring, bounds, mutation_rate)

            # Replace population
            population = mutated

        # Return best individual
        final_fitness = self._evaluate_population(population, objective)
        best_idx = np.argmin(final_fitness)

        return population[best_idx], history

    def _initialize_population(self,
                             size: int,
                             bounds: List[Tuple[float, float]]) -> List[np.ndarray]:
        """Initialize population with valid Fibonacci constraints."""
        population = []

        for _ in range(size):
            individual = np.array([
                random.uniform(low, high) for low, high in bounds
            ])
            # Project to valid representation
            individual = self._project_to_valid_parameters(individual)
            population.append(individual)

        return population

    def _evaluate_population(self,
                           population: List[np.ndarray],
                           objective: Callable[[np.ndarray], float]) -> List[float]:
        """Evaluate population with constraint penalties."""
        fitness = []

        for individual in population:
            obj_value = objective(individual)
            encoded = self.encode_parameters(individual)
            penalty = self.constraint_penalty(encoded)
            fitness.append(obj_value + penalty)

        return fitness

    def _golden_selection(self,
                        population: List[np.ndarray],
                        fitness: List[float]) -> List[np.ndarray]:
        """Select parents using golden ratio tournament selection."""
        n_selected = int(len(population) * self.golden.inv_phi)
        selected = []

        fitness_array = np.array(fitness)
        sorted_indices = np.argsort(fitness_array)

        # Select top golden ratio fraction
        for i in range(n_selected):
            selected.append(population[sorted_indices[i]])

        return selected

    def _fibonacci_crossover(self,
                           parents: List[np.ndarray],
                           bounds: List[Tuple[float, float]]) -> List[np.ndarray]:
        """Crossover using Fibonacci sequence ratios."""
        offspring = []
        n_parents = len(parents)

        for i in range(len(parents) * 2):  # Produce 2x offspring
            parent1 = parents[i % n_parents]
            parent2 = parents[(i + 1) % n_parents]

            # Fibonacci ratio for crossover weight
            fib_weight = self.golden._fibonacci_ratio(i + 1)

            child = fib_weight * parent1 + (1 - fib_weight) * parent2
            child = self._project_to_valid_parameters(child)
            offspring.append(child)

        return offspring

    def _golden_mutation(self,
                       population: List[np.ndarray],
                       bounds: List[Tuple[float, float]],
                       mutation_rate: float) -> List[np.ndarray]:
        """Mutate using golden ratio step sizes."""
        mutated = []

        for individual in population:
            if random.random() < mutation_rate:
                # Golden ratio mutation step
                mutation_step = np.random.normal(0, 1.0/self.golden.phi, len(individual))
                mutant = individual + mutation_step

                # Clip to bounds
                for i, (low, high) in enumerate(bounds):
                    mutant[i] = np.clip(mutant[i], low, high)

                mutant = self._project_to_valid_parameters(mutant)
                mutated.append(mutant)
            else:
                mutated.append(individual.copy())

        return mutated
