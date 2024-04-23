**Genetic Algorithm**

This Python code implements a simple genetic algorithm (GA) for optimization problems. Genetic algorithms are heuristic search algorithms inspired by the process of natural selection and genetics. They are commonly used for optimization and search problems.

### Usage

1. **Install Dependencies**: Ensure you have the necessary dependencies installed. You can install them via pip:

   ```bash
   pip install numpy tqdm
   ```

2. **Run the Code**: Run the main script to see the GA in action.

   ```bash
   python main.py
   ```

### Description

The `GeneticAlgorithm` class defines the genetic algorithm. Here's a brief description of its main methods:

- `generate_population`: Initializes a population of individuals.
- `tournament_selection`: Selects individuals from the population using tournament selection.
- `crossover`: Performs crossover (recombination) between selected individuals.
- `generate_offspring`: Generates offspring from selected individuals.
- `mutation`: Mutates the offspring population.
- `select`: Selects a subset of individuals from the population.
- `elitism`: Implements elitism, preserving the best individual.
- `evolve`: Executes the genetic algorithm for optimization.

### Parameters

- `bounds`: Tuple specifying the lower and upper bounds of the search space.
- `pop_size`: Population size.
- `mutation_rate`: Probability of mutation.
- `mutation_size`: Size of mutation.
- `recombination_rate`: Size of recombination.
- `dim`: Dimensionality of the search space.

### Example

An example usage of the genetic algorithm is provided in the `main` function:

- Defines parameters for the GA.
- Creates an instance of `GeneticAlgorithm`.
- Generates an initial population.
- Executes the GA using the Rosenbrock function as the objective function.
- Prints the best individual found and its objective function value.
- You can also simulate by calling the `simulate` method passing the number of executions of the algorithm you want to test.

### Objective Function

The example uses the Rosenbrock function for optimization. This is a classic optimization test function often used to evaluate optimization algorithms.

### Note

- This code is a basic implementation and may need modifications for specific problems or performance improvements.
- Ensure proper parameter tuning for your problem domain.
- Feel free to modify the code to suit your needs.

### References

- [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function)
- [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)