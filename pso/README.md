
# Particle Swarm Optimization (PSO) for Optimization Problems

This repository contains Python code for optimizing functions using Particle Swarm Optimization (PSO) algorithm. The PSO algorithm is a population-based stochastic optimization technique inspired by the social behavior of birds flocking or fish schooling.

## Requirements
- Python 3.x
- NumPy
- PyMOO

## Installation
You can install the required packages using pip:

```
pip3 install numpy
pip3 install pymoo
```

## Usage
1. Define your objective function(s).
2. Set the appropriate parameters for the PSO algorithm in the `parameters.py` file.
3. Run the `main.py` file.

## Example
```python
python main.py
```

## Parameters
- **pop_size**: Size of the particle swarm.
- **bounds**: Boundary values for the search space.
- **dim**: Dimensionality of the problem.
- **personal_coefficient**: Coefficient for personal best update.
- **social_coefficient**: Coefficient for global best update.
- **inertia_weight**: Inertia weight.
- **max_velocity**: Maximum velocity of particles.

## Results
The best solution and best individual found by PSO will be printed out.
![PSO Optimization](images/convergence_ackley_2024-05-02.gif)