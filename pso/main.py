import numpy as np
import pso
from parameters import ParameterLoader
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.problems.single import Rosenbrock
from pymoo.optimize import minimize


# Função de Rosenbrock para N dimensões
def rosenbrock(X, a=1, b=100):
    return sum(a * (X[i+1] - X[i]**2)**2 + b * (X[i] - 1)**2 for i in range(len(X)-1))

def ackley(X):
    x, y = X
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + 20 + np.e


def main():
    pop_size = 30
    bounds = [-5, 5]
    dim = 2
    swarm = pso.Swarm(pop_size, bounds, dim, rosenbrock)
    loader = ParameterLoader(dim) # insert -1 for plot visualization params

    personal_coefficient, social_coefficient, inertia_weight, max_velocity = loader.load_params()
    optimizer = pso.PSO(personal_coefficient, social_coefficient, inertia_weight, max_velocity)
    positions, velocities = optimizer.pso(swarm)

    # swarm.generate_gif(positions, velocities)

    print("\nPYMOO: ")
    problem = Rosenbrock()

    algorithm = PSO()

    res = minimize(problem,
                   algorithm,
                   verbose=False)

    print(f"Best solution: {res.F}")
    print(f"Best individual: {res.X}")

if __name__ == "__main__":
    main()
