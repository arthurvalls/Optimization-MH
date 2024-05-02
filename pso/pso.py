import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Função de Rosenbrock para N dimensões
def rosenbrock(X, a=1, b=100):
    return sum(a * (X[i+1] - X[i]**2)**2 + b * (X[i] - 1)**2 for i in range(len(X)-1))

def ackley(X):
    x, y = X
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + 20 + np.e


class Particle:
    def __init__(self, genes):
        self.genes = genes
        self.velocity = np.zeros_like(genes)
        self.fitness = np.inf
        self.best_genes = genes.copy()
        self.best_fitness = np.inf

class Swarm:
    def __init__(self, swarm_size, bounds, dim):
        self.dim = dim
        self.bounds = bounds
        self.particles = []
        self.best_fitness = np.inf
        self.best_genes = None
        for _ in range(swarm_size):
            genes = np.random.uniform(self.bounds[0], self.bounds[1], dim)
            particle = Particle(genes)
            self.particles.append(particle)
            if self.best_genes is None or particle.fitness < self.best_fitness:
                self.best_genes = particle.genes.copy()
                self.best_fitness = particle.fitness


pop_size = 30
bounds = [-5., 5.]
dim = 2

# hyperparameters
P_C = S_C = 0.1
W = 0.5
V_MAX = 0.1
swarm = Swarm(pop_size, bounds, dim)


def pso(swarm, cost_function, max_iter=100, plot=True):
    plot = False if swarm.dim > 2 else True

    if plot:    # Initialize plotting variables
        x = np.linspace(swarm.bounds[0], swarm.bounds[1], 50)
        y = np.linspace(swarm.bounds[0], swarm.bounds[1], 50)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure("Particle Swarm Optimization", figsize=(8, 6))
    
    for i in range(max_iter):
        if plot:
            fig.clf()
            ax = fig.add_subplot(1, 1, 1)
            if cost_function == rosenbrock:
            	ac = ax.contourf(X, Y, cost_function((X, Y)), levels=np.logspace(-1, 3, 10), cmap='plasma', locator=plt.LogLocator())  # Use logarithmic scale
            else:
            	ac = ax.contourf(X, Y, cost_function((X, Y)), cmap='plasma')
            fig.colorbar(ac)
        
        for particle in swarm.particles:
            random_coefficients = np.random.uniform(size=swarm.dim)
            p_c = P_C * random_coefficients[0] * (particle.best_genes - particle.genes) # personal coefficient
            s_c = S_C * random_coefficients[1] * (swarm.best_genes - particle.genes) # social coefficient
            velocity = W * particle.velocity + p_c + s_c
            velocity = np.clip(velocity, -V_MAX, V_MAX)
            particle.velocity = velocity.copy()

            if plot:
                ax.scatter(particle.genes[0], particle.genes[1], marker='X', c='r')
                ax.arrow(particle.genes[0], particle.genes[1], particle.velocity[0], particle.velocity[1], head_width=0.1, head_length=0.2, color='black')

            particle.genes += velocity
            particle.fitness = cost_function(particle.genes)

            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_genes = particle.genes.copy()

                if particle.best_fitness < swarm.best_fitness:
                    swarm.best_genes = particle.genes.copy()
                    swarm.best_fitness = particle.best_fitness

            particle.genes = np.clip(particle.genes.copy(), swarm.bounds[0], swarm.bounds[1])
            particle.fitness = cost_function(particle.genes)

        if plot:
            plt.subplots_adjust(right=0.95)
            plt.pause(0.00001)

    print(f"Best fitness: {swarm.best_fitness}")
    print(f"Best genes: {swarm.best_genes}")
    if plot:
        plt.show()


pso(swarm, ackley)

# print("Plotting gif...")
# swarm.generate_gif(rosenbrock, generations)