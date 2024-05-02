import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import time


# Função de Rosenbrock para N dimensões
def rosenbrock(X, a=1, b=100):
    return sum(a * (X[i+1] - X[i]**2)**2 + b * (X[i] - 1)**2 for i in range(len(X)-1))

def ackley(X):
    x, y = X
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + 20 + np.e


def func(X):
	x, y = X
	return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)
 

class Particle:
    def __init__(self, genes):
        self.genes = genes
        self.velocity = np.zeros_like(genes)
        self.fitness = np.inf
        self.best_genes = genes.copy()
        self.best_fitness = np.inf

class Swarm:
    def __init__(self, swarm_size, bounds, dim, cost_function):
        self.dim = dim
        self.bounds = bounds
        self.particles = []
        self.best_fitness = np.inf
        self.best_genes = []
        self.cost_function = cost_function
        for _ in range(swarm_size):
            genes = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
            particle = Particle(genes)
            particle.fitness = self.cost_function(particle.genes)
            self.particles.append(particle)
            if particle.fitness < self.best_fitness:
                self.best_genes = particle.genes.copy()
                self.best_fitness = particle.fitness
    
    def generate_gif(self, positions, velocities):
    	if self.dim < 3:
	        def plot_generation(positions, velocities, i, ax):
	            x = np.linspace(self.bounds[0], self.bounds[1], 100)
	            y = np.linspace(self.bounds[0], self.bounds[1], 100)
	            X, Y = np.meshgrid(x, y)
	            Z = self.cost_function((X, Y))

	            current_positions = positions[i]
	            current_velocities = velocities[i]
	            x = [point[0] for point in current_positions]
	            y = [point[1] for point in current_positions]

	            ax.clear()
	            ax.contourf(X, Y, Z, levels=50, cmap='plasma', alpha=0.5)
	            ax.scatter(x, y, marker="*", c="r")  # Change the size here (100 is just an example)


	            # Plot arrows
	            for pos, vel in zip(current_positions, current_velocities):
	                ax.arrow(pos[0], pos[1], vel[0], vel[1], head_width=self.bounds[1]*0.02, head_length=self.bounds[1]*0.02, color='black')

	            ax.set_xlabel('X')
	            ax.set_ylabel('Y')
	            ax.set_title('Generation {}'.format(i + 1))
	            # ax.legend()


	        positions = positions[:60]
	        velocities = velocities[:60]

	        fig, ax = plt.subplots(figsize=(10, 10))

	        def update_plot(i):
	            plot_generation(positions, velocities, i, ax)

	        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	        gif_filename = f'images/convergence_{self.cost_function.__name__}_{current_date}.gif'
	        ani = FuncAnimation(fig, update_plot, frames=len(positions), interval=115)

	        ani.save(gif_filename, writer='pillow')

	        print(f"Gif saved at: {gif_filename}")

	        plt.close()


# hyperparameters taken from: "Good parameters for particle swarm optimization", MEH Pedersen.
# good enough for the rosenbrock function

P_C = 2.5586 # personal_coefficient previously 0.2
S_C = 1.3358 # social_coefficient 0.2
W = 0.3925 # inertia weight
V_MAX = 1 # max velocity

def pso(swarm, max_iter=100):
    positions = []  # Store positions of particles
    velocities = []

    for _ in range(max_iter):
        for particle in swarm.particles:

        	# update velocity
            random_coefficients = np.random.uniform(size=2)

            personal_c = P_C * random_coefficients[0] * (particle.best_genes - particle.genes) # personal coefficient
            social_c = S_C * random_coefficients[1] * (swarm.best_genes - particle.genes) # social coefficient
            velocity = (W * particle.velocity) + personal_c + social_c
            velocity = np.clip(velocity, -V_MAX, V_MAX)
            particle.velocity = velocity.copy()

            # update genes
            particle.genes += velocity
            particle.genes = np.clip(particle.genes, swarm.bounds[0], swarm.bounds[1])

            # get fitness
            particle.fitness = swarm.cost_function(particle.genes)

            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_genes = particle.genes.copy()

                if particle.best_fitness < swarm.best_fitness:
                    swarm.best_fitness = particle.best_fitness
                    swarm.best_genes = particle.genes.copy()

        positions.append([particle.genes.copy() for particle in swarm.particles])
        velocities.append([particle.velocity.copy() for particle in swarm.particles])

    print(f"Best fitness: {swarm.best_fitness}")
    print(f"Best genes: {swarm.best_genes}")
    return positions, velocities



pop_size = 20
bounds = [-5, 5]
dim = 2
swarm = Swarm(pop_size, bounds, dim, rosenbrock)
positions, velocities = pso(swarm)


#swarm.generate_gif(positions, velocities)

print("PYMOO: ")
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.problems.single import Rosenbrock
from pymoo.optimize import minimize

problem = Rosenbrock()

algorithm = PSO()

res = minimize(problem,
               algorithm,
               verbose=False)

print(f"Best solution: {res.F}")
print(f"Best individual: {res.X}")
