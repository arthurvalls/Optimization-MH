import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime


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
    def __init__(self, swarm_size, bounds, dim, cost_function):
        self.dim = dim
        self.bounds = bounds
        self.particles = []
        self.best_fitness = np.inf
        self.best_genes = None
        self.cost_function = cost_function
        for _ in range(swarm_size):
            genes = np.random.uniform(self.bounds[0], self.bounds[1], dim)
            particle = Particle(genes)
            self.particles.append(particle)
            if self.best_genes is None or particle.fitness < self.best_fitness:
                self.best_genes = particle.genes.copy()
                self.best_fitness = particle.fitness
    
    def generate_gif(self, positions, velocities):
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
            ax.contourf(X, Y, Z, levels=50, cmap='rainbow', alpha=0.5)
            ax.scatter(x, y, marker="8", c="r", s=50)  # Change the size here (100 is just an example)


            # Plot arrows
            for pos, vel in zip(current_positions, current_velocities):
                ax.arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.1, head_length=0.1, color='black')

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


pop_size = 30
bounds = [-5., 5.]
dim = 2
# hyperparameters
P_C = S_C = 0.2
W = 0.7
V_MAX = 0.15
swarm = Swarm(pop_size, bounds, dim, rosenbrock)

def pso(swarm, max_iter=100):
    positions = []  # Store positions of particles
    velocities = []
    for i in range(max_iter):
        for particle in swarm.particles:
            random_coefficients = np.random.uniform(size=2)
            p_c = P_C * random_coefficients[0] * (particle.best_genes - particle.genes) # personal coefficient
            s_c = S_C * random_coefficients[1] * (swarm.best_genes - particle.genes) # social coefficient
            velocity = W * particle.velocity + p_c + s_c
            velocity = np.clip(velocity, -V_MAX, V_MAX)
            particle.velocity = velocity.copy()
            particle.genes += velocity
            particle.fitness = swarm.cost_function(particle.genes)

            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_genes = particle.genes.copy()

                if particle.best_fitness < swarm.best_fitness:
                    swarm.best_genes = particle.genes.copy()
                    swarm.best_fitness = particle.best_fitness

            particle.genes = np.clip(particle.genes.copy(), swarm.bounds[0], swarm.bounds[1])
            particle.fitness = swarm.cost_function(particle.genes)

        positions.append([particle.genes for particle in swarm.particles])
        velocities.append([particle.velocity for particle in swarm.particles])

    print(f"Best fitness: {swarm.best_fitness}")
    print(f"Best genes: {swarm.best_genes}")
    return positions, velocities


positions, velocities = pso(swarm)

#swarm.generate_gif(positions, velocities)