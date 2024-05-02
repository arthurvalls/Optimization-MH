import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import time

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


class PSO:
    def __init__(self, P_C, S_C, W, V_MAX):
        self.personal_c = P_C
        self.social_c = S_C
        self.inertia_weight = W
        self.max_velocity = V_MAX

    def pso(self, swarm, max_iter=100):
        positions = []  # Store positions of particles
        velocities = []

        for _ in range(max_iter):
            for particle in swarm.particles:

            	# update velocity
                random_coefficients = np.random.uniform(size=2)

                personal_c = self.personal_c * random_coefficients[0] * (particle.best_genes - particle.genes) # personal coefficient
                social_c = self.social_c * random_coefficients[1] * (swarm.best_genes - particle.genes) # social coefficient
                velocity = (self.inertia_weight * particle.velocity) + personal_c + social_c
                velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
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
