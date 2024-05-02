import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Função de Rosenbrock para N dimensões
def rosenbrock(X, a=1, b=100):
    return sum(a * (X[i+1] - X[i]**2)**2 + b * (X[i] - 1)**2 for i in range(len(X)-1))

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


    def generate_gif(self, function, generations):
	    def plot_generation(points, i, ax):
	        x = np.linspace(self.bounds[0], self.bounds[1], 100)
	        y = np.linspace(self.bounds[0], self.bounds[1], 100)
	        X, Y = np.meshgrid(x, y)
	        Z = function((X, Y))

	        # Plot only a subset of points for better performance
	        subset_points = points[i]

	        print(f"gen {i}")
	        ax.clear()
	        ax.contourf(X, Y, Z, levels=20, cmap='rainbow', alpha=0.5)
	        ax.scatter([p.genes[0] for p in subset_points], [p.genes[1] for p in subset_points], color='red', alpha=0.3)
	        ax.set_xlabel('X')
	        ax.set_ylabel('Y')
	        ax.set_title('Generation {}'.format(i + 1))

	    # Example array of arrays of points
	    points = generations[:30]

	    # Initialize figure and axis
	    fig, ax = plt.subplots(figsize=(10, 10))

	    # Function to update plot for each frame
	    def update_plot(i):
	        plot_generation(points, i, ax)

	    # Generate and save GIF
	    gif_filename = 'images/convergence.gif'
	    frames = min(len(points), 100)  # Limit number of frames to 100 for performance
	    ani = FuncAnimation(fig, update_plot, frames=frames, interval=250)

	    # Save the GIF
	    ani.save(gif_filename, writer='pillow')

	    plt.close()


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
        fig = plt.figure("Particle Swarm Optimization")
    
    for i in range(max_iter):
        if plot:
            fig.clf()
            ax = fig.add_subplot(1, 1, 1)
            ac = ax.contourf(X, Y, cost_function(X, Y), cmap='viridis')
            fig.colorbar(ac)
        
        for particle in swarm.particles:
            random_coefficients = np.random.uniform(size=swarm.dim)
            p_c = P_C * random_coefficients[0] * (particle.best_genes - particle.genes) # personal coefficient
            s_c = S_C * random_coefficients[1] * (swarm.best_genes - particle.genes) # social coefficient
            velocity = W * particle.velocity + p_c + s_c
            velocity = np.clip(velocity, -V_MAX, V_MAX)
            particle.velocity = velocity.copy()

            if plot:
                ax.scatter(particle.genes[0], particle.genes[1], marker='*', c='r')
                ax.arrow(particle.genes[0], particle.genes[1], particle.velocity[0], particle.velocity[1], head_width=0.1, head_length=0.2, color='k')

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


pso(swarm, rosenbrock)

# print("Plotting gif...")
# swarm.generate_gif(rosenbrock, generations)