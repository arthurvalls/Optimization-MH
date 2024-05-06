import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime

class DifferentialEvolution:
    def __init__(self, fun, dim, pop_size, bounds, CR=0.9, F=0.8):
        self.fun = fun
        self.dim = dim
        self.pop_size = pop_size
        self.bounds = bounds
        self.CR = CR
        self.F = F

    def mutate(self, population):
        mutated_population = np.zeros_like(population)
        for i in range(self.pop_size):
            idxs = list(range(self.pop_size))
            idxs.remove(i)
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = population[a] + self.F * (population[b] - population[c])
            mutated_population[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutated_population

    def crossover(self, population, mutated_population):
        crossover_population = np.zeros_like(population)
        for i in range(self.pop_size):
            gene = np.random.randint(self.dim)
            mask = np.random.rand(self.dim) < self.CR
            mask[gene] = True
            # if true herda dos mutados else herda da populacao atual
            crossover_population[i] = np.where(mask, mutated_population[i], population[i])
        return crossover_population

    def select_population(self, population, crossover_population):
        new_pop = np.zeros_like(population)
        for i in range(len(population)):
            if self.fun(population[i]) < self.fun(crossover_population[i]):
                new_pop[i] = population[i]
            else:
                new_pop[i] = crossover_population[i]
        return new_pop

    def optimize(self, max_iter=100):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        generations = []
        for it in tqdm(range(max_iter)):
            mutated_population = self.mutate(population)
            crossover_population = self.crossover(population, mutated_population)
            population = self.select_population(population, crossover_population)
            generations.append(population)
        
        best_ind = min(population, key=self.fun)
        return best_ind, self.fun(best_ind), generations

class Simulator:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def plot_dims(self, dims):
        for d in dims:
            self.optimizer.dim = d
            _, _, gens = self.optimizer.optimize(3000)
            bests = []
            for gen in gens:
                b = min(gen, key=self.optimizer.fun)
                bests.append(self.optimizer.fun(b))  
            plt.plot(bests, label='d={}'.format(d)) 

        plt.legend()
        plt.savefig(f"images/{'_'.join(map(str, dims))}.png")
        plt.show()

    def generate_gif(self, generations):
        if self.optimizer.dim < 3:
            print("Plotting gif...")
            def plot_generation(points, i, ax):
                x = np.linspace(self.optimizer.bounds[0], self.optimizer.bounds[1], 100)  
                y = np.linspace(self.optimizer.bounds[0], self.optimizer.bounds[1], 100) 
                X, Y = np.meshgrid(x, y)
                Z = self.optimizer.fun((X, Y))

                generation = points[i]
                x = [point[0] for point in generation]
                y = [point[1] for point in generation]

                ax.clear()
                ax.contourf(X, Y, Z, levels=50, cmap='plasma', alpha=0.5)
                best = min(generation, key=lambda x: self.optimizer.fun(x))
                ax.plot(best[0], best[1], 'green', marker='X', markersize=10, alpha=1, label="Best fit")  # Removed f-string as 'i' is not used
                ax.annotate(f"({best[0]:.5f}, {best[1]:.5f})", (best[0], best[1]), textcoords="offset points", xytext=(5,5), ha='center')
                ax.plot(x, y, 'ro', alpha=0.6)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('Generation {}'.format(i + 1))
                ax.legend()

            # Example array of arrays of points
            points = generations

            # Initialize figure and axis
            fig, ax = plt.subplots(figsize=(10, 10))

            # Function to update plot for each frame
            def update_plot(i):
                plot_generation(points, i, ax)

            # Generate and save GIF
            current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            gif_filename = f'images/generations_{current_date}.gif'
            ani = FuncAnimation(fig, update_plot, frames=len(points), interval=100)

            # Save the GIF
            ani.save(gif_filename, writer='pillow')

            plt.close()
            print("Done!")
