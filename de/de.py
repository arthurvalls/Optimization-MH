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

    def _mutate(self, population):
        mutated_population = np.zeros_like(population)
        idxs = list(np.arange(self.pop_size))
        for i in range(self.pop_size):
            temp_idxs = idxs[:]
            temp_idxs.remove(i)
            np.random.shuffle(temp_idxs)
            a, b, c = temp_idxs[:3]
            mutant = population[a] + self.F * (population[b] - population[c])
            mutated_population[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutated_population

    def _mutate_rand_2(self, population):
        mutated_population = np.zeros_like(population)
        idxs = list(np.arange(self.pop_size))
        for i in range(self.pop_size):
            temp_idxs = idxs[:]
            temp_idxs.remove(i)
            np.random.shuffle(temp_idxs)
            a, b, c, d, e = temp_idxs[:5]
            mutant = population[a] + self.F * (population[b] - population[c]) + self.F * (population[d] - population[e])
            mutated_population[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutated_population

    def _mutate_best_1(self, population):
        mutated_population = np.zeros_like(population)
        idxs = list(np.arange(self.pop_size))
        best = min(population, key=self.fun)
        for i in range(self.pop_size):
            temp_idxs = idxs[:]
            temp_idxs.remove(i)
            np.random.shuffle(temp_idxs)
            a, b = temp_idxs[:2]
            mutant = best + self.F * (population[a] - population[b])
            mutated_population[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutated_population

    def _mutate_best_2(self, population):
        mutated_population = np.zeros_like(population)
        idxs = list(np.arange(self.pop_size))
        best = min(population, key=self.fun)
        for i in range(self.pop_size):
            temp_idxs = idxs[:]
            temp_idxs.remove(i)
            np.random.shuffle(temp_idxs)
            a, b, c, d = temp_idxs[:4]
            mutant = best + self.F * (population[a] - population[b]) + self.F * (population[c] - population[d])
            mutated_population[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutated_population

    def _mutate_currentbest_1(self, population):
        mutated_population = np.zeros_like(population)
        idxs = list(np.arange(self.pop_size))
        best = min(population, key=self.fun)
        for i in range(self.pop_size):
            temp_idxs = idxs[:]
            temp_idxs.remove(i)
            np.random.shuffle(temp_idxs)
            a, b = temp_idxs[:2]
            mutant = population[a] + self.F * (best - population[b])
            mutated_population[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutated_population

    def _mutate_currentbest_2(self, population):
        mutated_population = np.zeros_like(population)
        idxs = list(np.arange(self.pop_size))
        best = min(population, key=self.fun)
        for i in range(self.pop_size):
            temp_idxs = idxs[:]
            temp_idxs.remove(i)
            np.random.shuffle(temp_idxs)
            a, b, c, d = temp_idxs[:4]
            mutant = population[a] + self.F * (best - population[b]) + self.F * (population[c] - population[d])
            mutated_population[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutated_population

    def _crossover(self, population, mutated_population):

        mask = np.random.uniform(size=(self.pop_size, self.dim)) < self.CR

        # é necessario garantir que o gene seja mutado?
        # genes = np.random.randint(self.dim, size=self.pop_size)
        # masks = np.random.rand(self.pop_size, self.dim) < self.CR
        # masks[np.arange(self.pop_size), genes] = True

        return np.where(mask, mutated_population, population)


    def _selection(self, population, crossover_population):
        return np.array([population[i] if self.fun(population[i]) < self.fun(crossover_population[i])
                    else crossover_population[i] for i in range(len(population))])

    def optimize(self, max_iter=100):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        generations = []
        for it in (range(max_iter)):
            mutated_population = self._mutate(population)
            crossover_population = self._crossover(population, mutated_population)
            population = self._selection(population, crossover_population)
            generations.append(population)

        best_ind = min(population, key=self.fun)
        return best_ind, self.fun(best_ind), generations


    def optimize_rand_2_bin(self, max_iter=100):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        generations = []
        for it in (range(max_iter)):
            mutated_population = self._mutate_rand_2(population)
            crossover_population = self._crossover(population, mutated_population)
            population = self._selection(population, crossover_population)
            generations.append(population)

        best_ind = min(population, key=self.fun)
        return best_ind, self.fun(best_ind), generations


    def optimize_best_1_bin(self, max_iter=100):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        generations = []
        for it in (range(max_iter)):
            mutated_population = self._mutate_best_1(population)
            crossover_population = self._crossover(population, mutated_population)
            population = self._selection(population, crossover_population)
            generations.append(population)

        best_ind = min(population, key=self.fun)
        return best_ind, self.fun(best_ind), generations

    def optimize_best_2_bin(self, max_iter=100):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        generations = []
        for it in (range(max_iter)):
            mutated_population = self._mutate_best_2(population)
            crossover_population = self._crossover(population, mutated_population)
            population = self._selection(population, crossover_population)
            generations.append(population)

        best_ind = min(population, key=self.fun)
        return best_ind, self.fun(best_ind), generations

    def optimize_currentbest_1_bin(self, max_iter=100):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        generations = []
        for it in (range(max_iter)):
            mutated_population = self._mutate_currentbest_1(population)
            crossover_population = self._crossover(population, mutated_population)
            population = self._selection(population, crossover_population)
            generations.append(population)

        best_ind = min(population, key=self.fun)
        return best_ind, self.fun(best_ind), generations

    def optimize_currentbest_2_bin(self, max_iter=100):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        generations = []
        for it in (range(max_iter)):
            mutated_population = self._mutate_currentbest_2(population)
            crossover_population = self._crossover(population, mutated_population)
            population = self._selection(population, crossover_population)
            generations.append(population)

        best_ind = min(population, key=self.fun)
        return best_ind, self.fun(best_ind), generations



class Simulator:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def plot_dims(self, dims):
        for d in dims:
            self.optimizer.dim = d
            _, _, gens = self.optimizer.optimize(1000)
            bests = []
            for gen in gens:
                b = min(gen, key=self.optimizer.fun)
                bests.append(self.optimizer.fun(b))
            plt.plot(bests, label='d={}'.format(d))

        plt.legend()
        plt.savefig(f"images/{'_'.join(map(str, dims))}.png")
        plt.show()


    def simulate(self, n_rounds):
            rand_1 = []
            for _ in tqdm(range(n_rounds)):  # Removed unnecessary parentheses
                gen_t = []  # Moved inside the loop to reset for each round
                _, _, gens = self.optimizer.optimize(100)
                for gen in gens:
                    best_fitness = self.optimizer.fun(min(gen, key=self.optimizer.fun))
                    gen_t.append(best_fitness)
                rand_1.append(gen_t)

            avg_rand_1 = np.mean(rand_1, axis=0)
            print(f"\nBest (rand_1): {avg_rand_1[-1]}")

            rand_2 = []
            for _ in tqdm(range(n_rounds)):  # Removed unnecessary parentheses
                gen_t = []  # Moved inside the loop to reset for each round
                _, _, gens = self.optimizer.optimize_rand_2_bin(100)
                for gen in gens:
                    best_fitness = self.optimizer.fun(min(gen, key=self.optimizer.fun))
                    gen_t.append(best_fitness)
                rand_2.append(gen_t)

            avg_rand_2 = np.mean(rand_2, axis=0)
            print(f"\nBest (rand_2): {avg_rand_2[-1]}")


            best_1 = []
            for _ in tqdm(range(n_rounds)):  # Removed unnecessary parentheses
                gen_t = []  # Moved inside the loop to reset for each round
                _, _, gens = self.optimizer.optimize_best_1_bin(100)
                for gen in gens:
                    best_fitness = self.optimizer.fun(min(gen, key=self.optimizer.fun))
                    gen_t.append(best_fitness)
                best_1.append(gen_t)

            avg_best_1 = np.mean(best_1, axis=0)
            print(f"\nBest (best_1): {avg_best_1[-1]}")


            best_2 = []
            for _ in tqdm(range(n_rounds)):  # Removed unnecessary parentheses
                gen_t = []  # Moved inside the loop to reset for each round
                _, _, gens = self.optimizer.optimize_best_2_bin(100)
                for gen in gens:
                    best_fitness = self.optimizer.fun(min(gen, key=self.optimizer.fun))
                    gen_t.append(best_fitness)
                best_2.append(gen_t)

            avg_best_2 = np.mean(best_2, axis=0)
            print(f"\nBest (best_2): {avg_best_2[-1]}")

            currentbest_1 = []
            for _ in tqdm(range(n_rounds)):  # Removed unnecessary parentheses
                gen_t = []  # Moved inside the loop to reset for each round
                _, _, gens = self.optimizer.optimize_currentbest_1_bin(100)
                for gen in gens:
                    best_fitness = self.optimizer.fun(min(gen, key=self.optimizer.fun))
                    gen_t.append(best_fitness)
                currentbest_1.append(gen_t)

            avg_currentbest_1 = np.mean(currentbest_1, axis=0)
            print(f"\nBest (currentbest_1): {avg_currentbest_1[-1]}")

            currentbest_2 = []
            for _ in tqdm(range(n_rounds)):  # Removed unnecessary parentheses
                gen_t = []  # Moved inside the loop to reset for each round
                _, _, gens = self.optimizer.optimize_currentbest_2_bin(100)
                for gen in gens:
                    best_fitness = self.optimizer.fun(min(gen, key=self.optimizer.fun))
                    gen_t.append(best_fitness)
                currentbest_2.append(gen_t)

            avg_currentbest_2 = np.mean(currentbest_2, axis=0)
            print(f"\nBest (currentbest_2): {avg_currentbest_2[-1]}")

            # Plotting
            plt.figure(figsize=(12,6))
            plt.plot(avg_rand_1, linestyle="--", color="blue", label='Random 1')
            plt.plot(avg_rand_2, linestyle="-", color="green", label='Random 2')
            plt.plot(avg_best_1, linestyle=":", color="red", label='Best 1')
            plt.plot(avg_best_2, linestyle="-", color="orange", label='Best 2')
            plt.plot(avg_currentbest_1, linestyle="-.", color="purple", label='Current to Best 1')
            plt.plot(avg_currentbest_2, linestyle="-", color="black", label='Current to Best 2')
            plt.xlabel('Generation')
            plt.ylabel('Average Best Cost')
            plt.title('Average Best Cost per Generation over all Rounds')
            current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.legend()  # Added to show label
            plt.savefig(f"images/average_{current_date}.png")

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
                ax.contourf(X, Y, Z, levels=np.logspace(self.optimizer.bounds[0], self.optimizer.bounds[1], 50), cmap='plasma', alpha=0.5)
                best = min(generation, key=lambda x: self.optimizer.fun(x))
                ax.plot(best[0], best[1], 'green', marker='X', markersize=10, alpha=1, label="Best fit")  # Removed f-string as 'i' is not used
                ax.annotate(f"({best[0]:.5f}, {best[1]:.5f})", (best[0], best[1]), textcoords="offset points", xytext=(5,5), ha='center')
                ax.plot(x, y, 'ro', alpha=0.6)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('Generation {}'.format(i + 1))
                ax.legend()

            # Example array of arrays of points
            points = generations[:100]

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
