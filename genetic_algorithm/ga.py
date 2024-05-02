import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time

class GeneticAlgorithm:
    def __init__(self, function, bounds=(-5., 5.), pop_size=100, max_iter=100, mutation_rate=0.03, mutation_size=0.1,
                 recombination_rate=0.6, dim=2, plot=True):
        self.function = function
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.mutation_size = mutation_size
        self.recombination_rate = recombination_rate
        self.dim = dim
        self.selected_count = int(np.ceil(pop_size * recombination_rate))
        self.generations = []
        self.plot = plot


    def generate_population(self):
        return [[np.random.uniform(self.bounds[0], self.bounds[1]) for _ in range(self.dim)] 
             for _ in range(self.pop_size)]

    def tournament_selection(self, population):
        np.random.shuffle(population) # shuffling one more time improved precision
                                      # with almost no computational overhead        
        return [min((population[i], population[i+1]), key=lambda x: self.function(x))
                         for i in range(0, len(population), 2)]

    def crossover(self, *parents):
        return np.mean(parents, axis=0)

    def generate_offspring(self, population):
        if len(population) % 2 != 0:
            print("Population is not even!")
        return [self.crossover(population[i], population[i + 1]) for i in range(0, len(population), 2)]

    def mutation(self, population):
        mask = np.random.uniform(0, 1, size=population.shape[0]) < self.mutation_rate
        mutation_values = np.random.uniform(-self.mutation_size, self.mutation_size, 
                                            size=(mask.sum(), population.shape[1]))
        population[mask] += mutation_values
        return population

    def select(self, population):
        np.random.shuffle(population)
        return population[:self.selected_count]

    def elitism(self, population):
        population[-1] = population[0]
        return population

    def evolve(self, starting_pop, eps=1e-7):
        population = starting_pop
        k = 0
        bests = []
        if self.plot:
            generations = []
        for i in tqdm(range(self.max_iter)):
            # SELECTION
            selected = self.select(population)

            # TOURNAMENT
            chosen_ones = self.tournament_selection(selected)

            # CROSSOVER
            offspring = self.generate_offspring(chosen_ones)

            # EXTENSION
            new_population = np.concatenate((population, offspring))

            # MUTATION
            mutated_population = self.mutation(new_population)

            # Remove the worst while keeping the population size constant
            population = sorted(mutated_population, key=lambda x: self.function(x))[:self.pop_size]

            if self.plot:
                generations.append(population)


            # ELITISM
            population = self.elitism(population)

            k += 1
            bests.append(self.function(min(population, key=lambda x: self.function(x))))            

        # BEST
        best_individual = min(population, key=lambda x: self.function(x))
        print(f'\n{k}: {self.function(best_individual)}')


        if self.plot:
            self.generations = generations
            print(f'Plotting gif in images folder...')
            self.generate_gif()
            print(f'Potting done!')

        return best_individual, population, k, bests


    def simulate(self, number_of_simulations):
        self.plot = False
        avg_iters = 0
        avg_times = 0
        best_val = np.inf
        best_val_iter = 0
        avg_values = []

        all_bests = []

        for _ in range(number_of_simulations):
            p = self.generate_population()
            start_time = time.time()
            best, pop, iters, bests = self.evolve(p)
            all_bests.append(bests)
            avg_times += (time.time() - start_time)
            avg_iters += iters
            avg_values.append(self.function(best))
            if self.function(best) < best_val:
                best_val = self.function(best)
                best_val_iter = iters

        print(f'Best value found at {best_val_iter}: {best_val}')
        print(f'Average values found: {np.mean(avg_values)}')
        print(f'Average iters taken: {avg_iters/number_of_simulations}')
        print(f'Average time taken: {avg_times/number_of_simulations}')

        return all_bests

    def generate_gif(self):
        def plot_generation(points, i, ax):
            x = np.linspace(self.bounds[0], self.bounds[1], 100)  
            y = np.linspace(self.bounds[0], self.bounds[1], 100) 
            X, Y = np.meshgrid(x, y)
            Z = self.function((X, Y))

            generation = points[i]
            x = [point[0] for point in generation]
            y = [point[1] for point in generation]

            ax.clear()
            ax.contourf(X, Y, Z, levels=50, cmap='rainbow', alpha=0.5)
            best = min(generation, key=lambda x: self.function(x))
            ax.plot(best[0], best[1], 'green', marker='X', markersize=10, alpha=1, label="Best fit")  # Removed f-string as 'i' is not used
            ax.annotate(f"({best[0]:.5f}, {best[1]:.5f})", (best[0], best[1]), textcoords="offset points", xytext=(5,5), ha='center')
            ax.plot(x, y, 'ro', alpha=0.3)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Generation {}'.format(i + 1))
            ax.legend()

        # Example array of arrays of points
        points = self.generations[:50]

        # Initialize figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))

        # Function to update plot for each frame
        def update_plot(i):
            plot_generation(points, i, ax)

        # Generate and save GIF
        gif_filename = 'images/convergence.gif'
        ani = FuncAnimation(fig, update_plot, frames=len(points), interval=250)

        # Save the GIF
        ani.save(gif_filename, writer='pillow')

        plt.close()
    

    def average_curve(self, number_of_simulations):        
        # MY IMPLEMENTATION
        my_best_per_generation = self.simulate(number_of_simulations)
        my_average_vector = np.zeros_like(my_best_per_generation[0])
        generations = range(len(my_best_per_generation))

        for rounds in my_best_per_generation:
            for i in range(len(rounds)):
                my_average_vector[i] += rounds[i]

        # Divide the sum by number_of_simulations to get the average
        my_average_vector = [x / number_of_simulations for x in my_average_vector]

        # PYMOO IMPLEMENTATION

        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.problems import get_problem
        from pymoo.core.callback import Callback
        from pymoo.optimize import minimize

        # Classe para recuperar melhor por geracao
        class MyCallback(Callback):
            def __init__(self) -> None:
                super().__init__()
                self.data["best"] = []

            def notify(self, algorithm):
                self.data["best"].append(algorithm.pop.get("F").min())


        problem = get_problem("rosenbrock", n_var=self.dim)

        algorithm = GA(pop_size=self.pop_size)

        pymoo_best_vector = []

        for i in tqdm(range(number_of_simulations)):
          res = minimize(problem,
                       algorithm,
                       ('n_gen', self.max_iter),
                       callback=MyCallback(),
                       verbose=False)

          pymoo_best_vector.append(res.algorithm.callback.data["best"])

        pymoo_average_vector = np.zeros_like(pymoo_best_vector[0])

        for rounds in pymoo_best_vector:
            for i in range(len(rounds)):
                pymoo_average_vector[i] += rounds[i]

        # Divide the sum by number_of_simulations to get the average
        pymoo_average_vector = [x / number_of_simulations for x in pymoo_average_vector]

        # Plotting the pymoo_average_vectorn curve
        plt.plot(pymoo_average_vector, label='Pymoo', color='r')
        # Plotting the median curve
        plt.plot(my_average_vector, label='Arthur', linestyle='dashed')

        plt.plot(len(my_best_per_generation[0]), my_average_vector[-1], marker='o', markersize=8, color='green', label=f'Best (Arthur): {my_average_vector[-1]}')
        plt.plot(len(pymoo_best_vector[0]), pymoo_average_vector[-1], marker='o', markersize=3, color='yellow', label=f'Best (Pymoo): {pymoo_average_vector[-1]}')
        plt.xlabel('Geração')
        plt.ylabel('Fitness média')
        plt.title('Curvas de convergência média')
        plt.legend()
        plt.savefig(f"images/average_convergence_{self.dim}_dims.png")

