import numpy as np
import random
from tqdm import tqdm
import time

class GeneticAlgorithm:
    def __init__(self, function, bounds=(-5., 5.), pop_size=100, max_iter=100, mutation_rate=0.03, mutation_size=0.1,
                 recombination_rate=0.6, dim=2):
        self.function = function
        self.bounds = bounds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.mutation_size = mutation_size
        self.recombination_rate = recombination_rate
        self.dim = dim
        self.selected_count = int(np.ceil(pop_size * recombination_rate))


    def generate_population(self):
        p = [np.array([np.random.uniform(self.bounds[0], self.bounds[1]) for _ in range(self.dim)]) 
             for _ in range(self.pop_size)]
        return p

    def tournament_selection(self, population, tournament_size=2):
        chosen_ones = []

        while len(population) >= tournament_size:
            current_round_indices = random.sample(range(len(population)), k=tournament_size)
            current_round = [population[i] for i in current_round_indices]

            winner = min(current_round, key=lambda x: self.function(x))
            chosen_ones.append(winner)

            selected_indices = set(current_round_indices)
            population = [pop for i, pop in enumerate(population) if i not in selected_indices]

        return chosen_ones

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

            # ELITISM
            population = self.elitism(population)

            k += 1

        # CURRENT BEST
        best_individual = min(population, key=lambda x: self.function(x))
        print(f'\n{k}: {self.function(best_individual)}')

        return best_individual, population, k


    def simulate(self, number_of_simulations):

        avg_iters = 0
        avg_times = 0
        best_val = np.inf
        best_val_iter = 0
        avg_values = []

        for _ in range(number_of_simulations):
            p = self.generate_population()
            start_time = time.time()
            best, pop, iters = self.evolve(p)
            avg_times += (time.time() - start_time)
            avg_iters += iters
            if self.function(best) < best_val:
                best_val = self.function(best)
                avg_values.append(best_val)
                best_val_iter = iters

        print(f'Best value found at {best_val_iter}: {best_val}')
        print(f'Average values found: {np.mean(avg_values)}')
        print(f'Average iters taken: {avg_iters/number_of_simulations}')
        print(f'Average time taken: {avg_times/number_of_simulations}')

