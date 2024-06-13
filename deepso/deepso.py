import numpy as np
from tqdm import tqdm
import datetime
import logging


class Particle:
    def __init__(self, cost_function, num_dimensions, bounds, w, c1, c2, c3, F, CR, v_max):
        self.cost_function = cost_function
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.F = F
        self.CR = CR
        self.v_max = v_max
        self.position = np.random.uniform(bounds[0], bounds[1], num_dimensions)
        self.velocity = np.random.uniform(-self.v_max, self.v_max, num_dimensions)
        self.score = self.cost_function(self.position)
        self.pbest_position = self.position.copy()
        self.pbest_score = self.score

    def update(self, gbest_position, swarm, it):
        indices = list(range(swarm.num_particles))
        indices.remove(swarm.particles.index(self))
        a, b = np.random.choice(indices, 2, replace=False)
        mutant = swarm.particles[a].position + self.F * (gbest_position - swarm.particles[b].position)
        
        mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

        # trial vector
        self.position = np.where(np.random.uniform(size=self.num_dimensions) < self.CR, mutant, self.position)

        r1, r2, r3 = np.random.uniform(size=3)
        self.velocity = (self.w * self.velocity +
                         self.c1 * r1 * (self.pbest_position - self.position) +
                         self.c2 * r2 * (gbest_position - self.position) + 
                         self.c3 * r3 * (gbest_position - self.pbest_position))

        
        self.velocity = np.clip(self.velocity, -self.v_max, self.v_max)
        
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        
        score = self.cost_function(self.position)

        if score < self.pbest_score:
            self.pbest_score = score
            self.pbest_position = self.position.copy()
        else:
            self.position = np.where(np.random.uniform(size=self.num_dimensions) < self.CR, self.position, self.pbest_position)

        return score


class Swarm:
    def __init__(self, cost_function, num_particles, num_dimensions, num_iterations, bounds, w, c1, c2, c3, F, CR, v_max, verb=False):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.num_iterations = num_iterations
        self.cost_function = cost_function
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.F = F
        self.CR = CR
        self.v_max = v_max
        self.f_evals = 0
        self.max_f_evals = 100_000
        self.particles = [Particle(self.cost_function_eval, num_dimensions, bounds, w, c1, c2, c3, F, CR, v_max) for _ in range(num_particles)]
        self.gbest_position = min(self.particles, key=lambda p: p.pbest_score).pbest_position
        self.gbest_score = min(p.pbest_score for p in self.particles)
        self.verb = verb


    class Result:
        def __init__(self, gbest_score, gbest_position, f_evals, iters, bests):
            self.F = gbest_score
            self.X = gbest_position
            self.evals = f_evals
            self.max_iters = iters
            self.history = bests


    def tournament(self, tournament_size=2):
        # Sort particles based on a suitable criterion (e.g., score)
        selected_tournament = sorted(self.particles, key=lambda p: p.score)[:16]
        selected_parents = []

        # Ensure selected_tournament is a numpy array for np.random.choice
        selected_tournament = np.array(selected_tournament)

        for _ in range(len(selected_tournament)):
            # Ensure tournament size is not larger than the number of selected_tournament
            participants = np.random.choice(selected_tournament, tournament_size, replace=False)
            
            # Find the winner among the participants based on score
            winner = min(participants, key=lambda p: p.score)
            selected_parents.append(winner)
        
        return selected_parents

    def update_swarm(self, offspring):
        # Replace worst particles with offspring (or merge them)
        combined_swarm = self.particles + offspring
        combined_swarm.sort(key=lambda p: p.score)
        self.particles = combined_swarm[:len(self.particles)]

    def crosspoint(self, p1, p2):
        # Example single-point crossover
        child1 = Particle(self.cost_function_eval, self.num_dimensions, self.bounds, self.w, self.c1, self.c2, self.c3, self.F, self.CR, self.v_max)
        pos = np.where(np.random.uniform(size=self.num_dimensions) < self.CR, p1, p2)
        vel = np.mean([p1.velocity, p2.velocity], axis=0)
        child1.position = pos
        child1.velocity = vel
        return child1

    def crossover(self, selected_parents):
        offspring = []
        for i in range(0, len(selected_parents), 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            child1 = self.crosspoint(parent1, parent2)
            offspring.append(child1)
        return offspring

    def cost_function_eval(self, x):
        self.f_evals += 1
        return self.cost_function(x)

    def optimize(self):
        bests = []
        k = 0
        if self.verb:
            print(f'f_evals | gb')
        while self.f_evals < self.max_f_evals:
            for particle in self.particles:
                if self.f_evals >= self.max_f_evals:
                    break
                score = particle.update(self.gbest_position, self, k)
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = particle.position.copy()
            # parents = self.tournament()
            # children = self.crossover(parents)
            # self.update_swarm(children)
            k += 1
            if self.verb:
                if self.f_evals % 100 == 0:
                    print(f'{self.f_evals}| {self.gbest_score}')
            if self.verb:
                print(f'X: {self.gbest_position}')
                print(f'F: {self.gbest_score}')

        return self.Result(self.gbest_score, self.gbest_position, self.f_evals, k, bests)


    def optimize_params(self, logger):
        print("Params optimization starting...")
        print(f"Current best score: {self.gbest_score}")
        
        for iteration in range(self.num_iterations):
            is_improved = False
            
            for particle in self.particles:
                if self.f_evals >= self.max_f_evals:
                    return self.gbest_score, self.gbest_position
                score = particle.update(self.gbest_position, self, iteration)
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = particle.position.copy()
                    is_improved = True

            
            self.print_info(is_improved, iteration, logger)
        
        return self.gbest_score, self.gbest_position

    def print_info(self, is_improved, iteration, logger):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = []

        if is_improved:
            log_message.append("\n")
            log_message.append(f"Iteration [{iteration}]: {current_time}")
            log_message.append(f"Best score improved: *--> {self.gbest_score} <--*")
            log_message.append(f"Best params: {self.gbest_position}")
        else:
            log_message.append("\n")
            log_message.append(f"Iteration [{iteration}]: {current_time}")
            log_message.append(f"Best score: {self.gbest_score}")
            log_message.append(f"Best params: {self.gbest_position}")

        full_message = "\n".join(log_message)
        logger.info(full_message)
        print(full_message)


class DEEPSO:
    def __init__(self, cost_function, num_dimensions=2, num_particles=30, num_iterations=100, w=0.5, c1=2.0, c2=2.0, c3=0.85, F=0.5, CR=0.9, v_max=1, bounds=(-2.048, 2.048), verb=False):
        self.swarm = Swarm(cost_function, num_particles, num_dimensions, num_iterations, bounds, w, c1, c2, c3, F, CR, v_max, verb)

    def run(self):
        print("Minimizing...")
        print(f"Max # of function evaluations: {self.swarm.max_f_evals}")
        return self.swarm.optimize()

    def optimize_params(self):
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logging.basicConfig(
            filename=f'logs/results{current_time}.log',
            level=logging.INFO,
            format='%(message)s'
        )
        logger = logging.getLogger(__name__)
        return self.swarm.optimize_params(logger)
