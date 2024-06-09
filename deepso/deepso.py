import numpy as np
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
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
        self.pbest_position = self.position.copy()
        self.pbest_score = self.cost_function(self.position)
        self.alpha = 0.2

    def update(self, gbest_position, swarm):
        
        indices = list(range(swarm.num_particles))
        indices.remove(swarm.particles.index(self))
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = swarm.particles[a].position + self.F * (gbest_position - swarm.particles[c].position)
        
        mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
        
        # trial vec
        self.position = np.where(np.random.uniform(size=self.num_dimensions) < self.CR, mutant, self.position) 
        
        r1, r2, r3 = np.random.uniform(size=3)
        self.velocity = (self.w * self.velocity +
                             self.c1 * r1 * (self.pbest_position - self.position) +
                             self.c2 * r2 * (gbest_position - self.position) + 
                             self.c3 * r3 * (gbest_position - self.pbest_position)) # delta weight
        
        
        self.velocity = np.clip(self.velocity, -self.v_max, self.v_max)
        

        self.position = self.position + self.velocity
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        
        score = self.cost_function(self.position)

        if score < self.pbest_score:
            self.pbest_score = score
            self.pbest_position = self.position
        
        return score


class Swarm:
    def __init__(self, cost_function, num_particles, num_dimensions, num_iterations, bounds, w, c1, c2, c3, F, CR, v_max):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.num_iterations = num_iterations
        self.bounds = bounds
        self.particles = [Particle(cost_function, num_dimensions, bounds, w, c1, c2, c3, F, CR, v_max) for _ in range(num_particles)]
        self.gbest_position = min(self.particles, key=lambda p: p.pbest_score).pbest_position
        self.gbest_score = min(p.pbest_score for p in self.particles)
    
    def optimize(self):
        bests = []
        for iteration in tqdm(range(self.num_iterations)):
            for particle in self.particles:
                score = particle.update(self.gbest_position, self)
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = particle.position
            bests.append(self.gbest_score)
        
            # print(f"Global Best Score: {self.gbest_score}")
            # print(f"Global Best Position: {self.gbest_position}")
        return self.gbest_score, self.gbest_position, bests

    def optimize_params(self, logger):
        print("Params optimization starting...")
        print(f"Current best score: {self.gbest_score}")
        
        for iteration in range(self.num_iterations):
            is_improved = False
            
            for particle in self.particles:
                score = particle.update(self.gbest_position, self)
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = particle.position
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

        # Join the log messages into a single string and log it
        full_message = "\n".join(log_message)
        logger.info(full_message)

        # Optionally, also print to console if desired
        print(full_message)


class DEEPSO:
    def __init__(self, cost_function, num_dimensions=2, num_particles=30, num_iterations=100, w=0.5, c1=2.0, c2=2.0, c3=0.85,F=0.5, CR=0.9, v_max=1, bounds=(-2.048, 2.048)):
        self.swarm = Swarm(cost_function, num_particles, num_dimensions, num_iterations, bounds, w, c1, c2, c3, F, CR, v_max)

    def run(self):
        return self.swarm.optimize()

    def optimize_params(self):
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # Configure logging
        logging.basicConfig(
            filename=f'logs/results{current_time}.log',        # Name of the log file
            level=logging.INFO,           # Set the logging level to INFO
            format='%(message)s'          # Use a simple format for the log messages
        )
        logger = logging.getLogger(__name__)  # Get the logger object
        return self.swarm.optimize_params(logger)