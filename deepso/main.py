import numpy as np
from tqdm import tqdm
import datetime

import logging

class Particle:
    def __init__(self, cost_function, num_dimensions, bounds, w, c1, c2, F, CR, v_max):
        self.cost_function = cost_function
        self.num_dimensions = num_dimensions
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.F = F
        self.CR = CR
        self.v_max = v_max
        self.position = np.random.uniform(bounds[0], bounds[1], num_dimensions)
        self.velocity = np.random.uniform(-self.v_max, self.v_max, num_dimensions)
        self.pbest_position = self.position.copy()
        self.pbest_score = self.cost_function(self.position)

    def error_function(particle):
        return abs(np.sqrt(np.pow(0 - self.cost_function(particle)), 2))
    
    def update(self, gbest_position, swarm):
        indices = list(range(swarm.num_particles))
        indices.remove(swarm.particles.index(self))
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = swarm.particles[a].position + self.F * (swarm.particles[b].position - swarm.particles[c].position)
        
        mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
        trial = np.where(np.random.uniform(size=self.num_dimensions) < self.CR, mutant, self.position)
        
        # trial_score = self.cost_function(trial)
        self.position = trial
        # if trial_score < self.cost_function(self.position):
        #     self.position = trial
        #     if trial_score < self.pbest_score:
        #         self.pbest_score = trial_score
        #         self.pbest_position = trial
        
        
        r1, r2 = np.random.uniform(size=2)
        self.velocity = (self.w * self.velocity +
                         self.c1 * r1 * (self.pbest_position - self.position) +
                         self.c2 * r2 * (gbest_position - self.position))
        
        self.velocity = np.clip(self.velocity, -self.v_max, self.v_max)
        
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
        
        score = self.cost_function(self.position)
        if score < self.pbest_score:
            self.pbest_score = score
            self.pbest_position = self.position
        return score

class Swarm:
    def __init__(self, cost_function, num_particles, num_dimensions, num_iterations, bounds, w, c1, c2, F, CR, v_max):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.num_iterations = num_iterations
        self.bounds = bounds
        self.particles = [Particle(cost_function, num_dimensions, bounds, w, c1, c2, F, CR, v_max) for _ in range(num_particles)]
        self.gbest_position = min(self.particles, key=lambda p: p.pbest_score).pbest_position
        self.gbest_score = min(p.pbest_score for p in self.particles)
    
    def optimize(self):
        for iteration in range(self.num_iterations):
            for particle in self.particles:
                score = particle.update(self.gbest_position, self)
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = particle.position
        
            # print(f"Global Best Score: {self.gbest_score}")
            # print(f"Global Best Position: {self.gbest_position}")
        return self.gbest_score, self.gbest_position

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
    def __init__(self, cost_function, num_dimensions=2, num_particles=30, num_iterations=100, w=0.5, c1=2.0, c2=2.0, F=0.5, CR=0.9, v_max=1, bounds=(-2.048, 2.048)):
        self.swarm = Swarm(cost_function, num_particles, num_dimensions, num_iterations, bounds, w, c1, c2, F, CR, v_max)

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

def rosenbrock(x, a=1, b=100):
    return sum((a-x[i])**2 + b*(x[i+1]-x[i]**2)**2 for i in range(len(x)-1))

def optimize_deepso(params):
    w, c1, c2, F, CR, v_max = params
    num_dimensions = 30
    num_particles = 30
    num_iterations = 100
    deepso = DEEPSO(rosenbrock, num_dimensions, num_particles, num_iterations, w, c1, c2, F, CR, v_max, bounds=(-2.048, 2.048))
    cost, _ = deepso.run()
    return cost


# deepso = DEEPSO(optimize_deepso, 6, 30, 100, 0.21782937, 0.11017007, 1.1200412, 1.01800311, bounds=(0, 2))
# best_cost, best_params = deepso.optimize_params()
# print(f'Best parameters: {best_params}')
# print(f'Best cost: {best_cost}')


# best for now
w,c1,c2,F,CR,v_max = [0.31419906, 1.47620838, 1.2800813,  0.36238664, 0.56160154, 0.35317888] #21 +avg -std

# w,c1,c2,F,CR,v_max = [0.31419906, 1.07620838, 1.3800813,  0.36238664, 0.56160154, 0.35317888] #21 -avg +std

# w,c1,c2,F,CR,v_max = [0.45203125, 0.9019769, 0.84886763, 0.83087821, 0.45429264, 1.88688463] #23

dim = 30
particles = dim
iters = 3333
best_vec = []
for _ in tqdm(range(30)):
    deepso = DEEPSO(rosenbrock, dim, particles, iters, w, c1, c2, F, CR, v_max, bounds=(-2.048, 2.048))
    best_cost, best_params = deepso.run()
    print(f'Best parameters: {best_params}')
    print(f'Best cost: {best_cost}')
    best_vec.append(best_cost)

print(f'avg: {np.mean(best_vec)}')
print(f'std: {np.std(best_vec)}')