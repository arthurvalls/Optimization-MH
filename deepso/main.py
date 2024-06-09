from deepso import DEEPSO
import numpy as np
from tqdm import tqdm
import deepso2

def rosenbrock(x, a=1, b=100):
    return sum((a-x[i])**2 + b*(x[i+1]-x[i]**2)**2 for i in range(len(x)-1))


# def optimize_deepso(params):
#     w, c1, c2, F, CR, v_max = params
#     num_dimensions = 30
#     num_particles = 30
#     num_iterations = 100
#     deepso = DEEPSO(rosenbrock, num_dimensions, num_particles, num_iterations, w, c1, c2, F, CR, v_max, bounds=(-2.048, 2.048))
#     cost, _ = deepso.run()
#     return cost


# def optimize_deepso2(params):
#     w=0.31419906
#     F=0.36238664
#     CR=0.56160154
#     v_max=0.35317888
#     c1, c2 = params
#     num_dimensions = 30
#     num_particles = 30
#     num_iterations = 100
#     deepso = DEEPSO(rosenbrock, num_dimensions, num_particles, num_iterations, w, c1, c2, F, CR, v_max, bounds=(-2.048, 2.048))
#     cost, _ = deepso.run()
#     return cost

# best for now
w,c1,c2,c3,F,CR,v_max = [0.31419906, 1.47620838, 1.2800813, 0.85, 0.36238664, 0.56160154, 0.35317888] #21 +avg -std

dim = 30
particles = dim
iters = int(np.round(100_000/particles))

best_vec = []
for _ in tqdm(range(30)):
    deepso = DEEPSO(rosenbrock, dim, particles, iters, w, c1, c2, c3, F, CR, v_max, bounds=(-2.048, 2.048))
    best_cost, best_params, _ = deepso.run()
    print(f'Best parameters: {best_params}')
    print(f'Best cost: {best_cost}')
    best_vec.append(best_cost)


print(f'avg: {np.mean(best_vec)}')
print(f'std-dv: {np.std(best_vec)}')
# deepso = DEEPSO(rosenbrock, dim, particles, iters, w, c1, c2, c3, F, CR, v_max, bounds=(-2.048, 2.048))
# best_cost, best_params, _ = deepso.run()
# print(f'Best parameters: {best_params}')
# print(f'Best cost: {best_cost}')
