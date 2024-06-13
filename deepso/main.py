from deepso import DEEPSO
import numpy as np
from tqdm import tqdm
# import deepso2


def rosenbrock(x, a=1, b=100):
    return sum((a-x[i])**2 + b*(x[i+1]-x[i]**2)**2 for i in range(len(x)-1))

def rastrigin(x, a=10):
    return a * len(x) + np.sum(x**2 - a * np.cos(2 * np.pi * x))

def optimize_deepso(params):
    w, c1, c2, c3, F, CR, v_max = params
    num_dimensions = 30
    num_particles = 30
    num_iterations = 100
    deepso = DEEPSO(rosenbrock, num_dimensions, num_particles, num_iterations, w, c1, c2, c3, F, CR, v_max, bounds=(-2.048, 2.048))
    cost, _, _ = deepso.run()
    return cost

def run_rounds():
    w,c1,c2,c3,F,CR,v_max = [0.31419906, 1.47620838, 1.2800813, 0.85, 0.36238664, 0.56160154, 0.35317888] #21 +avg -std
    dim = 30
    particles = dim if dim > 30 else 30
    iters = int(np.round(100_000/particles))

    best_vec = []
    for _ in tqdm(range(30)):
        deepso = DEEPSO(rosenbrock, dim, particles, iters, w, c1, c2, c3, F, CR, v_max, bounds=(-2.048, 2.048))
        res = deepso.run()
        print(f'Best parameters: {res.X}')
        print(f'Best cost: {res.F}')
        best_vec.append(res.F)

    print('30 dim:')
    print(f'avg: {np.mean(best_vec)}')
    print(f'std-dv: {np.std(best_vec)}')

    w,c1,c2,c3,F,CR,v_max = [0.31419906, 1.47620838, 1.2800813, 0.85, 0.36238664, 0.56160154, 0.35317888] #21 +avg -std

    dim = 50
    particles = dim if dim > 30 else 30
    iters = int(np.round(100_000/particles))


    best_vec = []
    for _ in tqdm(range(30)):
        deepso = DEEPSO(rosenbrock, dim, particles, iters, w, c1, c2, c3, F, CR, v_max, bounds=(-2.048, 2.048))
        res = deepso.run()
        print(f'Best parameters: {res.X}')
        print(f'Best cost: {res.F}')
        best_vec.append(res.F)

    print('50 dim')
    print(f'avg: {np.mean(best_vec)}')
    print(f'std-dv: {np.std(best_vec)}')

    w,c1,c2,c3,F,CR,v_max = [0.31419906, 1.47620838, 1.2800813, 0.85, 0.36238664, 0.56160154, 0.35317888] #21 +avg -std

    dim = 100
    particles = dim if dim > 30 else 30
    iters = int(np.round(100_000/particles))

    best_vec = []
    for _ in tqdm(range(30)):
        deepso = DEEPSO(rosenbrock, dim, particles, iters, w, c1, c2, c3, F, CR, v_max, bounds=(-2.048, 2.048))
        res = deepso.run()
        print(f'Best parameters: {res.X}')
        print(f'Best cost: {res.F}')
        best_vec.append(res.F)

    print('100 dim')
    print(f'avg: {np.mean(best_vec)}')
    print(f'std-dv: {np.std(best_vec)}')


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


def run_single():
    # best for now
    w,c1,c2,c3,F,CR,v_max = [0.31419906, 1.47620838, 1.2800813, 0.85, 0.36238664, 0.56160154, 0.35317888] #21 +avg -std
    # w,c1,c2,c3,F,CR,v_max= np.array([ 0.35507922, -0.57976092, 1.60272123, -0.84557449, 0.8177705, 0.32162406, 1.65246025]) 
    dim = 30
    particles = dim if dim > 30 else 30
    iters = 10
    deepso = DEEPSO(rosenbrock, dim, particles, iters, w, c1, c2, c3, F, CR, v_max, (-2.048, 2.048))
    res = deepso.run()
    print(f'Best parameters: {res.X}')
    print(f'Best cost: {res.F}')

run_single()
# run_rounds()