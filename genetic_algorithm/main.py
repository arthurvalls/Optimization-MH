from ga import GeneticAlgorithm

def rosenbrock(X, a=1, b=100):
    return sum(a * (X[i+1] - X[i]**2)**2 + b * (X[i] - 1)**2 for i in range(len(X)-1))

def sphere(X):
    return X[0]**2 + X[1]**2

def main():
    BOUNDS = [-2.048, 2.048]
    POP_SIZE = 100
    MUTATION_RATE = 0.03
    MUTATION_SIZE = 0.1
    RECOMBINATION_RATE = 0.6
    DIM = 2
    MAX_ITER = 100
    ga = GeneticAlgorithm(sphere, BOUNDS, POP_SIZE, MAX_ITER, MUTATION_RATE, MUTATION_SIZE, RECOMBINATION_RATE, DIM)
    start = ga.generate_population()

    best, pop, iters, _ = ga.evolve(start)
    print(f'Best: {best}')
    print(f'Best value: {sphere(best)}')

    # Simulate 15 executions
    # ga.simulate(15)


    ga.average_curve(30)

if __name__ == '__main__':
    main()