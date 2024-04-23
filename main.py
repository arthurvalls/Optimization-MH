from ga import GeneticAlgorithm

def rosenbrock(X, a=1, b=100):
    return sum(a * (X[i+1] - X[i]**2)**2 + b * (X[i] - 1)**2 for i in range(len(X)-1))

def main():
    BOUNDS = [-5., 5.]
    POP_SIZE = 100
    MUTATION_RATE = 0.03
    MUTATION_SIZE = 0.1
    RECOMBINATION_RATE = 0.6
    DIM = 2
    MAX_ITER = 1000
    ga = GeneticAlgorithm(rosenbrock, BOUNDS, POP_SIZE, MAX_ITER, MUTATION_RATE, MUTATION_SIZE, RECOMBINATION_RATE, DIM)
    start = ga.generate_population()

    # best, pop, iters = ga.evolve(start, rosenbrock, 1000)
    # print(f'Best: {best}')
    # print(f'Best value: {rosenbrock(best)}')

    ga.simulate(15)

if __name__ == '__main__':
    main()