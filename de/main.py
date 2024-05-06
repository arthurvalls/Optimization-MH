from de import DifferentialEvolution, Simulator

def main():
	rosenbrock = lambda x: sum([100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1)])
	pop_size = 30
	dim = 2
	bounds = [-5., 5.]
	CR=0.9
	F=0.8
	de = DifferentialEvolution(rosenbrock, dim, pop_size, bounds, CR, F)
	best_ind, best_cost, generations = de.optimize()
	print(f"Best individual: {best_ind}")
	print(f"Best cost: {best_cost}")
	sim = Simulator(de)
	# sim.plot_dims([8,16,32,64])
	#sim.generate_gif(generations)

if __name__ == "__main__":
	main()