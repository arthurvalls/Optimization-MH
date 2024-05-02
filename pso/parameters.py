# hyperparameters taken from: "Good parameters for particle swarm optimization", MEH Pedersen.
# good enough for the rosenbrock function

class ParameterLoader:
	def __init__(self):
		pass

	def load_plot_params(self):
		# FOR PLOTTING
		P_C = S_C = 0.2 # coefficients
		W = 0.5 # inertia weight
		V_MAX = 0.15 # max velocity
		return P_C, S_C, W, V_MAX

	def load_2d(self):
		# FOR 2D
		P_C = 2.5586 # personal_coefficient (previously 0.2)
		S_C = 1.3358 # social_coefficient (previously 0.2)
		W = 0.3925 # inertia weight (previously 0.5)
		V_MAX = 1 # max velocity (previously 0.15) 
		return P_C, S_C, W, V_MAX

	def load_5d(self):
		# FOR 5D
		P_C = -0.7238
		S_C = 2.0289
		W = -0.3593
		V_MAX = 1
		return P_C, S_C, W, V_MAX
	
	def load_10d(self):
		# FOR 10D
		P_C = 1.6319
		S_C = 0.6239
		W = 0.6571
		V_MAX = 1
		return P_C, S_C, W, V_MAX