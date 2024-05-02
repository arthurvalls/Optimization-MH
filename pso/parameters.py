# hyperparameters taken from: "Good parameters for particle swarm optimization", MEH Pedersen.
# good enough for the rosenbrock function

class ParameterLoader:
    def __init__(self, dims):
        self.dims = dims

    def load_params(self):
        if self.dims == 2:
            return self.load_2d()
        elif self.dims == 5:
            return self.load_5d()
        elif self.dims == 10:
            return self.load_10d()
        else:
            P_C = S_C = 0.2
            W = 0.5
            V_MAX = 0.15
            return P_C, S_C, W, V_MAX

    def load_2d(self):
        P_C = 2.5586
        S_C = 1.3358
        W = 0.3925
        V_MAX = 1
        return P_C, S_C, W, V_MAX

    def load_5d(self):
        P_C = -0.7238
        S_C = 2.0289
        W = -0.3593
        V_MAX = 1
        return P_C, S_C, W, V_MAX

    def load_10d(self):
        P_C = 1.6319
        S_C = 0.6239
        W = 0.6571
        V_MAX = 1
        return P_C, S_C, W, V_MAX
