from pymoo.core.sampling import Sampling
import numpy as np
from pymoo.core.mutation import Mutation

class Permutation_Sampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return np.array([np.random.permutation(problem.n_var) for _ in range(n_samples)])



