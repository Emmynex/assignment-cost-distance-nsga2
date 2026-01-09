from pymoo.core.sampling import Sampling
import numpy as np
from pymoo.core.mutation import Mutation

class Permutation_Sampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return np.array([np.random.permutation(problem.n_var) for _ in range(n_samples)])


class SwapMutation(Mutation):
    def __init__(self, prob=0.2):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        X = X.copy()  # Important: work on a copy
        for k in range(X.shape[0]):
            if np.random.rand() < self.prob:
                i, j = np.random.choice(X.shape[1], 2, replace=False)
                X[k, i], X[k, j] = X[k, j], X[k, i]
        return X
