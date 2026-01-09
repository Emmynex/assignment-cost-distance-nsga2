import numpy as np
from pymoo.core.mutation import Mutation

class SwapMutation(Mutation):
    def __init__(self, prob=0.2):
        super().__init__(1)  # 1 parent -> 1 child
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        for k in range(X.shape[0]):
            if np.random.rand() < self.prob:
                i, j = np.random.choice(X.shape[1], 2, replace=False)
                X[k, i], X[k, j] = X[k, j], X[k, i]
        return X
