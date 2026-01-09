from pymoo.core.crossover import Crossover
import numpy as np


class PMXCrossover(Crossover):
    def __init__(self, prob=1.0):
        super().__init__(2, 2)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        # X shape is (n_parents, n_matings, n_vars)
        # We need to reshape it to (n_matings, n_parents, n_vars)
        _, n_matings, n_vars = X.shape

        # Reshape: (n_parents, n_matings, n_vars) -> (n_matings, n_parents, n_vars)
        X = np.swapaxes(X, 0, 1)

        Y = np.full((n_matings, 2, n_vars), -1, dtype=int)

        for k in range(n_matings):
            p1 = X[k, 0].copy()
            p2 = X[k, 1].copy()

            if np.random.rand() > self.prob:
                Y[k, 0] = p1
                Y[k, 1] = p2
                continue

            cx1 = np.random.randint(0, n_vars - 1)
            cx2 = np.random.randint(cx1 + 1, n_vars)

            c1 = -np.ones(n_vars, dtype=int)
            c2 = -np.ones(n_vars, dtype=int)

            c1[cx1:cx2] = p2[cx1:cx2]
            c2[cx1:cx2] = p1[cx1:cx2]

