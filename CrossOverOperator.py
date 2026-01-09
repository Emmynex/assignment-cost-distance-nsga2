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
            def pmx_fill(child, parent, start, end):
                        for i in range(start, end):
                            val = parent[i]
                            if val not in child:
                                pos = i
                                while True:
                                    val_in_child = child[pos]
                                    if val_in_child == -1:
                                        child[pos] = val
                                        break
                                    pos = np.where(parent == val_in_child)[0][0]

            pmx_fill(c1, p1, cx1, cx2)
            pmx_fill(c2, p2, cx1, cx2)
            for i in range(n_vars):
                if c1[i] == -1:
                    c1[i] = p1[i]
                if c2[i] == -1:
                    c2[i] = p2[i]

            Y[k, 0] = c1
            Y[k, 1] = c2

            # Reshape back: (n_matings, n_offspring, n_vars) -> (n_offspring, n_matings, n_vars)
        return np.swapaxes(Y, 0, 1)

