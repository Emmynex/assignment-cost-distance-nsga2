import math
import numpy as np
from scipy.spatial.distance import cdist
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.selection.tournament import TournamentSelection
from CrossOverOperator import PMXCrossover
from SwapMutationOperator import SwapMutation
from Permutation__Sampling import Permutation_Sampling

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class Objectives:
    def __init__(self, n):
        self.n = n

    def distanceMatrix(self):
        variable_size = math.sqrt(self.n)
        if(variable_size % 10 > 0):
            variable_size = math.ceil(variable_size)

        coordinates = []
        for i in range(self.n):
            # convert the number to a grid view- coordinate system
            x = i % variable_size
            y = i // variable_size
            coordinates.append((x, y))
        # calculate the distance btw coordinates
        Dist = cdist(coordinates, coordinates, metric='euclidean')
        return Dist


class Assignment_Problem(Problem):
    def __init__(self, costmatrix, distancematrix):
        self.Cost = costmatrix
        self.Distance = distancematrix
        n_var = costmatrix.shape[0]
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, type_var=int)

    def _evaluate(self, X, out, **kwargs):
        costs = []
        distances = []

        for solution in X:
            total_cost = sum(self.Cost[i, solution[i]] for i in range(self.n_var))
            total_distance = sum(self.Distance[i, solution[i]] for i in range(self.n_var))
            costs.append(total_cost)
            distances.append(total_distance)
        out["F"] = np.column_stack([costs, distances])

    def NSGAII_Algorithm(self, size=100):
        algorithmResponse = NSGA2(
            pop_size=size,
            sampling=Permutation_Sampling(),
            selection=TournamentSelection(func_comp=dominance_comp),
            crossover=PMXCrossover(prob=1.0),
            mutation=SwapMutation(prob=0.2),
            eliminate_duplicates=True
        )
        return algorithmResponse


# This function performs pairwise Pareto dominance comparisons between individuals in a population by evaluating their objective vectors and returns, for each comparison, whether the first individual dominates the second, the second dominates the first, or neither dominates the otheR
def dominance_comp(pop, P, **kwargs):

    if P.ndim == 1:
        P = P.reshape(-1, 2)

    n_comparisons = P.shape[0]
    result = np.zeros(n_comparisons, dtype=int)

    F = pop.get("F")

    for i in range(n_comparisons):
        a_idx = P[i, 0]
        b_idx = P[i, 1]

        a_obj = F[a_idx]
        b_obj = F[b_idx]

        if a_obj is None or b_obj is None:
            result[i] = 0
            continue

        # Check if a dominates b
        a_better = np.all(a_obj <= b_obj) and np.any(a_obj < b_obj)
        b_better = np.all(b_obj <= a_obj) and np.any(b_obj < a_obj)

        if a_better:
            result[i] = -1  # a dominates b
        elif b_better:
            result[i] = 1  # b dominates a
        else:
            result[i] = 0  # non-dominated

    return result