import math
import numpy as np
from scipy.spatial.distance import cdist
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.selection.tournament import TournamentSelection

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

