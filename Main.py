from pymoo.optimize import minimize
from ReadText import load_data
from Optimizer import Objectives, Assignment_Problem
from pymoo.termination import get_termination
import numpy as np


# Load the john beasly dataset which is the cost matrix.
cost_matrix = load_data('data\\assign100.txt')

# Create objectives instance
objective = Objectives(100)

# Generate distance matrix - second objective
distancematrix = objective.distanceMatrix()
