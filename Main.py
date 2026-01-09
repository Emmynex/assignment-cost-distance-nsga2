from pymoo.optimize import minimize
from ReadText import load_data
from Optimizer import Objectives, Assignment_Problem
from pymoo.termination import get_termination
import numpy as np
from Plotter import  plot


# Load the john beasly dataset which is the cost matrix.
cost_matrix = load_data('data\\assign100.txt')

# Create objectives instance
objective = Objectives(100)

# Generate distance matrix - second objective
distancematrix = objective.distanceMatrix()

# Define problem -
problem_def = Assignment_Problem(cost_matrix, distancematrix)

#plotting part the conflict
result = []
plt = plot(result)
plt.conflict_plot(cost_matrix, distancematrix)


# Create NSGA-II algorithm
algorithm = problem_def.NSGAII_Algorithm(100)

# Set termination criteria
termination = get_termination("n_gen", 500)

# Run optimization
result = minimize(
    problem_def,
    algorithm,
    termination=termination,
    seed=1,
    save_history=True,
    verbose=True
)