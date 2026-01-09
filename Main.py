from pymoo.optimize import minimize
from ReadText import load_data
from Optimizer import Objectives, Assignment_Problem
from pymoo.termination import get_termination
import numpy as np
from Plotter import  plot

# Load the john beasly dataset which is the cost matrix.
cost_matrix = load_data('data\\assign100.txt')
print("costmatrix", cost_matrix)
# Create objectives instance
objective = Objectives(100)

# Generate distance matrix - second objective
distancematrix = objective.distanceMatrix()
print("distancematrix", distancematrix)
# Define problem -
problem_def = Assignment_Problem(cost_matrix, distancematrix)

print("problem_def", problem_def)
#plotting part the conflict
result = []
plt = plot(result)
plt.conflict_plot(cost_matrix, distancematrix)

# Create NSGA-II algorithm
algorithm = problem_def.NSGAII_Algorithm(100)
print("algorithm", algorithm)
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
#plot the pareto front
print("result", result)
plt = plot(result)
plt.plotPareto()

print("Pareto-optimal solutions found:", result.F.shape[0])
print("First assignment:", result.X[0])
print("Cost, Distance:", result.F[0])



# Print statistics
print("\n=== Pareto Front Statistics ===")
print(f"Number of solutions: {result.F.shape[0]}")
print(f"Cost range: [{result.F[:, 0].min():.2f}, {result.F[:, 0].max():.2f}]")
print(f"Distance range: [{result.F[:, 1].min():.2f}, {result.F[:, 1].max():.2f}]")
print(f"\nBest cost solution: Cost={result.F[:, 0].min():.2f}, Distance={result.F[result.F[:, 0].argmin(), 1]:.2f}")
print(f"Best distance solution: Cost={result.F[result.F[:, 1].argmin(), 0]:.2f}, Distance={result.F[:, 1].min():.2f}")