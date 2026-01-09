# Plot the Pareto front
import matplotlib.pyplot as plt
import seaborn as sns
class plot:
    def __init__(self, result):
        self.result = result


    def conflict_plot(self, cost_matrix, dist_matrix):
        C_sample = cost_matrix[:10, :10]
        D_sample = dist_matrix[:10, :10]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.heatmap(C_sample, annot=True, cmap="Reds")
        plt.title("Cost Heatmap (sample)")

        plt.subplot(1, 2, 2)
        sns.heatmap(D_sample, annot=True, cmap="Blues")
        plt.title("Distance Heatmap (sample)")

        plt.show()