# Plot the Pareto front
import matplotlib.pyplot as plt
import seaborn as sns
class plot:
    def __init__(self, result):
        self.result = result

    def plotPareto(self):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            self.result.F[:, 0],
            self.result.F[:, 1],
            c='green',
            alpha=0.4,
            s=50,
            edgecolors='black'
        )

        ax.set_xlabel('Total Cost', fontsize=12)
        ax.set_ylabel('Total Distance', fontsize=12)
        ax.set_title(
            'Pareto Front - Multi-Objective Assignment Problem',
            fontsize=14,
            fontweight='bold'
        )

        ax.grid(True, alpha=0.3)

        # Caption BELOW the figure
        fig.text(
            0.5,  # x-position (centered)
            -0.08,  # y-position (below the plot)
            'Figure 1: Pareto-optimal trade-off between total cost and total distance obtained using NSGA-II.',
            ha='center',
            fontsize=11
        )

        plt.tight_layout()
        plt.show()

        # Plot convergence over generations if history is available
        if self.result.history:
            numberofevals = []
            costs_ = []
            distances_ = []

            for value in self.result.history:
                numberofevals.append(value.evaluator.n_eval)
                # Get the best (minimum) values for each objective
                costs_.append(value.opt.get("F")[:, 0].min())
                distances_.append(value.opt.get("F")[:, 1].min())

            # Single figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot both objectives on the same axes
            ax.plot(numberofevals, costs_, 'r-o', linewidth=1, markersize=2, label='Cost')
            ax.plot(numberofevals, distances_, 'g-s', linewidth=1, markersize=2, label='Distance')

            ax.set_xlabel('Number of Evaluations', fontsize=12)
            ax.set_ylabel('Objective Value', fontsize=12)
            ax.set_title('Convergence of Objectives', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)

            plt.show()

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