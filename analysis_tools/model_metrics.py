import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ModelMetricsBarChart:
    def __init__(self, models, metrics, scores):
        """
        Initialize the bar chart class.

        Parameters:
        - models (list): List of model names.
        - metrics (list): List of metrics.
        - scores (list of lists): List of scores corresponding to each model.
        """
        self.models = models
        self.metrics = metrics
        self.scores = np.array(scores)  # Convert the list to a NumPy array for easier manipulation

    def plot_and_save(self, metric_name):
        """
        Plot the bar chart and save it as a file.

        Parameters:
        - metric_name (str): The name to be used in the saved file name.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        index = np.arange(len(self.models))
        bar_width = 0.25
        opacity = 0.8

        # Plot bars for each metric
        for i, metric in enumerate(self.metrics):
            ax.bar(index + i * bar_width, self.scores[:, i], bar_width, alpha=opacity, label=metric)

        # Set chart title and labels
        ax.set_xlabel('Models', fontsize=15)
        ax.set_ylabel('Scores', fontsize=15)
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(self.models, fontsize=15)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis='y', labelsize=15)

        # Bold the tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # Add legend
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(self.metrics), frameon=False, fontsize=15)

        # Save the chart and display
        plt.savefig(f'{metric_name}_performance.pdf', format='pdf', dpi=300)
        plt.show()

    def export_to_csv(self, csv_filename):
        """
        Export the model metrics and scores to a CSV file.

        Parameters:
        - csv_filename (str): The base name for the CSV file.
        """
        # Create DataFrame
        data_dict = {'Model': np.repeat(self.models, len(self.metrics))}
        expanded_metrics = self.metrics * len(self.models)  # Repeat metric names to match the expanded models
        data_dict.update({
            'Metric': expanded_metrics,
            'Score': self.scores.flatten()  # Flatten the multi-dimensional array into a 1D array
        })

        df = pd.DataFrame(data_dict)
        df.to_csv(f'{csv_filename}_performance.csv', index=False)
