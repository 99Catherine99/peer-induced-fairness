import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter


class DiscriminationPlotter:
    def __init__(self, dfs):
        """
        Initialize the DiscriminationPlotter class.

        Parameters:
        - dfs (list of DataFrames): List of DataFrames containing the data for analysis.
        """
        self.dfs = dfs
        self.rejected_micro_firms = []
        self.accepted_micro_firms = []

    def extract_data(self):
        """
        Extract and segregate data for rejected and accepted micro firms based on 'Binary Y' values.
        """
        for df in self.dfs:
            self.rejected_micro_firms.extend(df[df['Binary Y'] == 0]['Ratio'].tolist())
            self.accepted_micro_firms.extend(df[df['Binary Y'] == 1]['Ratio'].tolist())

    def plot_stacked_bar_chart(self):
        """
        Plot a stacked bar chart to visualize the distribution of discrimination categories for rejected and accepted micro firms.
        """
        labels = ['Extremely discriminated', 'Slightly discriminated', 'Fairly treated', 'Slightly privileged', 'Extremely privileged']
        colors = ['#B02425', '#F09BA0', '#4485C7', '#EAB883', '#E7724F']

        # Set Seaborn context
        sns.set_context("talk", rc={"lines.linewidth": 2.5})

        # Create the stacked bar chart
        fig, ax = plt.subplots(figsize=(8, 8))

        # Set the width of the bars
        bar_width = 0.2
        index = np.array([1.6, 2])  # Adjust x positions to bring the bars closer together

        # Plot the first bar (rejected micro firms)
        bottom1 = 0
        for i in range(len(labels)):
            ax.bar(index[0], self.rejected_micro_firms[i], bar_width, bottom=bottom1, color=colors[i], label=labels[i])
            bottom1 += self.rejected_micro_firms[i]

        # Plot the second bar (accepted micro firms)
        bottom2 = 0
        for i in range(len(labels)):
            ax.bar(index[1], self.accepted_micro_firms[i], bar_width, bottom=bottom2, color=colors[i])
            bottom2 += self.accepted_micro_firms[i]

        # Add labels and title
        ax.set_xlabel('Accessing finance outcomes', fontsize=15)
        ax.set_ylabel('Percentage', fontsize=15)
        ax.set_xticks(index)
        ax.set_xticklabels(['Rejected', 'Accepted'], fontsize=15)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=2, fontsize=15, frameon=False)

        # Bold the tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # Format y-axis labels as percentages
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))

        # Save the figure as a PDF file
        plt.savefig('discrimination_in_outcome.pdf', format='pdf', dpi=300)

        # Display the plot
        plt.tight_layout()
        plt.show()

    def save_data_to_csv(self):
        """
        Save the discrimination data to a CSV file.

        Returns:
        - df (DataFrame): DataFrame containing the discrimination data categorized by outcome.
        """
        categories = ['Extremely Discriminated', 'Slightly Discriminated', 'Fairly Treated', 'Slightly privileged', 'Extremely privileged']

        data = {
            'Category': categories,
            'Rejected': self.rejected_micro_firms[:len(categories)],
            'Accepted': self.accepted_micro_firms[:len(categories)]
        }
        df = pd.DataFrame(data)
        df.to_csv('discrimination_in_outcomes.csv', index=False)

        return df
