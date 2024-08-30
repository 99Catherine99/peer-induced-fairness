import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


class DiscriminationBarChart:
    def __init__(self, dfs, line_dfs, detailed=False):
        """
        Initialize the DiscriminationBarChart class.

        Parameters:
        - dfs (list of DataFrames): List of DataFrames for bar chart data.
        - line_dfs (list of DataFrames): List of DataFrames for line plot data.
        - detailed (bool): Flag to determine the level of detail for the categories.
        """
        self.dfs = dfs
        self.line_dfs = line_dfs
        self.as_binary_y0 = []
        self.line_means = []
        self.line_errors = []
        if detailed:
            self.categories = ['ED', 'SD', 'FT', 'SP', 'EP']
            self.colors = ['#B02425', '#F09BA0', '#4485C7', '#EAB883', '#E7724F']
            self.title = 'detailed_outcomes_in_discrimination'
        else:
            self.categories = ['ED', 'FT', 'EP']
            self.colors = ['#B02425', '#4485C7', '#E7724F']
            self.title = 'outcomes_in_discrimination'

    def extract_data(self):
        """
        Extract the relevant data for plotting from the provided DataFrames.
        """
        for df in self.dfs:
            self.as_binary_y0.extend(df[(df['Type'] == 'Protect') & (df['Binary Y'] == 0)]['Proportion'].tolist())

        for line_df in self.line_dfs:
            self.line_means.append(line_df['Proportion 0'].mean())
            self.line_errors.append(line_df['Proportion 0'].std())

        data = {
            'Category': self.categories,
            'As Rejected': self.as_binary_y0
        }
        df = pd.DataFrame(data)
        df.to_csv(f'{self.title}.csv', index=False)

        # Print the difference between each bar and the line plot means
        differences = [self.as_binary_y0[i] - self.line_means[i] for i in range(len(self.categories))]
        for category, diff in zip(self.categories, differences):
            print(f"Difference for {category}: {diff}")
        return df

    def plot_stacked_bar_chart(self):
        """
        Plot the stacked bar chart along with a line plot overlay.
        """
        # Set the width and position of the bars
        bar_width = 0.5
        index = np.arange(len(self.categories))

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot the 'As Rejected' bars
        for i in range(len(self.categories)):
            ax.bar(index[i], self.as_binary_y0[i], bar_width, label=f'Rejected micro-firms ({self.categories[i]})',
                   color=self.colors[i])

        # Plot the line chart
        ax.errorbar(index, self.line_means, yerr=self.line_errors, fmt='-o', color='black', label='Proportion 0',
                    capsize=8)

        # Set y-axis range
        ax.set_ylim(0, 1)

        # Bold the tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # Format the y-axis labels as percentages
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))

        # Add labels and title
        ax.set_xlabel('Algorithmic treatment categories', fontsize=15)
        ax.set_ylabel('Percentage', fontsize=15)
        ax.set_xticks(index)
        ax.set_xticklabels(self.categories, fontsize=15)

        # Customize the legend
        handles = []
        labels = []
        for i, category in enumerate(self.categories):
            handles.append(plt.Line2D([0], [0], color=self.colors[i], lw=4))
            labels.append(f'Rejected micro-firms ({category})')
        handles.append(plt.Line2D([0], [0], color='black', lw=2))
        labels.append('Rejected peers')

        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fontsize=9, frameon=False)

        plt.tight_layout()
        plt.savefig(f'{self.title}.pdf', format='pdf', dpi=300)
        plt.show()
