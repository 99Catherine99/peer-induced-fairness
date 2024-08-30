import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis

class BeforeBootstrapping:
    def __init__(self, matched_df, nonprotect_df):
        self.matched_df = matched_df
        self.nonprotect_df = nonprotect_df

    def plot_before_bootstrapping_density(self, treatment_index, column, bw_adjust=2):
        """
        Get the corresponding control_index from matched_df based on the specified treatment_index,
        then filter the data corresponding to these indices in nonprotect_df and plot the density plot of the specified column.

        Parameters:
        - treatment_index (int): The specified treatment_index value.
        - column (str): The column name for which the density plot is to be drawn.
        - bw_adjust (float): Parameter to adjust the bandwidth.
        """
        # Get the corresponding control_index
        control_indices = self.matched_df.loc[self.matched_df['treatment_index'] == treatment_index, 'control_index']
        control_indices = control_indices.explode().unique()  # Handle list data and ensure uniqueness

        # Filter non-protected group data
        filtered_data = self.nonprotect_df.loc[control_indices]

        # Calculate kurtosis and standard deviation
        data = filtered_data[column]
        data_kurtosis = kurtosis(data)
        data_std = np.std(data)

        # Print the calculated metrics
        print(f"Kurtosis: {data_kurtosis}")
        print(f"Standard Deviation: {data_std}")

        # Plot the density plot
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        fig, ax = plt.subplots(figsize=(5, 4))  # Set the size of the figure
        ax = sns.kdeplot(data, shade=False, bw_adjust=bw_adjust, color='blue', linewidth=7)
        ax.set_xlabel('$Pr(\hat Y=1)$', fontsize=15)  # Set x-axis label
        ax.set_ylabel('Density', fontsize=15)  # Set y-axis label
        ax.set_xlim(0, 1)  # Set x-axis range

        # Bold the tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        plt.tight_layout()
        plt.grid(False)
        plt.savefig(f'{treatment_index}_before_bootstrapping.pdf', format='pdf', dpi=300)
        plt.show()
