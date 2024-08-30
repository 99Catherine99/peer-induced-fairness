import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import kurtosis

class DensityScatterPlotter:
    def __init__(self, protect_df, nonprotect_df, bootstrapped_samples):
        """
        Initialize the DensityScatterPlotter class.

        Parameters:
        - protect_df (pd.DataFrame): DataFrame containing the protected group data.
        - nonprotect_df (pd.DataFrame): DataFrame containing the non-protected group data.
        - bootstrapped_samples (pd.DataFrame): DataFrame containing bootstrapped sample information.
        """
        self.protect_df = protect_df
        self.nonprotect_df = nonprotect_df
        self.bootstrapped_samples = bootstrapped_samples

    def group_ground_truth_all(self, treatment_indices):
        """
        Find the 'control_index' by matching color_df['Treatment Index'] with bootstrapped_samples['treatment_index'],
        then look for the corresponding Binary Y in the indices of nonprotect_df,
        and calculate the counts and proportions of Binary Y being 0 and 1.

        Parameters:
        - treatment_indices (list): List of treatment indices.

        Returns:
        - pd.DataFrame: DataFrame containing counts and proportions of Binary Y being 0 and 1 for all treatment indices.
        """
        all_control_indices = []
        protect_binary_y_values = []

        for treatment_index in treatment_indices:
            control_indices = self.bootstrapped_samples.loc[
                self.bootstrapped_samples['treatment_index'] == treatment_index, 'control_index']
            all_control_indices.extend(control_indices.explode().tolist())
            protect_binary_y_values.append(self.protect_df.loc[treatment_index, 'Binary Y'])

        binary_y_values = self.nonprotect_df.loc[all_control_indices, 'Binary Y']
        count_0 = (binary_y_values == 0).sum()
        count_1 = (binary_y_values == 1).sum()
        total = len(binary_y_values)
        proportion_0 = count_0 / total
        proportion_1 = count_1 / total

        protect_count_0 = (np.array(protect_binary_y_values) == 0).sum()
        protect_count_1 = (np.array(protect_binary_y_values) == 1).sum()
        protect_total = len(protect_binary_y_values)
        protect_proportion_0 = protect_count_0 / protect_total
        protect_proportion_1 = protect_count_1 / protect_total

        data = {
            'Type': ['Nonprotect', 'Nonprotect', 'Protect', 'Protect'],
            'Binary Y': [0, 1, 0, 1],
            'Count': [count_0, count_1, protect_count_0, protect_count_1],
            'Proportion': [proportion_0, proportion_1, protect_proportion_0, protect_proportion_1]
        }

        return pd.DataFrame(data)

    def plot_group_density(self, pr_y_means, treatment_indices, index_name, bw_adjust=2):
        """
        Plot density graphs based on a set of treatment_indices.
        One plot represents the Pr(Y=1) values from protect_df for these treatment_indices,
        and the other represents the means of these treatment_indices from pr_y_means.

        Parameters:
        - treatment_indices (list): List of indices with color data points.
        - index_name (str): File name used for saving the image.
        - bw_adjust (float): Adjusts the bandwidth of the kernel density estimate.
        """
        protect_values = self.protect_df.loc[treatment_indices, 'Pr(Y=1)']

        pr_y_means_values = [np.mean(pr_y_means[treatment_index]) for treatment_index in treatment_indices]

        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        plt.figure(figsize=(5, 4))

        sns.kdeplot(protect_values, shade=False, bw_adjust=bw_adjust, color='blue', linewidth=2.5, label='Protect')
        sns.kdeplot(pr_y_means_values, shade=False, bw_adjust=bw_adjust, color='red', linewidth=2.5, label='Peers')

        plt.xlabel('Mean $Pr(\hat Y=1)$', fontsize=15)
        plt.ylabel('Density', fontsize=15)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False, fontsize=11)
        plt.tight_layout()
        plt.grid(False)
        plt.savefig(f'{index_name}_group_density.pdf', format='pdf', dpi=300)
        plt.show()
