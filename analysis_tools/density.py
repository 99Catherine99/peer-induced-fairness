import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from metrics import KL_DivergenceCalculator
from metrics import KS_DistanceCalculator
from metrics import Wasserstein_DistanceCalculator
import seaborn as sns


class Density:
    def __init__(self, df, group_col, variable_col, matching='matching direction'):
        """
        Calculate and visualize the probability density estimation (KDE) of a specified variable for different groups.

        Parameters:
        df (DataFrame): Input DataFrame containing relevant columns.
        group_col (str): Name of the column indicating group membership (0 or 1).
        variable_col (str): Name of the column for which to calculate the density.

        Returns:
        None.
        """
        self.df = df
        self.group_col = group_col
        self.variable_col = variable_col
        self.matching = matching

    def calculate_density(self):
        sns.set_context("talk", rc={"lines.linewidth": 2.5})

        protect_df = self.df[self.df[self.group_col] == 0]
        nonprotect_df = self.df[self.df[self.group_col] == 1]

        population_data = self.df[self.variable_col]
        protect_data = protect_df[self.variable_col]
        nonprotect_data = nonprotect_df[self.variable_col]

        population_data_mean = population_data.mean()
        protect_data_mean = protect_data.mean()
        nonprotect_data_mean = nonprotect_data.mean()

        print('Mean of Approval Likelihood regarding Population: ', population_data_mean)
        print('Mean of Approval Likelihood regarding Protect Group: ', protect_data_mean)
        print('Mean of Approval Likelihood regarding Nonprotect Group: ', nonprotect_data_mean)

        population_kde = gaussian_kde(population_data)
        protect_kde = gaussian_kde(protect_data)
        nonprotect_kde = gaussian_kde(nonprotect_data)

        # Generate the x-axis range for the probability density function
        x_range = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and axis object

        plt.plot(x_range, protect_kde(x_range), label='protected group', color='red')
        plt.plot(x_range, nonprotect_kde(x_range), label='nonprotected group', color='blue')
        plt.plot(x_range, population_kde(x_range), label='population', color='orange')

        plt.xlabel(self.variable_col, fontsize=15)
        plt.ylabel('Probability Density', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)  # Set font size for tick labels
        plt.legend(fontsize=15)
        plt.grid(True, color='#D3D3D3', alpha=0.5)  # Light gray grid lines, semi-transparent
        plt.tight_layout()
        plt.savefig(f'{self.matching}_{self.variable_col}.png', dpi=300, bbox_inches='tight')
        plt.show()

        data1 = protect_data.to_numpy()
        data2 = nonprotect_data.to_numpy()

        # kl_calculator = KL_DivergenceCalculator(data1, data2)
        ks_calculator = KS_DistanceCalculator(data1, data2)
        wasserstein_calculator = Wasserstein_DistanceCalculator(data1, data2)

        # Compute distances using the provided calculator instances
        # kl_distance = kl_calculator.compute()
        ks_distance, ks_p_value = ks_calculator.compute()
        wasserstein_distance = wasserstein_calculator.compute()


class DensityComparasion:
    def __init__(self, df, column1, column2):
        """
        Initialize the DensityComparison class.

        Parameters:
        - df (DataFrame): Input DataFrame containing relevant columns.
        - column1 (str): Name of the first column to plot.
        - column2 (str): Name of the second column to plot.

        Returns:
        None.
        """
        self.df = df
        self.column1 = column1
        self.column2 = column2


    def plot_density(self):
        """
        Plot density curves for the specified columns and print their means.

        Returns:
        None.
        """
        column1_data = self.df[self.column1]
        column2_data = self.df[self.column2]

        column1_data_mean = column1_data.mean()
        column2_data_mean = column2_data.mean()

        print(f'Mean of Approval Likelihood regarding {self.column1}: {column1_data_mean}')
        print(f'Mean of Approval Likelihood regarding {self.column2}: {column2_data_mean}')

        column1_kde = gaussian_kde(column1_data)
        column2_kde = gaussian_kde(column2_data)

        # Generate the x-axis range for the probability density function
        x_range = np.linspace(0, 1, 100)

        plt.plot(x_range, column1_kde(x_range), label=f'{self.column1}', color='blue')
        plt.plot(x_range, column2_kde(x_range), label=f'{self.column2}', color='orange')

        plt.xlabel('Pr(Y=1)', fontsize=15)
        plt.ylabel('Probability Density', fontsize=15)
        plt.title(f'Kernel Density Estimation of Pr(Y=1)', fontsize=15)
        plt.legend()
        plt.show()

        data1 = column1_data.to_numpy()
        data2 = column2_data.to_numpy()

        # kl_calculator = KL_DivergenceCalculator(data1, data2)
        ks_calculator = KS_DistanceCalculator(data1, data2)
        wasserstein_calculator = Wasserstein_DistanceCalculator(data1, data2)

        # Compute distances using the provided calculator instances
        # kl_distance = kl_calculator.compute()
        ks_distance, ks_p_value = ks_calculator.compute()
        wasserstein_distance = wasserstein_calculator.compute()


def difference_density(df, group_col, threshold):
    """
    Calculate and visualize the probability density of differences between 'Pr(Y=1)' values.

    Parameters:
    df (DataFrame): Input DataFrame containing relevant columns.
    group_col (str): Name of the column indicating group membership (0 or 1).
    threshold (float): Threshold value which we could obtain from analysis.caplier_and_threshold_calculator.

    Returns:
    None.
    """
    protect_df = df[df[group_col] == 0]
    nonprotect_df = df[df[group_col] == 1]

    population_data = df['Pr(Y=1)']
    protect_data = protect_df['Pr(Y=1)']
    nonprotect_data = nonprotect_df['Pr(Y=1)']

    population_kde = gaussian_kde(population_data)
    protect_kde = gaussian_kde(protect_data)
    nonprotect_kde = gaussian_kde(nonprotect_data)

    # Calculate differences between values in the protect and nonprotect groups
    data_difference_list = []

    for protect_value in protect_data:
        for nonprotect_value in nonprotect_data:
            data_difference_list.append(nonprotect_value - protect_value)

    data_difference_list_mean = np.mean(data_difference_list)
    print('data_difference_list_mean:', data_difference_list_mean)

    # Generate the x-axis range for the probability density function
    x_range = np.linspace(min(data_difference_list), max(data_difference_list), 100)
    # Use seaborn to plot the probability density
    sns.kdeplot(data_difference_list, shade=True)
    # Add a mean difference line
    plt.axvline(x=data_difference_list_mean, color='red', linestyle='--', label='Mean Difference')

    plt.xlabel('Data Difference', fontsize=15)
    plt.ylabel('Probability Density', fontsize=15)
    plt.title('Data Difference Probability Density', fontsize=15)
    plt.legend()
    plt.show()

    count_within_threshold = len([diff for diff in data_difference_list if 0 - threshold < diff < threshold])
    total_count = len(data_difference_list)
    percentage_within_threshold = count_within_threshold / total_count * 100

    print('percentage_within_threshold:', percentage_within_threshold)
