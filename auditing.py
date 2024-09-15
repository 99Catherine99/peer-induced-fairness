import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import sem
from scipy.stats import kurtosis, t

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from analysis.case_explanations import CaseExplanations

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report


# from metrics.accuracy_calculator import accuracy
# from metrics.precision_calculator import precision
# from metrics.recall_calculator import recall
# from metrics.f1_calculator import f1
# from metrics.roc_calculator import ROC

# from sklearn.model_selection import cross_val_score
# from metrics.bacc_calculator import BalancedAccuracyCalculator
# from metrics.fnr_calculator import FNRCalculator
# from metrics.fpr_calculator import FPRCalculator
# from metrics.npv_calculator import NPVCalculator
# from metrics.ppv_calculator import PPVCalculator
# from metrics.proportion_calculator import ProportionCalculator
# from metrics.sp_calculator import StatisticalParityCalculator
# from metrics.tnr_calculator import TNRCalculator
# from metrics.tpr_calculator import TPRCalculator
# from metrics.distance_calculator import DistanceCalculatorBase
# from scipy.stats import ks_2samp
# from metrics.fairness_calculator import FairnessCalculator
# from scipy.stats import wasserstein_distance



class BootstrapSampler:
    def __init__(self, nonprotect_df, protect_df, matched_df, sampling_times=100, draws_per_sample=25, determine=45,
                 alpha=0.05):
        self.matched_df = matched_df
        self.sampling_times = sampling_times
        self.draws_per_sample = draws_per_sample
        self.determine = determine
        self.nonprotect_df = nonprotect_df
        self.protect_df = protect_df
        self.alpha = alpha

        # Set a fixed random seed for reproducibility
        np.random.seed(45)
        random.seed(45)

    def sample(self):
        samples_list = []
        for _, group in self.matched_df.groupby('treatment_index'):
            if len(group) >= self.determine:
                for _ in range(self.sampling_times):
                    sample = group.sample(n=self.draws_per_sample, replace=True)
                    samples_list.append(sample)
            else:
                samples_list.append(group)
        bootstrapped_samples = pd.concat(samples_list, ignore_index=True)
        bootstrapped_samples.to_csv('bootstrapped_samples.csv', index=False)
        return bootstrapped_samples

    def calculate_means(self, bootstrapped_samples):
        mean_values = {}
        grouped_samples = bootstrapped_samples.groupby('treatment_index')
        for treatment_index, samples in grouped_samples:
            original_count = self.matched_df[self.matched_df['treatment_index'] == treatment_index].shape[0]
            if len(samples) == original_count:
                control_indices = self.matched_df[self.matched_df['treatment_index'] == treatment_index][
                    'control_index']
                pr_values = self.nonprotect_df.loc[control_indices, 'Pr(Y=1)']
                mean_values[treatment_index] = [pr_values.mean()]
            else:
                means = []
                num_samples = len(samples) // self.draws_per_sample
                for i in range(num_samples):
                    sample_indices = samples.iloc[i * self.draws_per_sample:(i + 1) * self.draws_per_sample][
                        'control_index']
                    pr_values = self.nonprotect_df.loc[sample_indices, 'Pr(Y=1)']
                    means.append(pr_values.mean())
                mean_values[treatment_index] = means
        return mean_values



    def perform_test(self, mean_values, direction='two-sided'):
        results = []
        for treatment_index, means in mean_values.items():
            protected_value = self.protect_df.loc[treatment_index, 'Pr(Y=1)']
            if len(means) == 1:
                is_significant = "Unknown"
                t_stat = np.nan
                p_value = np.nan
            else:
                alternative = direction
                t_stat, p_value = ttest_1samp(means, protected_value, alternative=alternative)
                is_significant = "True" if p_value < self.alpha else "False"
            results.append({
                'treatment_index': treatment_index,
                't_stat': t_stat,
                'p_value': p_value,
                'mean_protected': protected_value,
                'mean_matched': np.mean(means) if means else np.nan,
                'is_significant': is_significant
            })
            results_df = pd.DataFrame(results)
            # Save the results to CSV
            results_df.to_csv(f'{direction}_test_results.csv', index=False)
        return results_df

    def plot_treatment_comparison(self, mean_values, results):
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        test_results = results.copy()
        test_results.set_index('treatment_index', inplace=True)

        data_to_plot = []
        for treatment_index, values in mean_values.items():
            # obtain Pr(Y=1) from protected group
            protected_value = self.protect_df.loc[treatment_index, 'Pr(Y=1)']
            # calculate the mean of non-protected group
            matched_mean = np.mean(values)
            # get the significance results
            is_significant = test_results.loc[treatment_index, 'is_significant']


            data_to_plot.append({
                'Treatment Index': treatment_index,
                'Protected Pr(Y=1)': protected_value,
                'Matched Mean Pr(Y=1)': matched_mean,
                'is_significant': is_significant
            })

        plot_df = pd.DataFrame(data_to_plot)

        # add colour according to if the data points are at the right or left side of Y=X
        plot_df['comparison'] = plot_df.apply(
            lambda row: 'Higher than' if row['Protected Pr(Y=1)'] > row['Matched Mean Pr(Y=1)'] and row['is_significant'] == 'True' else
            'Lower than' if row['is_significant'] == 'True' else
            'Equal' if row['is_significant'] == 'False' else
            'Unknown', axis=1)

        # filter out the data points labelled Unknown
        plot_df = plot_df[plot_df['comparison'] != 'Unknown']

        # plot the non-blue data point first
        non_blue = plot_df[plot_df['comparison'] != 'Equal']
        blue = plot_df[plot_df['comparison'] == 'Equal']


        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x='Protected Pr(Y=1)',
            y='Matched Mean Pr(Y=1)',
            hue='comparison',  # Using significance results as color classification
            palette={'Lower than': '#B02425', 'Higher than': '#FF6100'},
            data=non_blue,
            legend=None,
            s=50  # data points size
        )

        # then plot the blue data points
        sns.scatterplot(
            x='Protected Pr(Y=1)',
            y='Matched Mean Pr(Y=1)',
            color='blue',
            data=blue,
            legend=None,
            s=70  # data points size
        )

        plt.xlabel('$Pr(\hat Y=1|X=\\mathbf{x}, S=s-)$', fontsize=15)
        plt.ylabel('$E[\\bar{T}]$', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)

        # bold the x,y tick
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        plt.ylim(bottom=0.7)
        # plt.xlim(left=0, right=1)

        # plot dashed black Y=X
        min_value = min(plot_df['Protected Pr(Y=1)'].min(), plot_df['Matched Mean Pr(Y=1)'].min())
        max_value = max(plot_df['Protected Pr(Y=1)'].max(), plot_df['Matched Mean Pr(Y=1)'].max())
        plt.plot([min_value, max_value], [min_value, max_value], 'k--')


        custom_legend_labels = ['Higher than', 'Lower than', 'Equal']
        custom_legend_colors = ['#FF6100', '#B02425', 'blue']
        custom_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in custom_legend_colors]
        plt.legend(custom_legend_handles, custom_legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=13)

        plt.tight_layout()
        plt.savefig('single_double_matched_scatter_plot.pdf', format='pdf', dpi=300)
        plt.show()

        # Save the DataFrame to CSV
        plot_df.to_csv('single_double_treatment_comparison_results.csv', index=False)

        return plot_df

    def plot_density(self, protect_df, mean_values, treatment_index, bw_adjust=2, show_line=True, line_color='blue', density_color='blue'):
        """
        Plot the density of Pr(Y=1) means for a specified treatment index
        and print the kurtosis and standard deviation.
        """
        blue_line_value = protect_df.iloc[treatment_index]['Pr(Y=1)']
        if treatment_index in mean_values:
            data = mean_values[treatment_index]
            data_kurtosis = kurtosis(data)
            data_std = np.std(data)

            # Print the calculated indicators
            print(f"Kurtosis: {data_kurtosis}")
            print(f"Standard Deviation: {data_std}")

            sns.set_context("talk", rc={"lines.linewidth": 2.5})
            plt.figure(figsize=(5, 4))
            sns.kdeplot(data, shade=False, bw_adjust=bw_adjust, color=density_color, linewidth=2.5, label='Peers')
            if show_line:
                plt.axvline(x=blue_line_value, color=line_color, linestyle='--', linewidth=2, label='A')

            plt.xlabel('Mean $Pr(\hat Y=1)$', fontsize=15)
            plt.ylabel('Density', fontsize=15)
            plt.xlim(0.5, 1)
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=11)
            plt.tight_layout()
            plt.grid(False)
            plt.savefig(f'{treatment_index}_after_bootstrapping.pdf', format='pdf', dpi=300)
            plt.show()
        else:
            print("Treatment index not found in the provided mean values.")




class BulkCaseAnalysis:
    def __init__(self, protect_df, nonprotect_df, matched_df, explanation_df, alpha=0.05):
        """
        Initialize the bulk case analysis class.

        Parameters:
        - protect_df (DataFrame): Protected group data
        - nonprotect_df (DataFrame): Non-protected group data
        - matched_df (DataFrame): Matching results containing 'treatment_index' and 'control_indices'
        - explanation_df (DataFrame): Selected individual data for explanations
        - alpha (float): Significance level, default is 0.05
        """
        self.explanation_df = explanation_df
        self.protect_df = protect_df
        self.nonprotect_df = nonprotect_df
        self.alpha = alpha
        self.matched_df = matched_df
        self.case_explainer = CaseExplanations(protect_df, nonprotect_df, matched_df, alpha)

    def perform_bulk_analysis(self, less_features, greater_features, label='one_2_accepted_group', side='two-sided'):
        """
        Perform case analysis on all individuals in explanation_df and calculate the significance ratio for each feature.
        """
        all_results = []

        if label == 'one_2_accepted_group':
            # Perform analysis for each individual
            for idx in self.explanation_df.index.unique():  # Ensure the indices are unique
                results = self.case_explainer.analyze_fairly(idx, less_features, greater_features, 'accepted', side)
                all_results.append(results)
        if label == 'one_2_rejected_group':
            # Perform analysis for each individual
            for idx in self.explanation_df.index.unique():  # Ensure the indices are unique
                results = self.case_explainer.analyze_fairly(idx, less_features, greater_features, 'rejected', side)
                all_results.append(results)
        if label == 'group_2_group':
            # Perform analysis for each individual
            for idx in self.explanation_df.index.unique():  # Ensure the indices are unique
                results = self.case_explainer.analyze_case_two_groups(idx, less_features, greater_features, side)
                all_results.append(results)

        # Combine all results into a large DataFrame
        all_results_df = pd.concat(all_results)

        # Calculate the significance ratio for each feature, convert to percentage form, and keep two decimal places
        significant_counts = all_results_df.groupby('feature')['is_significant'].mean()
        significant_percentages = (significant_counts * 100).round(2).astype(str) + '%'

        # Result DataFrame
        result_df = pd.DataFrame({'significant_proportion': significant_percentages})
        # Sort by significance ratio in descending order
        result_df = result_df.sort_values(by='significant_proportion', ascending=True)
        # Save the result as a CSV file
        result_df.to_csv('bulk_explanations.csv', index=True)

        return result_df

    def perform_hypothesis_tests(self):
        """
        Perform two-sided hypothesis tests between filtered rows of explanation_df and nonprotect_df.

        Parameters:
        explanation_df (pd.DataFrame): Explanatory DataFrame
        matched_df (pd.DataFrame): DataFrame containing 'treatment_index' and 'control_index'
        nonprotect_df (pd.DataFrame): Non-protected DataFrame
        alpha (float): Significance level, default is 0.05

        Returns:
        pd.DataFrame: Contains t-test statistics, p-values, whether to reject the null hypothesis, and the means of both groups
        """
        results = []

        # Step 1: Find the rows in matched_df where the index matches the treatment_index in explanation_df
        matched_indices = self.matched_df[self.matched_df['treatment_index'].isin(self.explanation_df.index)]

        # Step 2: Based on the matched rows, find the corresponding control_index
        control_indices = matched_indices['control_index']

        # Step 3: Use control_index to filter the corresponding rows in nonprotect_df
        filtered_nonprotect_df = self.nonprotect_df.loc[control_indices]
        filtered_nonprotect_df = filtered_nonprotect_df[filtered_nonprotect_df['Binary Y'] == 0]

        # Step 4: Perform two-sided hypothesis tests for each column in explanation_df and the filtered nonprotect_df
        for column in self.explanation_df.columns:
            if column in filtered_nonprotect_df.columns:
                t_stat, p_value = ttest_ind(self.explanation_df[column].dropna(), filtered_nonprotect_df[column].dropna())
                mean1 = self.explanation_df[column].mean()
                mean2 = filtered_nonprotect_df[column].mean()
                reject_null = p_value < self.alpha
                results.append({
                    'Column': column,
                    'T-Statistic': round(t_stat, 4),
                    'P-Value': round(p_value, 4),
                    'Mean Group 1': round(mean1, 4),
                    'Mean Group 2': round(mean2, 4),
                    'Reject Null Hypothesis': reject_null
                })

        return pd.DataFrame(results)

    def plot_proportion_stacked_bar(self, column):
        """
        Calculate the proportion of each value in the specified column for both DataFrames and plot a stacked bar chart.

        Parameters:
        df1 (pd.DataFrame): The first DataFrame
        df2 (pd.DataFrame): The second DataFrame
        column (str): The column name for which to calculate proportions and plot
        """
        matched_indices = self.matched_df[self.matched_df['treatment_index'].isin(self.explanation_df.index)]

        # Step 2: Based on the matched rows, find the corresponding control_index
        control_indices = matched_indices['control_index']

        # Step 3: Use control_index to filter the corresponding rows in nonprotect_df
        filtered_nonprotect_df = self.nonprotect_df.loc[control_indices]
        filtered_nonprotect_df = filtered_nonprotect_df[filtered_nonprotect_df['Binary Y'] == 0]

        # Calculate the proportion of each value
        df1_proportions = self.explanation_df[column].value_counts(normalize=True).sort_index()
        df2_proportions = filtered_nonprotect_df[column].value_counts(normalize=True).sort_index()

        # Create a proportions DataFrame
        proportions_df = pd.DataFrame({
            'Value': df1_proportions.index.union(df2_proportions.index),
            'DF1 Proportion': df1_proportions.reindex(df1_proportions.index.union(df2_proportions.index), fill_value=0),
            'DF2 Proportion': df2_proportions.reindex(df1_proportions.index.union(df2_proportions.index), fill_value=0)
        }).set_index('Value')

        # Plot the stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Colors for each value
        colors = plt.cm.tab20(np.linspace(0, 1, len(proportions_df)))

        # Stacked bar chart
        proportions_df.T.plot(kind='bar', stacked=True, color=colors, ax=ax)

        # Add labels and title
        ax.set_xlabel('DataFrames')
        ax.set_ylabel('Proportion')
        ax.set_title(f'Stacked Bar Chart of {column} Proportions')
        ax.set_xticklabels(['DF1', 'DF2'], rotation=0)
        ax.legend(title=column, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()





class CaseExplanations:
    def __init__(self, protect_df, nonprotect_df, matching_df, alpha=0.05):
        """
        Initialize the analysis class.

        Parameters:
        - protect_df (DataFrame): Protected group data
        - nonprotect_df (DataFrame): Non-protected group data
        - matching_df (DataFrame): Matching results containing 'treatment_index' and 'control_indices'
        - alpha (float): Significance level, default is 0.05
        """
        self.protect_df = protect_df
        self.nonprotect_df = nonprotect_df
        self.matching_df = matching_df
        self.alpha = alpha

    def analyze(self, treatment_index):
        """
        Analyze the specified treatment_index, comparing data points between the protected group and the non-protected group.

        Parameters:
        - treatment_index (int): The specified treatment_index
        """
        # Extract control_indices corresponding to the specified treatment_index
        control_indices = self.matching_df.loc[
            self.matching_df['treatment_index'] == treatment_index, 'control_index'].explode().unique()

        # Extract data
        treatment_data = self.protect_df.loc[[treatment_index]]
        control_data = self.nonprotect_df.loc[control_indices]
        # control_data = control_data[control_data['Binary Y'] == 1]

        # Calculate the means of the control group data
        control_means = control_data.mean()

        # Perform two-sided hypothesis tests for all features
        results = []
        for feature in treatment_data.columns:
            if treatment_data[feature].dtype.kind in 'ifc' and control_data[feature].dtype.kind in 'ifc':  # Only process numeric data
                # Use a one-sample t-test to compare against the control group mean
                t_stat, p_value = ttest_1samp(control_data[feature].dropna(), treatment_data[feature].iloc[0], alternative='two-sided')
                results.append({
                    'feature': feature,
                    't_stat': t_stat,
                    'c': p_value,
                    'mean_treatment': treatment_data[feature].iloc[0],
                    'mean_control': control_means[feature],
                    'is_significant': p_value < self.alpha  # Check for significance
                })
            else:
                print(f"Skipping non-numeric feature: {feature}")

        return pd.DataFrame(results)

    def analyze_fairly(self, treatment_index, less_features, greater_features, label='accepted', side='two-sided'):
        """
        Analyze the specified treatment_index, comparing data points between the protected group and the non-protected group.

        Parameters:
        - treatment_index (int): The specified treatment_index
        """
        control_indices = self.matching_df.loc[
            self.matching_df['treatment_index'] == treatment_index, 'control_index'].explode().unique()

        # Extract data
        treatment_data = self.protect_df.loc[[treatment_index]]
        control_data = self.nonprotect_df.loc[control_indices]

        if label == 'accepted':
            control_data = control_data[control_data['Binary Y'] == 1]
        if label == 'rejected':
            control_data = control_data[control_data['Binary Y'] == 0]

        # Calculate the means of the control group data
        control_means = control_data.mean()

        # Perform hypothesis tests for all features
        results = []
        for feature in treatment_data.columns:
            if treatment_data[feature].dtype.kind in 'ifc' and control_data[feature].dtype.kind in 'ifc':  # Only process numeric data
                # Determine the direction of the test
                if feature in less_features:
                    side = 'less'
                elif feature in greater_features:
                    side = 'greater'
                else:
                    side = 'two-sided'

                # Use a one-sample t-test to compare against the control group mean
                t_stat, p_value = ttest_1samp(control_data[feature].dropna(), treatment_data[feature].iloc[0],
                                              alternative=side)
                results.append({
                    'feature': feature,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'mean_treatment': treatment_data[feature].iloc[0],
                    'mean_control': control_means[feature],
                    'is_significant': p_value < self.alpha  # Check for significance
                })
            else:
                print(f"Skipping non-numeric feature: {feature}")

        results_df = pd.DataFrame(results)

        return results_df

    def analyze_case_two_groups(self, treatment_index, side='two-sided'):
        """
        Analyze the specified treatment_index, comparing data points between the protected group and the non-protected group.

        Parameters:
        - treatment_index (int): The specified treatment_index
        """
        # Extract control_indices corresponding to the specified treatment_index
        control_indices = self.matching_df.loc[
            self.matching_df['treatment_index'] == treatment_index, 'control_index'].explode().unique()

        # Extract data
        treatment_data = self.protect_df.loc[[treatment_index]]
        control_data = self.nonprotect_df.loc[control_indices]
        control_data_1 = control_data[control_data['Binary Y'] == 0]
        control_data_2 = control_data[control_data['Binary Y'] == 1]

        control_means_1 = control_data_1.mean()
        control_means_2 = control_data_2.mean()

        # Perform two-sided hypothesis tests for all features
        results = []
        for feature in treatment_data.columns:
            if treatment_data[feature].dtype.kind in 'ifc' and control_data[feature].dtype.kind in 'ifc':  # Only process numeric data
                # Use an independent two-sample t-test to compare between the two control groups
                t_stat, p_value = ttest_ind(control_data_1[feature].dropna(), control_data_2[feature].dropna(), alternative=side)
                results.append({
                    'feature': feature,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'mean_treatment': control_means_1[feature],
                    'mean_control': control_means_2[feature],
                    'is_significant': p_value < self.alpha  # Check for significance
                })
            else:
                print(f"Skipping non-numeric feature: {feature}")

        results_df = pd.DataFrame(results)

        # Save results as a CSV file
        # results_df.to_csv(f'case_explanation.csv', index=False)

        return results_df


def caplier_and_threshold(df, group_col, ps_col='Pr(S=0)', proba_col='Pr(Y=1)', caplier_ratio=0.2, threshold_ratio=0.2):
    """
    Calculate Caplier limits and thresholds based on standard deviations.

    Parameters:
    - df: DataFrame containing data.
    - group_col: Column specifying the group (0 or 1).
    - caplier_ratio: Ratio for calculating Caplier limits.
    - threshold_ratio: Ratio for calculating thresholds.

    Returns:
    - protect_caplier: Caplier limit for the protected group.
    - nonprotect_caplier: Caplier limit for the non-protected group.
    - protect_threshold: Threshold for the protected group.
    - nonprotect_threshold: Threshold for the non-protected group.
    """

    protect_df = df[df[group_col] == 0]
    nonprotect_df = df[df[group_col] == 1]

    # Check if ps_col exists in the DataFrame
    if ps_col in df.columns:
        protect_ps_std = protect_df[ps_col].std()
        nonprotect_ps_std = nonprotect_df[ps_col].std()
    else:
        protect_ps_std = 0  # Set a default value if ps_col does not exist
        nonprotect_ps_std = 0  # Set a default value if ps_col does not exist

    protect_y_std = protect_df[proba_col].std()
    nonprotect_y_std = nonprotect_df[proba_col].std()

    protect_caplier = protect_ps_std * caplier_ratio
    nonprotect_caplier = nonprotect_ps_std * caplier_ratio

    protect_threshold = protect_y_std * threshold_ratio
    nonprotect_threshold = nonprotect_y_std * threshold_ratio

    return protect_caplier, nonprotect_caplier, protect_threshold, nonprotect_threshold






def data_divide(df, group_col=None, binary_col=None, prediction_col=None, ps_col=None):
    """
    Divide the DataFrame into protected and non-protected groups, accepted and rejected groups, and extract specific columns-Pr(S=0),
    which indicates the propensity scores.

    Parameters:
    df (DataFrame): Input DataFrame containing relevant columns.
    group_col (str): Name of the column indicating group membership (0 or 1).
    binary_col (str): Name of the column indicating binary membership (0 or 1).
    prediction_col (str): Name of the column indicating prediction membership (0 or 1).
    ps_col (str): Name of the column indicating propensity scores.

    Returns:
    protect_df (DataFrame): DataFrame containing data for the protected group.
    nonprotect_df (DataFrame): DataFrame containing data for the non-protected group.
    accepted_df (DataFrame): DataFrame containing data for the accepted group.
    rejected_df (DataFrame): DataFrame containing data for the rejected group.
    pred_accepted_df (DataFrame): DataFrame containing data for the accepted group based on prediction (if available).
    pred_rejected_df (DataFrame): DataFrame containing data for the rejected group based on prediction (if available).
    protect_ps (list): List of 'Pr(S=0)' values for the protected group (if available).
    nonprotect_ps (list): List of 'Pr(S=0)' values for the non-protected group (if available).
    """

    # Initialize empty DataFrames
    protect_df = nonprotect_df = accepted_df = rejected_df = pred_accepted_df = pred_rejected_df = None
    protect_ps = nonprotect_ps = None

    # Check if group_col exists and split data if available
    if group_col and group_col in df:
        protect_df = df[df[group_col] == 0]
        nonprotect_df = df[df[group_col] == 1]
        protect_df.reset_index(drop=True, inplace=True)
        nonprotect_df.reset_index(drop=True, inplace=True)

    # Check if binary_col exists and split data if available
    if binary_col and binary_col in df:
        accepted_df = df[df[binary_col] == 1]
        rejected_df = df[df[binary_col] == 0]
        accepted_df.reset_index(drop=True, inplace=True)
        rejected_df.reset_index(drop=True, inplace=True)

    # Check if prediction_col exists and split data if available
    if prediction_col and prediction_col in df:
        pred_accepted_df = df[df[prediction_col] == 1]
        pred_rejected_df = df[df[prediction_col] == 0]
        pred_accepted_df.reset_index(drop=True, inplace=True)
        pred_rejected_df.reset_index(drop=True, inplace=True)

    # Check if ps_col exists
    if ps_col and ps_col in df:
        protect_ps = protect_df[ps_col].tolist() if protect_df is not None else None
        nonprotect_ps = nonprotect_df[ps_col].tolist() if nonprotect_df is not None else None

    return protect_df, nonprotect_df, accepted_df, rejected_df, pred_accepted_df, pred_rejected_df, protect_ps, nonprotect_ps





class DensityScatterPlotter:
    def __init__(self, protect_df, nonprotect_df, bootstrapped_samples, result_type, comparison_type='less than'):
        self.protect_df = protect_df
        self.nonprotect_df = nonprotect_df
        self.bootstrapped_samples = bootstrapped_samples
        self.result_type = result_type
        self.comparison_type = comparison_type

    def calculate_p_value(self, t_statistic, freedom=99):
        """
        Calculate p-value for a given t-statistic and degrees of freedom.
        """
        p_value = t.sf(np.abs(t_statistic), freedom)  # sf is the survival function (1 - cdf)
        return p_value

    def plot_group_density(self, dataframe, column1, column2, color, density_color='#84BA84'):
        """
        绘制有颜色数据点的两个指定数据列的密度图，并在同一图中显示，并打印均值和标准差。
        """

        # 设置Seaborn情境，确保所有线条更粗
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        plt.figure(figsize=(6, 5))

        data1 = dataframe[column1]
        data2 = dataframe[column2]
        mean1 = data1.mean()
        mean2 = data2.mean()
        std1 = data1.std()
        std2 = data2.std()

        # 打印均值和标准差
        print(f"{color} Protected Pr(Y=1) Mean: {mean1}, Standard Deviation: {std1}")
        print(f"{color} Matched Mean Pr(Y=1) Mean: {mean2}, Standard Deviation: {std2}")

        # 绘制密度图
        ax = sns.kdeplot(data1, shade=False, color=color, linewidth=7, label='Micro-firms', linestyle='--')
        sns.kdeplot(data2, shade=False, color=density_color, linewidth=7, label='Peers group')

        # 设置标签和标题
        plt.xlabel('$Pr(\hat Y=1|X=\\mathbf{x})$', fontsize=15)
        plt.ylabel('Density', fontsize=15)
        plt.xlim(0.6, 1)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=15)
        plt.tight_layout()
        plt.grid(False)

        plt.xticks(np.linspace(0.6, 1, 5))  # 例如，设置5个刻度点

        # 加粗刻度标签
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # 保存和显示图形
        plt.savefig(f'{self.result_type}_{self.comparison_type}_group_density.pdf', format='pdf', dpi=300)
        plt.show()
        # Save the data to CSV
        density_df = pd.DataFrame({
            f'{column1}_mean': [mean1],
            f'{column2}_mean': [mean2]
        })
        density_df.to_csv(f'{self.result_type}_{self.comparison_type}_density_stats.csv', index=False)

    def plot_scatter(self, mean_values, selected_treatment_index, results, threshold_type='fixed', floating_ratio=0.1, mean_diff_threshold=0.03, p_value_threshold=0.05):
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        mean_values = {k: v for k, v in mean_values.items() if len(v) > 1}

        # 过滤掉'is_significant'为'Unknown'的结果
        test_results = results[results['is_significant'] != 'Unknown']
        test_results.set_index('treatment_index', inplace=True)
        data_to_plot = []

        for treatment_index, values in mean_values.items():
            protected_value = self.protect_df.loc[treatment_index, 'Pr(Y=1)']
            matched_mean = np.mean(values)
            mean_diff = matched_mean - protected_value
            reverse_mean_diff = protected_value - matched_mean
            p_value = test_results.loc[treatment_index, 'p_value']
            is_significant = test_results.loc[treatment_index, 'is_significant']

            # 确定阈值
            if threshold_type == 'floating':
                actual_mean_diff_threshold = floating_ratio * protected_value
            else:
                actual_mean_diff_threshold = mean_diff_threshold

            color = 'gray'
            # Determine the color based on mean_diff and p_value thresholds
            if is_significant == 'True':
                if self.result_type == 'single_less_sided_results':
                    if self.comparison_type == 'less than':
                        color = '#FF6100' if reverse_mean_diff > actual_mean_diff_threshold else 'gray'
                    elif self.comparison_type == 'between':
                        color = '#FF6100' if reverse_mean_diff <= actual_mean_diff_threshold else 'gray'

                elif self.result_type == 'single_greater_sided_results':
                    if self.comparison_type == 'less than':
                        color = '#B02425' if mean_diff > actual_mean_diff_threshold else 'gray'
                    elif self.comparison_type == 'between':
                        color = '#B02425' if mean_diff <= actual_mean_diff_threshold else 'gray'

            elif self.result_type == 'double_sided_results':
                color = 'blue' if p_value > p_value_threshold else 'gray'

            data_to_plot.append({
                'treatment_index': treatment_index,
                'protected Pr(Y=1)': protected_value,
                'matched mean Pr(Y=1)': matched_mean,
                'mean_diff': mean_diff,
                'p_value': p_value,
                'color': color,
                'is_selected': 'Yes' if treatment_index == selected_treatment_index else 'No'
            })

        plot_df = pd.DataFrame(data_to_plot)

        # Plot gray points first
        gray_df = plot_df[plot_df['color'] == 'gray']
        color_df = plot_df[plot_df['color'] != 'gray']

        # Print the number of colored points
        print(f"Number of colored points: {len(color_df)}")

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            x='protected Pr(Y=1)',
            y='matched mean Pr(Y=1)',
            color='gray',
            data=gray_df,
            legend=None,
            s=50,
            alpha=0.3  # 设置灰色点的透明度
        )

        alpha_value = 1

        # Plot colored points
        sns.scatterplot(
            x='protected Pr(Y=1)',
            y='matched mean Pr(Y=1)',
            hue='color',
            palette={'#FF6100': '#FF6100', '#B02425': '#B02425', '#84BA84': '#84BA84', 'blue': 'blue'},
            data=color_df,
            legend=None,
            s=70,
            alpha=alpha_value
        )

        # Highlight selected point
        highlight_row = plot_df[plot_df['treatment_index'] == selected_treatment_index]
        highlight_color = highlight_row['color'].values[0]
        plt.scatter(
            highlight_row['protected Pr(Y=1)'],
            highlight_row['matched mean Pr(Y=1)'],
            color=highlight_color,
            marker='^',
            edgecolor='black',
            label='Selected micro-firm',
            s=90
        )

        plt.xlabel('$Pr(\hat Y=1|X=\\mathbf{x}, S=s-)$', fontsize=15)
        plt.ylabel('$E[\\bar{T}]$', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        plt.ylim(bottom=0.7)

        # 绘制Y=X的黑色虚线
        min_value = min(plot_df['protected Pr(Y=1)'].min(), plot_df['matched mean Pr(Y=1)'].min())
        max_value = max(plot_df['protected Pr(Y=1)'].max(), plot_df['matched mean Pr(Y=1)'].max())
        plt.plot([min_value, max_value], [min_value, max_value], 'k--')


        if self.result_type == 'single_less_sided_results':
            if self.comparison_type == 'less than':
                custom_legend_labels = ['EP', 'Others', 'Selected micro-firm']
                custom_legend_colors = ['#FF6100', 'gray', highlight_color]
            elif self.comparison_type == 'between':
                custom_legend_labels = ['SP', 'Others', 'Selected micro-firm']
                custom_legend_colors = ['#FF6100', 'gray', highlight_color]
        elif self.result_type == 'single_greater_sided_results':
            if self.comparison_type == 'less than':
                custom_legend_labels = ['ED', 'Others', 'Selected micro-firm']
                custom_legend_colors = ['#B02425', 'gray', highlight_color]
            elif self.comparison_type == 'between':
                custom_legend_labels = ['SD', 'Others', 'Selected micro-firm']
                custom_legend_colors = ['#B02425', 'gray', highlight_color]
        elif self.result_type == 'double_sided_results':
            custom_legend_labels = ['FT', 'Others', 'Selected micro-firm']
            custom_legend_colors = ['blue', 'gray', highlight_color]
        else:
            custom_legend_labels = ['Others', 'Selected micro-firm']
            custom_legend_colors = ['gray', highlight_color]

        custom_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in custom_legend_colors]
        custom_legend_handles[-1] = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=highlight_color, markeredgecolor='black', markersize=10)
        plt.legend(custom_legend_handles, custom_legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False, fontsize=12)

        plt.tight_layout()
        plt.savefig(f'{self.result_type}_{self.comparison_type}_scatter.pdf', format='pdf', dpi=300)
        plt.show()

        data_ratio = color_df.shape[0] / test_results.shape[0]
        print(f'the data ratio is: {data_ratio}')

        # Save the dataframes to CSV
        plot_df.to_csv(f'{self.result_type}_{self.comparison_type}_plot_df.csv', index=False)
        gray_df.to_csv(f'{self.result_type}_{self.comparison_type}_gray_df.csv', index=False)
        color_df.to_csv(f'{self.result_type}_{self.comparison_type}_color_df.csv', index=False)

        return plot_df, gray_df, color_df


    def group_ground_truth(self, color_df,label='sigle_greater_between'):
        """
        根据 color_df['treatment_index'] 到 bootstrapped_samples['treatment_index'] 寻找 'control_index'，
        然后在 nonprotect_df 的 index 中寻找相应的 Binary Y，
        计算 Binary Y 为 0 和 1 的个数以及比例。

        参数：
        - color_df (pd.DataFrame): 包含有颜色数据点的 DataFrame。

        返回：
        - pd.DataFrame: 包含所有 treatment_index 的 Binary Y 为 0 和 1 的个数以及比例的 DataFrame。
        """
        all_control_indices = []
        protect_binary_y_values = []

        treatment_indices = color_df['treatment_index']

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

        df = pd.DataFrame(data)

        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        # 绘制堆积柱形图
        fig, ax = plt.subplots(figsize=(5,4))

        # Nonprotect组的比例
        nonprotect_0 = df[(df['Type'] == 'Nonprotect') & (df['Binary Y'] == 0)]['Proportion'].values[0]
        nonprotect_1 = df[(df['Type'] == 'Nonprotect') & (df['Binary Y'] == 1)]['Proportion'].values[0]

        # Protect组的比例
        protect_0 = df[(df['Type'] == 'Protect') & (df['Binary Y'] == 0)]['Proportion'].values[0]
        protect_1 = df[(df['Type'] == 'Protect') & (df['Binary Y'] == 1)]['Proportion'].values[0]

        bar_width = 0.2
        x = np.array([1.6, 2])  # 调整x位置，使柱子更近

        # 绘制堆积柱形图
        bars1 = ax.bar(x[0], nonprotect_0, bar_width, color='#B02425')
        bars2 = ax.bar(x[0], nonprotect_1, bar_width, bottom=nonprotect_0, color='blue')

        bars3 = ax.bar(x[1], protect_0, bar_width, color='#B02425')
        bars4 = ax.bar(x[1], protect_1, bar_width, bottom=protect_0, color='blue')

        ax.set_xlabel('Group', fontsize=15)
        ax.set_ylabel('Percentage', fontsize=15)
        # ax.set_title('Proportion of Binary Y for Protect and Nonprotect Groups')
        ax.set_xticks(x)
        ax.set_xticklabels(['Peers', 'Micro-firms'], fontsize=15)

        # 设置图例，仅对两个色块进行解释
        ax.legend(['Rejected', 'Accepted'], loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2,
                  frameon=False, fontsize=15)

        plt.tight_layout()
        plt.savefig(f'{label}_proportion_stacked_bar_plot.pdf', format='pdf', dpi=300)
        plt.show()
        # Save the dataframe to CSV
        df.to_csv(f'{label}_group_ground_truth.csv', index=False)

        return df

    def try_2(self, color_df, pr_y_means, label='sigle_greater_between'):
        """
        根据 color_df['treatment_index'] 到 bootstrapped_samples['treatment_index'] 寻找 'control_index'，
        然后在 nonprotect_df 的 index 中寻找相应的 Binary Y，
        计算 Binary Y 为 0 和 1 的个数以及比例，并将 pr_y_means 字典中的值作为 E[T] 加入结果。

        参数：
        - color_df (pd.DataFrame): 包含有颜色数据点的 DataFrame。
        - pr_y_means (dict): 包含 treatment_index 对应的 E[T] 值的字典。

        返回：
        - pd.DataFrame: 包含所有 treatment_index 的 Binary Y 为 0 和 1 的个数、比例及 E[T] 的 DataFrame。
        """
        all_control_indices = []
        protect_binary_y_values = []

        treatment_indices = color_df['treatment_index']

        results = []

        for treatment_index in treatment_indices:
            control_indices = self.bootstrapped_samples.loc[
                self.bootstrapped_samples['treatment_index'] == treatment_index, 'control_index']
            all_control_indices.extend(control_indices.explode().tolist())

            binary_y_values = self.nonprotect_df.loc[control_indices.explode().tolist(), 'Binary Y']
            count_0 = (binary_y_values == 0).sum()
            count_1 = (binary_y_values == 1).sum()
            total = len(binary_y_values)
            proportion_0 = count_0 / total if total > 0 else 0
            proportion_1 = count_1 / total if total > 0 else 0

            # 计算 E[T] 值
            e_t_values = pr_y_means.get(treatment_index, [])
            e_t_mean = np.mean(e_t_values) if e_t_values else np.nan

            results.append({
                'treatment_index': treatment_index,
                'Binary Y 0 Count': count_0,
                'Binary Y 1 Count': count_1,
                'Proportion 0': proportion_0,
                'Proportion 1': proportion_1,
                'Y': self.protect_df.loc[treatment_index, 'Binary Y'],  # 新增一列标识Y
                'Pr(Y=1)': self.protect_df.loc[treatment_index, 'Pr(Y=1)'],  # 新增 Pr(Y=1)
                # 'Binary Prediction': self.protect_df.loc[treatment_index, 'Binary Prediction'],  # 新增 Binary Prediction
                'E[T]': e_t_mean  # 新增 E[T]
            })

            protect_binary_y_values.append(self.protect_df.loc[treatment_index, 'Binary Y'])

        protect_count_0 = (np.array(protect_binary_y_values) == 0).sum()
        protect_count_1 = (np.array(protect_binary_y_values) == 1).sum()
        protect_total = len(protect_binary_y_values)
        protect_proportion_0 = protect_count_0 / protect_total
        protect_proportion_1 = protect_count_1 / protect_total

        protect_data = {
            'Type': ['Protect', 'Protect'],
            'Binary Y': [0, 1],
            'Count': [protect_count_0, protect_count_1],
            'Proportion': [protect_proportion_0, protect_proportion_1]
        }

        results_df = pd.DataFrame(results)
        protect_df = pd.DataFrame(protect_data)

        # results_df.to_csv(f'{label}_group_ground_truth.csv', index=False)
        # protect_df.to_csv(f'{label}_protect_group_ground_truth.csv', index=False)

        return results_df, protect_df


    def plot_case_density(self, protect_df, mean_values, treatment_index, bw_adjust=2, show_line=True,
                          line_color='#B02425',
                          density_color='#84BA84'):
        """
        Plot the density of Pr(Y=1) means for a specified treatment index
        and print the kurtosis, standard deviation, and density mean.
        """
        line_value = protect_df.iloc[treatment_index]['Pr(Y=1)']
        if treatment_index in mean_values:
            data = mean_values[treatment_index]
            data_kurtosis = kurtosis(data)
            data_std = np.std(data)
            data_mean = np.mean(data)

            # Print the calculated indicators
            print(f"Kurtosis: {data_kurtosis}")
            print(f"Standard Deviation: {data_std}")
            print(f"Density Mean: {data_mean}")
            print(f"Line Value (A): {line_value}")

            sns.set_context("talk", rc={"lines.linewidth": 2.5})
            plt.figure(figsize=(6,5))
            ax = sns.kdeplot(data, shade=False, bw_adjust=bw_adjust, color=density_color, linewidth=7, label='Peers')
            if show_line:
                plt.axvline(x=line_value, color=line_color, linestyle='--', linewidth=7, label='Selected micro-firm')

            plt.xlabel('$Pr(\hat Y=1|X=\\mathbf{x})$', fontsize=15)
            plt.ylabel('Density', fontsize=15)
            plt.xlim(0.5, 1)
            # 加粗刻度标签
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,
                       frameon=False, fontsize=15)
            plt.tight_layout()
            plt.grid(False)
            plt.savefig(f'{self.result_type}_{self.comparison_type}_{treatment_index}_case_density.pdf', format='pdf', dpi=300)
            plt.show()
            # Save the data to CSV
            density_df = pd.DataFrame({
                'Peers Value': data
            })
            density_df['Line Value (A)'] = line_value
            density_df.to_csv(f'{self.result_type}_{self.comparison_type}_{treatment_index}_case_density_stats.csv',
                              index=False)
        else:
            print("Treatment index not found in the provided mean values.")



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


class accuracy:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def cross_validate(self, scoring='accuracy'):
        cv_accuracy_scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring=scoring)

        print(f"{self.cv}-fold Cross Validation Accuracy:", cv_accuracy_scores)
        print("Mean Accuracy:", cv_accuracy_scores.mean())
        print("Standard Deviation of the Mean Accuracy:", cv_accuracy_scores.std())
        print("Standard Error of the Mean Accuracy:", sem(cv_accuracy_scores))

        return cv_accuracy_scores


class f1:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def cross_validate(self, scoring='f1'):
        cv_f1_scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring=scoring)

        print(f"{self.cv}-fold Cross Validation F1:", cv_f1_scores)
        print("Mean F1:", cv_f1_scores.mean())
        print("Standard Deviation of the Mean F1:", cv_f1_scores.std())
        print("Standard Error of the Mean F1:", sem(cv_f1_scores))

        return cv_f1_scores


class precision:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def cross_validate(self, scoring='precision'):
        cv_precision_scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring=scoring)

        print(f"{self.cv}-fold Cross Validation Precision:", cv_precision_scores)
        print("Mean Precision:", cv_precision_scores.mean())
        print("Standard Deviation of the Mean Precision:", cv_precision_scores.std())
        print("Standard Error of the Mean Precision:", sem(cv_precision_scores))

        return cv_precision_scores


class recall:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def cross_validate(self, scoring='recall'):
        cv_recall_scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring=scoring)

        print(f"{self.cv}-fold Cross Validation Recall:", cv_recall_scores)
        print("Mean Recall:", cv_recall_scores.mean())
        print("Standard Deviation of the Mean Recall:", cv_recall_scores.std())
        print("Standard Error of the Mean Recall:", sem(cv_recall_scores))
        return cv_recall_scores


class ROC:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def cross_validate(self):
        cv_auc_scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring='roc_auc')

        print(f"{self.cv}-fold Cross Validation AUC:", cv_auc_scores)
        print("Mean AUC:", cv_auc_scores.mean())
        print("Standard Deviation of the Mean AUC:", cv_auc_scores.std())
        print("Standard Error of the Mean AUC:", sem(cv_auc_scores))

        return cv_auc_scores


class ModelTrainer:
    """
    A class for training different classifiers using grid search for hyperparameter tuning.

    Methods:
    grid_search(self, model): Perform grid search for hyperparameter tuning and return the best model.

    Attributes:
    classifier: The classifier instance to be trained, which we could choose.
    param_grid: The hyperparameter grid to be used in grid search, different classifier has different hyperparameters.
    """

    def __init__(self):
        """
        Initialize the ModelTrainer class.
        """

        self.classifier = None
        self.param_grid = None

    def grid_search(self, model):
        """
        Perform grid search for hyperparameter tuning and return the best model.

        Args:
        model (str): The type of classifier to train. Supported values: 'LogisticRegression', 'RandomForest', 'SVM', we could add more then.

        Returns:
        grid_search (GridSearchCV): A GridSearchCV object configured with the specified classifier and parameter grid.
        """

        if model == "LogisticRegression":
            self.classifier = LogisticRegression(random_state=42)
            self.param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2']
            }
        elif model == "RandomForest":
            self.classifier = RandomForestClassifier(random_state=42)
            self.param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model == "SVM":
            self.classifier = SVC(probability=True, random_state=42)
            self.param_grid = {
                'C': [1.0],
                'kernel': ['linear'],
                'gamma': ['auto']
            }
        elif model == "XGBClassifier":
            self.classifier = XGBClassifier(random_state=42, use_label_encoder=False)
            self.param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.1],
                'subsample': [1.0]
            }
        elif model == "DecisionTree":
            self.classifier = DecisionTreeClassifier(random_state=42)
            self.param_grid = {
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'criterion': ['gini']
            }

        else:
            raise ValueError("Invalid model type. Supported models: 'LogisticRegression', 'RandomForest', 'SVM'")

        grid_search = GridSearchCV(estimator=self.classifier, param_grid=self.param_grid, cv=5)

        return grid_search




class RFModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier = RandomForestClassifier(random_state=42)  # 根据需要调整 XGBoost 的参数
        self.param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        self.grid_search = GridSearchCV(estimator=self.classifier, param_grid=self.param_grid, cv=5, scoring='roc_auc')

    def train_model(self):
        """
        Train the logistic regression model using grid search.
        """
        self.grid_search.fit(self.X_train, self.y_train)

    def evaluate_performance(self):
        """
        Evaluate the performance of the logistic regression model using various metrics.
        """
        AUC_evaluator = ROC(self.grid_search, self.X, self.y, cv=5)
        accuracy_evaluator = accuracy(self.grid_search, self.X, self.y, cv=5)
        precision_evaluator = precision(self.grid_search, self.X, self.y, cv=5)
        recall_evaluator = recall(self.grid_search, self.X, self.y, cv=5)
        f1_evaluator = f1(self.grid_search, self.X, self.y, cv=5)

        # cv_auc_scores = AUC_evaluator.cross_validate(scoring='roc_auc')

        # accuracy_evaluator.cross_validate(scoring='roc_auc')
        AUC_evaluator.cross_validate()
        accuracy_evaluator.cross_validate(scoring='accuracy')
        precision_evaluator.cross_validate(scoring='precision')
        recall_evaluator.cross_validate(scoring='recall')
        f1_evaluator.cross_validate(scoring='f1')


    def generate_test_report(self):
        """
        Generate classification report on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print(report)

    def calculate_auc(self):
        """
        Calculate AUC (Area Under the ROC Curve) on the test set.
        """
        y_prob = self.grid_search.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, y_prob)
        print('AUC:', auc)

    def calculate_acc(self):
        """
        Calculate Accuracy on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print('Accuracy:', acc)

    def calculate_f1(self):
        """
        Calculate F1 Score on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred)
        print('F1 Score:', f1)

    def calculate_precision(self):
        """
        Calculate Precision on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        precision = precision_score(self.y_test, y_pred)
        print('Precision:', precision)

    def calculate_recall(self):
        """
        Calculate Recall on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        recall = recall_score(self.y_test, y_pred)
        print('Recall:', recall)

    def predict(self, X_pred):
        """
        Predict the target variable for a given set of features.

        Parameters:
        - X_pred (array-like): Features for prediction.

        Returns:
        - array-like: Predicted target variable.
        """
        return self.grid_search.predict(X_pred)

    def predict_proba(self, X_pred):
        """
        Predict class probabilities for a given set of features.

        Parameters:
        - X_pred (array-like): Features for prediction.

        Returns:
        - array-like: Predicted class probabilities.
        """
        return self.grid_search.predict_proba(X_pred)


class XGBoostModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier = XGBClassifier(random_state=42)  # 根据需要调整 XGBoost 的参数
        self.param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.1],
            'subsample': [0.7, 1.0]
            }
        self.grid_search = GridSearchCV(estimator=self.classifier, param_grid=self.param_grid, cv=5, scoring='roc_auc')

    def train_model(self):
        """
        Train the logistic regression model using grid search.
        """
        self.grid_search.fit(self.X_train, self.y_train)

    def evaluate_performance(self):
        """
        Evaluate the performance of the logistic regression model using various metrics.
        """
        AUC_evaluator = ROC(self.grid_search, self.X, self.y, cv=5)
        accuracy_evaluator = accuracy(self.grid_search, self.X, self.y, cv=5)
        precision_evaluator = precision(self.grid_search, self.X, self.y, cv=5)
        recall_evaluator = recall(self.grid_search, self.X, self.y, cv=5)
        f1_evaluator = f1(self.grid_search, self.X, self.y, cv=5)


        AUC_evaluator.cross_validate()
        accuracy_evaluator.cross_validate(scoring='accuracy')
        precision_evaluator.cross_validate(scoring='precision')
        recall_evaluator.cross_validate(scoring='recall')
        f1_evaluator.cross_validate(scoring='f1')

    def generate_test_report(self):
        """
        Generate classification report on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print(report)

    def calculate_auc(self):
        """
        Calculate AUC (Area Under the ROC Curve) on the test set.
        """
        y_prob = self.grid_search.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, y_prob)
        print('AUC:', auc)

    def calculate_acc(self):
        """
        Calculate Accuracy on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print('Accuracy:', acc)

    def calculate_f1(self):
        """
        Calculate F1 Score on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred)
        print('F1 Score:', f1)

    def calculate_precision(self):
        """
        Calculate Precision on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        precision = precision_score(self.y_test, y_pred)
        print('Precision:', precision)

    def calculate_recall(self):
        """
        Calculate Recall on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        recall = recall_score(self.y_test, y_pred)
        print('Recall:', recall)

    def predict(self, X_pred):
        """
        Predict the target variable for a given set of features.

        Parameters:
        - X_pred (array-like): Features for prediction.

        Returns:
        - array-like: Predicted target variable.
        """
        return self.grid_search.predict(X_pred)

    def predict_proba(self, X_pred):
        """
        Predict class probabilities for a given set of features.

        Parameters:
        - X_pred (array-like): Features for prediction.

        Returns:
        - array-like: Predicted class probabilities.
        """
        return self.grid_search.predict_proba(X_pred)


class LogisticRegressionModel:
    """
    Initialize the Logistic Regression Model.

    Parameters:
    - X (array-like): Features for training.
    - y (array-like): Target variable for training.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.lr = LogisticRegression(max_iter=1000, random_state=42)
        self.param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
        self.grid_search = GridSearchCV(estimator=self.lr, param_grid=self.param_grid, cv=5, scoring='roc_auc')

    def train_model(self):
        """
        Train the logistic regression model using grid search.
        """
        self.grid_search.fit(self.X_train, self.y_train)

    def evaluate_performance(self):
        """
        Evaluate the performance of the logistic regression model using various metrics.
        """
        AUC_evaluator = ROC(self.grid_search, self.X, self.y, cv=5)
        accuracy_evaluator = accuracy(self.grid_search, self.X, self.y, cv=5)
        precision_evaluator = precision(self.grid_search, self.X, self.y, cv=5)
        recall_evaluator = recall(self.grid_search, self.X, self.y, cv=5)
        f1_evaluator = f1(self.grid_search, self.X, self.y, cv=5)

        # cv_auc_scores = AUC_evaluator.cross_validate(scoring='roc_auc')

        # accuracy_evaluator.cross_validate(scoring='roc_auc')
        AUC_evaluator.cross_validate()
        accuracy_evaluator.cross_validate(scoring='accuracy')
        precision_evaluator.cross_validate(scoring='precision')
        recall_evaluator.cross_validate(scoring='recall')
        f1_evaluator.cross_validate(scoring='f1')

    def generate_test_report(self):
        """
        Generate classification report on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        report = classification_report(self.y_test, y_pred)
        print(report)

    def calculate_auc(self):
        """
        Calculate AUC (Area Under the ROC Curve) on the test set.
        """
        y_prob = self.grid_search.predict_proba(self.X_test)[:, 1]
        auc = roc_auc_score(self.y_test, y_prob)
        print('AUC:', auc)

    def calculate_acc(self):
        """
        Calculate Accuracy on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print('Accuracy:', acc)

    def calculate_f1(self):
        """
        Calculate F1 Score on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred)
        print('F1 Score:', f1)

    def calculate_precision(self):
        """
        Calculate Precision on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        precision = precision_score(self.y_test, y_pred)
        print('Precision:', precision)

    def calculate_recall(self):
        """
        Calculate Recall on the test set.
        """
        y_pred = self.grid_search.predict(self.X_test)
        recall = recall_score(self.y_test, y_pred)
        print('Recall:', recall)

    def predict(self, X_pred):
        """
        Predict the target variable for a given set of features.

        Parameters:
        - X_pred (array-like): Features for prediction.

        Returns:
        - array-like: Predicted target variable.
        """
        return self.grid_search.predict(X_pred)

    def predict_proba(self, X_pred):
        """
        Predict class probabilities for a given set of features.

        Parameters:
        - X_pred (array-like): Features for prediction.

        Returns:
        - array-like: Predicted class probabilities.
        """
        return self.grid_search.predict_proba(X_pred)


class CaplierMatching:
    def __init__(self, df, group_col, K=15, caplier_ratio=0.2):
        self.df = df
        self.group_col = group_col
        self.K = K
        self.caplier_ratio = caplier_ratio
        self.protect_df, self.nonprotect_df, self.protect_pr, self.nonprotect_pr, self.protect_ps, self.nonprotect_ps = self.preparation_for_psm()
        self.weighted_protect_df, self.weighted_nonprotect_df, self.weighted_protect_ps, self.weighted_nonprotect_ps = self.calculate_weighted_ps()

    def preparation_for_psm(self):
        protect_df = self.df[self.df[self.group_col] == 0]
        nonprotect_df = self.df[self.df[self.group_col] == 1]
        self.protect_pr = protect_df['Pr(Y=1)'].tolist()
        self.nonprotect_pr = nonprotect_df['Pr(Y=1)'].tolist()
        protect_ps = protect_df['Pr(S=0)'].tolist()
        nonprotect_ps = nonprotect_df['Pr(S=0)'].tolist()
        return protect_df, nonprotect_df, self.protect_pr, self.nonprotect_pr, protect_ps, nonprotect_ps

    def calculate_weighted_ps(self):
        total_count = len(self.df)
        protect_ratio = len(self.protect_df) / total_count
        nonprotect_ratio = len(self.nonprotect_df) / total_count
        weighted_protect_ps = [ps / protect_ratio for ps in self.protect_ps]
        weighted_nonprotect_ps = [(1-ps) / nonprotect_ratio for ps in self.nonprotect_ps]
        return self.protect_df, self.nonprotect_df, weighted_protect_ps, weighted_nonprotect_ps


    def caplier_matching(self):
        weighted_protect_ps_np = np.array(self.weighted_protect_ps)
        weighted_nonprotect_ps_np = np.array(self.weighted_nonprotect_ps)
        caplier = self.caplier_ratio * np.std(weighted_protect_ps_np)
        matched_data_points = []

        for protect_index, weighted_protect_ps in enumerate(weighted_protect_ps_np):
            # calculate the distance
            differences = np.abs(weighted_nonprotect_ps_np - weighted_protect_ps)
            mask = differences <= caplier
            matching_indices = np.where(mask)[0]

            # If there are matching points, sort the matching indexes by the difference size
            if matching_indices.size > 0:
                sorted_indices = matching_indices[np.argsort(differences[matching_indices])]

                for match_index in sorted_indices:
                    control_ps = weighted_nonprotect_ps_np[match_index]
                    matched_data_points.append({
                        'treatment_index': protect_index,
                        'treatment_ps': weighted_protect_ps,
                        'control_index': match_index,
                        'control_ps': control_ps,
                        'abs_difference': differences[match_index]  # 已经计算的差异
                    })

        return pd.DataFrame(matched_data_points)








#
# class BalancedAccuracyCalculator(FairnessCalculator):
#     def compute(self):
#         # Calculate TPR for both protected and non-protected groups
#         tpr_calculator = TPRCalculator(self.df, self.group_col, self.prediction_col, self.true_col)
#         protect_tpr, nonprotect_tpr, population_tpr, diff_tpr = tpr_calculator.compute(true_value=1)  # No need to pass true_value
#
#         # Calculate TNR for both protected and non-protected groups
#         tnr_calculator = TNRCalculator(self.df, self.group_col, self.prediction_col, self.true_col)
#         protect_tnr, nonprotect_tnr, population_tnr, diff_tnr = tnr_calculator.compute(true_value=0)  # No need to pass true_value
#
#         # Calculate Balanced Accuracy
#         protect_bacc = (protect_tpr + protect_tnr) / 2
#         nonprotect_bacc = (nonprotect_tpr + nonprotect_tnr) / 2
#         population_bacc = (population_tpr + population_tnr) / 2
#         diff_bacc = nonprotect_bacc - protect_bacc
#
#         return protect_bacc, nonprotect_bacc, population_bacc, diff_bacc



#
#
# class CVCalculator(FairnessCalculator):
#     """
#     A class for calculating CV Score (CV) and Absolute CV scores for protected and non-protected groups.
#
#     This class inherits from the FairnessCalculator class.
#
#     Methods:
#     compute(self): Calculate CV and Absolute CV scores for protected and non-protected groups.
#                    Returns the CV and Absolute CV scores.
#     """
#
#     def compute(self):
#         """
#         Calculate CV Score (CV) and Absolute CV scores for protected and non-protected groups.
#
#         Returns:
#         cv_score (float): CV score representing the difference in Statistical Parity
#                           between protected and non-protected groups.
#         abs_cv_score (float): Absolute CV score representing the absolute difference in Statistical Parity
#                               between protected and non-protected groups.
#         """
#
#         statistical_parity_calculator = StatisticalParityCalculator(self.df, self.group_col, self.prediction_col,
#                                                                     self.true_col)
#         protect_sp, nonprotect_sp, population_sp = statistical_parity_calculator.compute()
#
#         cv_score = nonprotect_sp - protect_sp
#         abs_cv_score = abs(cv_score)
#
#         print('cv_score: ', cv_score)
#         print('abs_cv_score: ', abs_cv_score)
#
#         return cv_score, abs_cv_score
#
#
#
#
# class DiscriminationRatioCalculator(FairnessCalculator):
#     """
#     A class for calculating Discrimination Ratio for protected and non-protected groups.
#
#     This class inherits from the FairnessCalculator class.
#
#     Methods:
#     compute(self): Calculate Discrimination Ratio between protected and non-protected groups.
#                    Returns the Discrimination Ratio.
#     """
#
#     def compute(self):
#         """
#         Calculate Discrimination Ratio between protected and non-protected groups.
#
#         Returns:
#         discrimination_ratio (float): Discrimination Ratio representing the ratio of Statistical Parity
#                                      between protected and non-protected groups.
#         """
#
#         statistical_parity_calculator = StatisticalParityCalculator(self.df, self.group_col, self.prediction_col,
#                                                                     self.true_col)
#         protect_sp, nonprotect_sp, population_sp = statistical_parity_calculator.compute()
#
#         discrimination_ratio = protect_sp / nonprotect_sp
#
#         print('discrimination_ratio: ', discrimination_ratio)
#
#         return discrimination_ratio
#
#
#
#
# class DistanceCalculatorBase:
#     """
#     A base class for calculating distances between two sets of data.
#
#     Methods:
#     kl(p, q): Define the Kullback-Leibler (KL) divergence between two probability distributions.
#     __init__(data1, data2): Initialize the DistanceCalculatorBase instance with two sets of data.
#     compute(self): Calculate the distance between the two sets of data, which should be implemented in subclass.
#     """
#
#     @staticmethod
#     def kl(p, q):
#         """
#         Calculate the Kullback-Leibler (KL) divergence between two probability distributions.
#
#         Args:
#         p (array-like): The first probability distribution.
#         q (array-like): The second probability distribution.
#
#         Returns:
#         kl_divergence (float): The KL divergence between the two distributions.
#         """
#
#         return np.sum(np.where(p != 0, p * np.log(p / q), 0))
#
#     def __init__(self, data1, data2):
#         """
#         Initialize the DistanceCalculatorBase instance with two sets of data.
#
#         Args:
#         data1 (array-like): The first set of data.
#         data2 (array-like): The second set of data.
#         """
#
#         self.data1 = data1
#         self.data2 = data2
#
#     def compute(self):
#         """
#         Calculate the distance between the two sets of data.
#
#         This method should be implemented in subclasses.
#
#         Raises:
#         NotImplementedError: This method should be implemented in subclasses.
#         """
#
#         raise NotImplementedError("Subclasses should implement the 'compute' method.")
#



# class FairnessCalculator:
#     """
#     A base class for calculating fairness metrics between protected and non-protected groups.
#
#     Methods:
#     __init__(df, group_col, prediction_col, true_col): Initialize the FairnessCalculator instance.
#     _split_groups(self): Split the dataset into protected, non-protected, and population groups.
#     compute(self): Calculate fairness metrics between protected and non-protected groups, which should be implemented in subclass.
#
#     Attributes:
#     df (DataFrame): The input DataFrame containing data and predictions.
#     group_col (str): The column representing the group membership.
#     prediction_col (str): The column containing prediction, eg:Pr(\hat Y=1).
#     true_col (str): The column containing true labels, eg:Pr(Y=1).
#     """
#
#     def __init__(self, df, group_col, prediction_col, true_col):
#         self.df = df
#         self.group_col = group_col
#         self.prediction_col = prediction_col
#         self.true_col = true_col
#
#     def _split_groups(self):
#         protect_df = self.df[self.df[self.group_col] == 0]
#         nonprotect_df = self.df[self.df[self.group_col] == 1]
#         population_df = self.df
#
#         return protect_df, nonprotect_df, population_df
#
#     def compute(self):
#         raise NotImplementedError("Subclasses should implement the 'compute' method.")


#
#
#
# class FairnessMetricsCalculator:
#     def __init__(self, df, group_col, prediction_col, true_col):
#         """
#         Initialize the FairnessMetricsCalculator.
#
#         Parameters:
#         - df (DataFrame): The input DataFrame containing the data.
#         - group_col (str): The column specifying the protected group.
#         - prediction_col (str): The column containing model predictions.
#         - true_col (str): The column containing true labels.
#         """
#         self.df = df
#         self.group_col = group_col
#         self.prediction_col = prediction_col
#         self.true_col = true_col
#
#     def calculate_metrics(self, file_prefix):
#         """
#         Calculate fairness metrics and store them in a CSV file.
#
#         Parameters:
#         - file_prefix (str): Prefix for the output CSV file name.
#
#         Returns:
#         - metrics_df (DataFrame): DataFrame containing fairness metrics.
#         """
#         metrics = []
#
#         def calculate_and_store(metric_calculator, metric_name, **kwargs):
#             """
#             Calculate a fairness metric using a specified calculator and store the result in the metrics list.
#
#             Parameters:
#             - metric_calculator (class): The class responsible for calculating the fairness metric.
#             - metric_name (str): The name of the fairness metric.
#             - **kwargs: Additional keyword arguments to pass to the metric calculator.
#
#             This function initializes a metric calculator with the given data and calculates the specified fairness metric.
#             The calculated metric result is then appended to the metrics list along with its name.
#
#             Returns:
#             None
#             """
#             calculator = metric_calculator(self.df, self.group_col, self.prediction_col, self.true_col)
#             result = calculator.compute(**kwargs)
#             metrics.append([metric_name] + list(result))
#
#         calculate_and_store(ProportionCalculator, 'Proportion')
#         calculate_and_store(StatisticalParityCalculator, 'Statistical Parity')
#         calculate_and_store(TPRCalculator, 'TPR', true_value=1)
#         calculate_and_store(FPRCalculator, 'FPR', true_value=0)
#         # calculate_and_store(TNRCalculator, 'TNR', true_value=0)
#         # calculate_and_store(FNRCalculator, 'FNR', true_value=1)
#         # calculate_and_store(BalancedAccuracyCalculator, 'Balanced Accuracy')
#         calculate_and_store(PPVCalculator, 'PPV', prediction_value=1)
#         # calculate_and_store(NPVCalculator, 'NPV', prediction_value=0)
#         # calculate_and_store(DiscriminationRatioCalculator, 'Discrimination Ratio')
#         # calculate_and_store(CVCalculator, 'CV')
#
#         # Create a DataFrame from the metrics list
#         metrics_df = pd.DataFrame(metrics, columns=['Metric', 'Protected', 'Nonprotected', 'Population', 'Difference'])
#         output_filename = f'{file_prefix}_fairness_metrics.csv'
#         metrics_df.to_csv(output_filename, index=True)
#
#         return metrics_df
#


# class FNRCalculator(FairnessCalculator):
#     def compute(self, true_value=1):
#         protect_df, nonprotect_df, population_df = self._split_groups()
#
#         protect_y_df = protect_df[protect_df[self.true_col] == true_value]
#         nonprotect_y_df = nonprotect_df[nonprotect_df[self.true_col] == true_value]
#         population_y_df = population_df[population_df[self.true_col] == true_value]
#
#         protect_fnr = protect_y_df[self.prediction_col].value_counts()[0] / len(protect_y_df) if len(protect_y_df) > 0 else 0
#         nonprotect_fnr = nonprotect_y_df[self.prediction_col].value_counts()[0] / len(nonprotect_y_df) if len(nonprotect_y_df) > 0 else 0
#         population_fnr = population_y_df[self.prediction_col].value_counts()[0] / len(population_y_df) if len(population_y_df) > 0 else 0
#         diff_fnr = nonprotect_fnr - protect_fnr
#
#         # print('protect_fnr: ', protect_fnr)
#         # print('nonprotect_fnr: ', nonprotect_fnr)
#         # print('population_fnr: ', population_fnr)
#         # print('diff_fnr: ', diff_fnr)
#
#         return protect_fnr, nonprotect_fnr, population_fnr, diff_fnr
#
#
#
# class FPRCalculator(FairnessCalculator):
#     def compute(self, true_value=0):
#         protect_df, nonprotect_df, population_df = self._split_groups()
#
#         protect_y_df = protect_df[protect_df[self.true_col] == true_value]
#         nonprotect_y_df = nonprotect_df[nonprotect_df[self.true_col] == true_value]
#         population_y_df = population_df[population_df[self.true_col] == true_value]
#
#         protect_fpr = protect_y_df[self.prediction_col].value_counts()[1] / len(protect_y_df) if len(protect_y_df) > 0 else 0
#         nonprotect_fpr = nonprotect_y_df[self.prediction_col].value_counts()[1] / len(nonprotect_y_df) if len(nonprotect_y_df) > 0 else 0
#         population_fpr = population_y_df[self.prediction_col].value_counts()[1] / len(population_y_df) if len(population_y_df) > 0 else 0
#         diff_fpr = nonprotect_fpr - protect_fpr
#
#         # print('protect_fpr: ', protect_fpr)
#         # print('nonprotect_fpr: ', nonprotect_fpr)
#         # print('population_fpr: ', population_fpr)
#         # print('diff_fpr: ', diff_fpr)
#
#         return protect_fpr, nonprotect_fpr, population_fpr, diff_fpr

#
#
# class KL_DivergenceCalculator(DistanceCalculatorBase):
#     """
#     A class for calculating the Kullback-Leibler (KL) divergence between two datasets.
#
#     Methods:
#     compute(self): Calculate and return the KL divergence between the two datasets.
#
#     Inherits from:
#     DistanceCalculatorBase: A base class providing common methods for distance calculators.
#
#     Attributes:
#     data1 (numpy.ndarray): The first dataset for which to calculate KL divergence.
#     data2 (numpy.ndarray): The second dataset for which to calculate KL divergence.
#     """
#
#     def compute(self):
#         """
#         Calculate and return the KL divergence between the two datasets.
#         Returns: kl_divergence (float): The calculated KL divergence between the two datasets.
#         """
#
#         kl_divergence = self.kl(self.data1, self.data2)
#         print("KL Divergence:", kl_divergence)
#         return kl_divergence
#



# class KS_DistanceCalculator(DistanceCalculatorBase):
#     """
#     A class for calculating the Kolmogorov-Smirnov (KS) distance between two datasets.
#
#     Methods:
#     compute(self): Calculate and return the KS distance and p-value between the two datasets.
#
#     Inherits from:
#     DistanceCalculatorBase: A base class providing common methods for distance calculators.
#
#     Attributes:
#     data1 (numpy.ndarray): The first dataset for which to calculate KS distance.
#     data2 (numpy.ndarray): The second dataset for which to calculate KS distance.
#     """
#
#     def compute(self):
#         """
#         Calculate and return the KS distance and p-value between the two datasets.
#         Returns: ks_statistic (float): The calculated KS distance between the two datasets, ks_p_value (float): The p-value associated with the KS test.
#         """
#
#         ks_statistic, ks_p_value = ks_2samp(self.data1, self.data2)
#         print("KS Distance:", ks_statistic)
#         print('KS P Value:', ks_p_value)
#         return ks_statistic, ks_p_value



#
# class NPVCalculator(FairnessCalculator):
#     def compute(self, prediction_value=0):
#         protect_df, nonprotect_df, population_df = self._split_groups()
#
#         protect_yhat_df = protect_df[protect_df[self.prediction_col] == prediction_value]
#         nonprotect_yhat_df = nonprotect_df[nonprotect_df[self.prediction_col] == prediction_value]
#         population_yhat_df = population_df[population_df[self.prediction_col] == prediction_value]
#
#         protect_npv = protect_yhat_df[self.true_col].value_counts()[0] / len(protect_yhat_df)
#         nonprotect_npv = nonprotect_yhat_df[self.true_col].value_counts()[0] / len(nonprotect_yhat_df)
#         population_npv = population_yhat_df[self.true_col].value_counts()[0] / len(population_yhat_df)
#         diff_npv = nonprotect_npv - protect_npv
#
#         # print('protect_bacc: ', protect_npv)
#         # print('nonprotect_bacc: ', nonprotect_npv)
#         # print('population_bacc: ', population_npv)
#         # print('diff_npv: ', diff_npv)
#
#         return protect_npv, nonprotect_npv, population_npv, diff_npv
#
#
#
#
# class PPVCalculator(FairnessCalculator):
#     def compute(self, prediction_value=1):
#         protect_df, nonprotect_df, population_df = self._split_groups()
#
#         protect_yhat_df = protect_df[protect_df[self.prediction_col] == prediction_value]
#         nonprotect_yhat_df = nonprotect_df[nonprotect_df[self.prediction_col] == prediction_value]
#         population_yhat_df = population_df[population_df[self.prediction_col] == prediction_value]
#
#         protect_ppv = protect_yhat_df[self.true_col].value_counts()[1] / len(protect_yhat_df)
#         nonprotect_ppv = nonprotect_yhat_df[self.true_col].value_counts()[1] / len(nonprotect_yhat_df)
#         population_ppv = population_yhat_df[self.true_col].value_counts()[1] / len(population_yhat_df)
#         diff_ppv = nonprotect_ppv - protect_ppv
#
#         # print('protect_bacc: ', protect_ppv)
#         # print('nonprotect_bacc: ', nonprotect_ppv)
#         # print('population_bacc: ', population_ppv)
#         # print('diff_ppv: ', diff_ppv)
#
#         return protect_ppv, nonprotect_ppv, population_ppv, diff_ppv






#
# class ProportionCalculator(FairnessCalculator):
#     def compute(self):
#         protect_df, nonprotect_df, population_df = self._split_groups()
#
#         protect_pro = protect_df[self.true_col].value_counts()[1] / len(protect_df)
#         nonprotect_pro = nonprotect_df[self.true_col].value_counts()[1] / len(nonprotect_df)
#         population_pro = population_df[self.true_col].value_counts()[1] / len(population_df)
#         diff_pro = nonprotect_pro-protect_pro
#
#         # print('protect_pro: ', protect_pro)
#         # print('nonprotect_pro: ', nonprotect_pro)
#         # print('population_pro: ', population_pro)
#         # print('diff_pro: ', diff_pro)
#
#         return protect_pro, nonprotect_pro, population_pro, diff_pro





#
#
# class StatisticalParityCalculator(FairnessCalculator):
#     def compute(self):
#         protect_df, nonprotect_df, population_df = self._split_groups()
#
#         protect_sp = protect_df[self.prediction_col].value_counts()[1] / len(protect_df)
#         nonprotect_sp = nonprotect_df[self.prediction_col].value_counts()[1] / len(nonprotect_df)
#         population_sp = population_df[self.prediction_col].value_counts()[1] / len(population_df)
#         diff_sp = nonprotect_sp - protect_sp
#
#         # print('protect_sp: ', protect_sp)
#         # print('nonprotect_sp: ', nonprotect_sp)
#         # print('population_sp: ', population_sp)
#         # print('diff_sp: ', diff_sp)
#
#         return protect_sp, nonprotect_sp, population_sp, diff_sp

#
# class TNRCalculator(FairnessCalculator):
#     def compute(self, true_value=0):
#         protect_df, nonprotect_df, population_df = self._split_groups()
#
#         protect_y_df = protect_df[protect_df[self.true_col] == true_value]
#         nonprotect_y_df = nonprotect_df[nonprotect_df[self.true_col] == true_value]
#         population_y_df = population_df[population_df[self.true_col] == true_value]
#
#         protect_tnr = protect_y_df[self.prediction_col].value_counts()[0] / len(protect_y_df) if len(protect_y_df) > 0 else 0
#         nonprotect_tnr = nonprotect_y_df[self.prediction_col].value_counts()[0] / len(nonprotect_y_df) if len(nonprotect_y_df) > 0 else 0
#         population_tnr = population_y_df[self.prediction_col].value_counts()[0] / len(population_y_df) if len(population_y_df) > 0 else 0
#         diff_tnr = nonprotect_tnr-protect_tnr
#
#         # print('protect_tnr: ', protect_tnr)
#         # print('nonprotect_tnr: ', nonprotect_tnr)
#         # print('population_tnr: ', population_tnr)
#         # print('diff_tnr: ', diff_tnr)
#
#         return protect_tnr, nonprotect_tnr, population_tnr, diff_tnr


#
# class TPRCalculator(FairnessCalculator):
#     def compute(self, true_value=1):
#         protect_df, nonprotect_df, population_df = self._split_groups()
#
#         protect_y_df = protect_df[protect_df[self.true_col] == true_value]
#         nonprotect_y_df = nonprotect_df[nonprotect_df[self.true_col] == true_value]
#         population_y_df = population_df[population_df[self.true_col] == true_value]
#
#         protect_tpr = protect_y_df[self.prediction_col].value_counts()[1] / len(protect_y_df) if len(protect_y_df) > 0 else 0
#         nonprotect_tpr = nonprotect_y_df[self.prediction_col].value_counts()[1] / len(nonprotect_y_df) if len(nonprotect_y_df) > 0 else 0
#         population_tpr = population_y_df[self.prediction_col].value_counts()[1] / len(population_y_df) if len(population_y_df) > 0 else 0
#         diff_tpr = nonprotect_tpr - protect_tpr
#
#         # print('protect_tpr: ', protect_tpr)
#         # print('nonprotect_tpr: ', nonprotect_tpr)
#         # print('population_tpr: ', population_tpr)
#         # print('diff_tpr: ', diff_tpr)
#
#         return protect_tpr, nonprotect_tpr, population_tpr, diff_tpr
#


#
# class Wasserstein_DistanceCalculator(DistanceCalculatorBase):
#     """
#     A class for calculating the Wasserstein distance (Earth Mover's distance, EMD) between two datasets.
#
#     Methods:
#     compute(self): Calculate and return the Wasserstein distance between the two datasets.
#
#     Inherits from:
#     DistanceCalculatorBase: A base class providing common methods for distance calculators.
#
#     Attributes:
#     data1 (numpy.ndarray): The first dataset for which to calculate the Wasserstein distance.
#     data2 (numpy.ndarray): The second dataset for which to calculate the Wasserstein distance.
#     """
#
#     def compute(self):
#         """
#         Calculate and return the Wasserstein distance between the two datasets.
#         Returns: wasserstein (float): The calculated Wasserstein distance between the two datasets.
#         """
#
#         wasserstein = wasserstein_distance(self.data1, self.data2)
#         print("Wasserstein Distance:", wasserstein)
#         return wasserstein