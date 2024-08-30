import pandas as pd
from analysis.case_explanations import CaseExplanations
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import numpy as np

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
