import pandas as pd
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

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
        # Define the direction of the one-sided tests for features
        # less_features = [
        #     'regular management account', 'credit purchase', 'new location',
        #     'finance qualification for manager', 'loss or profit', 'new funds injections',
        #     'legal status', 'principal', 'written plan'
        # ]
        # greater_features = [
        #     'new establish time', 'turnover growth rate', 'risk', 'previous turndown',
        #     'business innovation', 'product or service development'
        # ]

        # Extract control_indices corresponding to the specified treatment_index
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
