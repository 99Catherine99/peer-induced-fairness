import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp

class TreatmentTester:
    def __init__(self, protect_df, nonprotect_df, matched_df, results_df, alpha=0.05):
        """
        Initialize the TreatmentTester class with necessary DataFrames.
        Parameters:
        - protect_df (pd.DataFrame): DataFrame containing protected group data.
        - nonprotect_df (pd.DataFrame): DataFrame containing non-protected group data.
        - matched_df (pd.DataFrame): DataFrame containing matched indices.
        - results_df (pd.DataFrame): DataFrame containing t-test results with a 'is_significant' column.
        """
        self.protect_df = protect_df
        self.nonprotect_df = nonprotect_df
        self.matched_df = matched_df
        self.results_df = results_df
        self.alpha = alpha

    def filter_unfairly_treated(self):
        """
        Filter for 'False' significant results and select corresponding protected data.
        """
        falsely_treated_df = self.results_df[self.results_df['is_significant'] == 'False']
        selected_protect_df = self.protect_df.iloc[falsely_treated_df['treatment_index'].values]
        selected_explanation_df = selected_protect_df[selected_protect_df['Binary Y'] == 0]
        selected_explanation_df = selected_explanation_df[~selected_explanation_df.index.duplicated(keep='first')]
        return selected_explanation_df

    def get_filtered_matched_df(self, selected_explanation_df):
        """
        Filter matched_df for entries that exist in selected_explanation_df.
        """
        filtered_df = self.matched_df[self.matched_df['treatment_index'].isin(selected_explanation_df.index)]
        return filtered_df

    def get_nonprotected_data(self, filtered_df):
        """
        Select nonprotected data based on control indices from filtered_df.
        """
        self.nonprotect_df.reset_index(drop=True, inplace=True)
        selected_nonprotect_df = self.nonprotect_df.loc[filtered_df['control_index']]
        selected_nonprotect_df.reset_index(drop=True, inplace=True)
        selected_nonprotect_df.set_index(filtered_df['control_index'].values, inplace=True)
        return selected_nonprotect_df

    def merge_dataframes(self, filtered_df, selected_nonprotect_df):
        """
        Merge filtered_df with selected_nonprotect_df horizontally.
        """
        total_df = pd.concat([filtered_df.reset_index(drop=True), selected_nonprotect_df.reset_index(drop=True)], axis=1)
        return total_df

    def perform_t_tests(self, total_df, treatment_index):
        """
        Filter total_df by treatment index, split by 'Binary Y', and perform t-tests.
        """
        a_df, b_df = self.filter_by_index(total_df, treatment_index)
        return self.t_test_between_groups(a_df, b_df)

    def filter_by_index(self, dataframe, treatment_index):
        """
        Split the dataframe by 'Binary Y' into two groups based on a specific treatment index.
        """
        selected_df = dataframe[dataframe['treatment_index'] == treatment_index]
        a_df = selected_df[selected_df['Binary Y'] == 0]
        b_df = selected_df[selected_df['Binary Y'] == 1]
        return a_df, b_df

    def t_test_between_groups(self, a_df, b_df):
        """
        Conduct two-sided t-tests between two groups for each feature using the instance's alpha value for significance testing.
        """
        t_test_results = {}
        for column in a_df.columns:
            if column not in ['treatment_index', 'Binary Y', 'control_index']:
                if a_df[column].notnull().sum() > 1 and b_df[column].notnull().sum() > 1:
                    stat, p_value = ttest_ind(a_df[column], b_df[column], equal_var=False, nan_policy='omit')
                    is_significant = p_value < self.alpha  # Use the alpha value set during initialization
                    t_test_results[column] = {
                        't-statistic': round(stat, 4),
                        'p-value': round(p_value, 4),
                        'significant': is_significant
                    }
                else:
                    t_test_results[column] = {
                        't-statistic': None,
                        'p-value': None,
                        'significant': False
                    }

        return pd.DataFrame(t_test_results).T

    def check_data(df):
        print("Data Summary:")
        print(df.describe())  # 提供描述性统计信息
        print("\nNaN Values in each column:")
        print(df.isna().sum())  # 显示每列的NaN数量
        print("\nUnique values in each column:")
        for col in df.columns:
            print(f"{col}: {df[col].nunique()} unique values")  # 显示每列的唯一值数量


    def compare_protected_with_group(self, total_df, treatment_index):
        """
        Compare a specific protected entry against the non-protected group entries after removing the first five columns of non-protected data.
        Parameters:
        - total_df (pd.DataFrame): The merged DataFrame containing both protected and non-protected data.
        - treatment_index (int): The index from protect_df to compare against non-protected group.
        """
        # Extract the specific protected entry
        protected_entry = self.protect_df.loc[self.protect_df['treatment_index'] == treatment_index]

        # Extract non-protected entries that match the treatment_index
        nonprotected_entries = total_df[total_df['treatment_index'] == treatment_index]
        # Assuming 'Binary Y' is in nonprotected_entries and it marks the entries for comparison
        b_df = nonprotected_entries[nonprotected_entries['Binary Y'] == 1]

        # Remove the first five columns from b_df
        b_df = b_df.iloc[:, 5:]  # Adjust this index according to your data structure

        # Ensure column alignment with protected_entry, assuming protected_entry does not include these first five columns
        if protected_entry.columns.tolist() != b_df.columns.tolist():
            b_df = b_df[protected_entry.columns.tolist()]

        # Initialize results dictionary
        t_test_results = {}

        # Perform t-tests on all columns
        for column in protected_entry.columns:
            # Check if both have enough data points to perform a t-test
            if protected_entry[column].notnull().sum() > 0 and b_df[column].notnull().sum() > 0:
                stat, p_value = ttest_1samp(protected_entry[column], b_df[column], equal_var=False, nan_policy='omit')
                is_significant = p_value < self.alpha
                t_test_results[column] = {
                    't-statistic': round(stat, 4),
                    'p-value': round(p_value, 4),
                    'significant': is_significant
                }
            else:
                t_test_results[column] = {
                    't-statistic': None,
                    'p-value': None,
                    'significant': False
                }

        return pd.DataFrame(t_test_results).T
