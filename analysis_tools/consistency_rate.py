import pandas as pd
import numpy as np

class ConsistencyCalculator:
    def __init__(self, df1, df2):
        """
        Initialize the ratio calculator.
        :param df1: The first DataFrame, containing 'treatment_index' and 'is_significant' columns.
        :param df2: The second DataFrame, containing 'treatment_index' and 'is_significant' columns.
        """
        self.df1 = df1
        self.df2 = df2

    def compute_ratio(self):
        # Find the common 'treatment_index' between the two DataFrames
        common_indices = np.intersect1d(self.df1['treatment_index'], self.df2['treatment_index'])

        # Filter the common 'treatment_index' in both DataFrames
        df1_common = self.df1[self.df1['treatment_index'].isin(common_indices)]
        df2_common = self.df2[self.df2['treatment_index'].isin(common_indices)]

        # Merge the two DataFrames on 'treatment_index' to compare 'is_significant'
        merged_df = pd.merge(df1_common, df2_common, on='treatment_index', suffixes=('_1', '_2'))

        # Calculate the number of entries where 'is_significant' values are the same for the same 'treatment_index'
        matched_significant = merged_df[merged_df['is_significant_1'] == merged_df['is_significant_2']].shape[0]

        # Calculate the ratio
        ratio = matched_significant / len(common_indices) if common_indices.size > 0 else 0
        return ratio

    def compute_exclude_nondetermine_in_denominator_ratio(self):
        # Find the common 'treatment_index' between the two DataFrames
        common_indices = np.intersect1d(self.df1['treatment_index'], self.df2['treatment_index'])

        # Filter the common 'treatment_index' in both DataFrames
        df1_common = self.df1[self.df1['treatment_index'].isin(common_indices)]
        df2_common = self.df2[self.df2['treatment_index'].isin(common_indices)]

        # Exclude rows where 'is_significant' is 'Unknown'
        df1_common = df1_common[df1_common['is_significant'] != 'Unknown']
        df2_common = df2_common[df2_common['is_significant'] != 'Unknown']

        # Merge the two DataFrames on 'treatment_index' to compare 'is_significant'
        merged_df = pd.merge(df1_common, df2_common, on='treatment_index', suffixes=('_1', '_2'))

        # Calculate the number of entries where 'is_significant' values are the same for the same 'treatment_index'
        matched_significant = merged_df[merged_df['is_significant_1'] == merged_df['is_significant_2']].shape[0]

        # Update the denominator for the ratio calculation, ensuring the denominator is the number of rows in the merged DataFrame
        ratio = matched_significant / merged_df.shape[0] if merged_df.shape[0] > 0 else 0
        return ratio

    def compute_exclude_nondetermine_ratio(self):
        # Find the common 'treatment_index' between the two DataFrames
        common_indices = np.intersect1d(self.df1['treatment_index'], self.df2['treatment_index'])

        # Filter the common 'treatment_index' in both DataFrames
        df1_common = self.df1[self.df1['treatment_index'].isin(common_indices)]
        df2_common = self.df2[self.df2['treatment_index'].isin(common_indices)]

        # Filter out rows where 'is_significant' is 'Unknown'
        df1_filtered = df1_common[df1_common['is_significant'] != 'Unknown']
        df2_filtered = df2_common[df2_common['is_significant'] != 'Unknown']

        # Merge the filtered DataFrames on 'treatment_index' to compare 'is_significant'
        merged_df = pd.merge(df1_filtered, df2_filtered, on='treatment_index', suffixes=('_1', '_2'))

        # Calculate the number of entries where 'is_significant' values are the same for the same 'treatment_index'
        matched_significant = merged_df[merged_df['is_significant_1'] == merged_df['is_significant_2']].shape[0]

        # Calculate the ratio
        common_indices_filtered = len(merged_df)  # Update the denominator to the number of rows in the filtered merged DataFrame
        ratio = matched_significant / common_indices_filtered if common_indices_filtered > 0 else 0
        return ratio

    def compute_include_nondetermine_ratio(self):
        # Find the common 'treatment_index' between the two DataFrames
        common_indices = np.intersect1d(self.df1['treatment_index'], self.df2['treatment_index'])

        # Filter the common 'treatment_index' in both DataFrames
        df1_common = self.df1[self.df1['treatment_index'].isin(common_indices)]
        df2_common = self.df2[self.df2['treatment_index'].isin(common_indices)]

        # Merge the two DataFrames on 'treatment_index' to compare 'is_significant'
        merged_df = pd.merge(df1_common, df2_common, on='treatment_index', suffixes=('_1', '_2'))

        # Define a function to determine if the values are consistent
        def is_consistent(row):
            return (row['is_significant_1'] == row['is_significant_2'] or
                    row['is_significant_1'] == 'Unknown' or
                    row['is_significant_2'] == 'Unknown')

        # Apply the function and count the number of consistent entries
        matched_significant = merged_df.apply(is_consistent, axis=1).sum()

        # Calculate the ratio
        common_total = len(merged_df)
        ratio = matched_significant / common_total if common_total > 0 else 0
        return ratio
