import pandas as pd

class DataFrameAnalyzer:
    def __init__(self, df, binary_y_col, group_col, binary_prediction_col):
        """
        Initialize the DataFrameAnalyzer class.

        Parameters:
        - df (DataFrame): The input DataFrame containing the data to be analyzed.
        - binary_y_col (str): The column name representing the binary outcome ('Binary Y').
        - group_col (str): The column name representing the group or 'new size' by which data will be grouped.
        - binary_prediction_col (str): The column name representing the binary predictions ('Binary Prediction').
        """
        self.df = df
        self.binary_y_col = binary_y_col
        self.group_col = group_col
        self.binary_prediction_col = binary_prediction_col

    def analyze_dataframe(self):
        """
        Analyze the DataFrame by computing value counts for 'Binary Y', 'new size', and 'Binary Prediction'.
        Also, calculate and display the value counts of 'Binary Y' and 'Binary Prediction' for each group defined by 'new size'.
        """
        # Compute value counts for 'Binary Y', 'new size', and 'Binary Prediction'
        binary_y_counts = self.df[self.binary_y_col].value_counts()
        new_size_counts = self.df[self.group_col].value_counts()
        binary_prediction_counts = self.df[self.binary_prediction_col].value_counts()

        # Print the results
        print("\nnew size value counts:")
        print(new_size_counts)
        print("=" * 80)

        print("Binary Y value counts:")
        print(binary_y_counts)
        print("=" * 80)

        print("\nBinary Prediction value counts:")
        print(binary_prediction_counts)
        print("=" * 80)

        # Compute value counts of 'Binary Y' and 'Binary Prediction' for each 'new size'
        for size in new_size_counts.index:
            subset_df = self.df[self.df[self.group_col] == size]
            binary_y_subset_counts = subset_df[self.binary_y_col].value_counts()
            binary_prediction_subset_counts = subset_df[self.binary_prediction_col].value_counts()

            # Print the results for each 'new size'
            print(f"\nSubset Analysis for new size '{size}':")
            print("Binary Y value counts:")
            print(binary_y_subset_counts)
            print("\nBinary Prediction value counts:")
            print(binary_prediction_subset_counts)
            print("=" * 80)
