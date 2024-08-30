import pandas as pd
import numpy as np

class DataFrameStatistics:
    def __init__(self, df):
        self.df = df

    def calculate_statistics(self):
        mode = self.df.mode().iloc[0].astype(int)  # Convert mode to integer
        median = self.df.median().astype(int)  # Convert median to integer
        mean = self.df.mean().round(4)  # Round mean to 4 decimal places
        std = self.df.std().round(4)  # Round std to 4 decimal places
        stats = {
            'Count': self.df.count(),
            'Median': median,
            'Mode': mode,
            'Mean': mean,
            'Standard Deviation': std
        }
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv('dataframe_statistics.csv', index=True)  # 保留索引作为列名
        return stats_df

    def calculate_proportions(self):
        """Calculate and export the proportion of each unique value in each column of the DataFrame to a single CSV file, ensuring 'Value' is an integer where possible."""
        all_proportions = []  # List to store all proportions DataFrames

        for column in self.df.columns:
            # Calculate value counts, normalize to get proportions, and sort by index (the values)
            proportions = self.df[column].value_counts(normalize=True, sort=False).sort_index().round(4) * 100  # Convert proportions to percentages
            # Attempt to convert index (values) to integers if possible
            if proportions.index.dtype.kind in 'i':  # Check if index is already integer type
                value_index = proportions.index
            else:
                try:
                    value_index = proportions.index.astype(int)  # Convert index to integer if possible
                except ValueError:
                    value_index = proportions.index  # Keep original if conversion is not possible

            proportions_df = pd.DataFrame({
                'Column': column,  # Add column name as a constant column in DataFrame
                'Value': value_index,
                'Proportion (%)': proportions.values
            })
            all_proportions.append(proportions_df)

        # Concatenate all DataFrames along rows
        final_proportions_df = pd.concat(all_proportions, ignore_index=True)

        # Save the concatenated DataFrame to a single CSV file
        final_proportions_df.to_csv('all_columns_proportions.csv', index=False)

        return final_proportions_df
