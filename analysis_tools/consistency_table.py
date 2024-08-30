import pandas as pd
import numpy as np
from analysis.consistency_rate import ConsistencyCalculator

class MultiConsistencyCalculator:
    def __init__(self, dfs):
        """
        Initialize the multi DataFrame consistency calculator.
        :param dfs: A list containing multiple DataFrames.
        """
        self.dfs = dfs
        self.labels = [f'df{i + 1}' for i in range(len(dfs))]  # Generate labels list df1 to df5

    def consistency_table(self):
        """
        Calculate the consistency rate between each pair of DataFrames in the list.
        Returns a DataFrame that includes the index combination of each pair of DataFrames and their consistency rate.
        Displayed as a percentage, with two decimal places.
        """
        n = len(self.dfs)
        # Initialize an empty DataFrame with NaN values
        ratio_df = pd.DataFrame(index=self.labels, columns=self.labels).astype('float')

        # Fill in the DataFrame cells for the lower triangle
        for i in range(n):
            for j in range(i):  # Calculate only the lower triangle
                calculator = ConsistencyCalculator(self.dfs[i], self.dfs[j])
                # Calculate the ratio, convert to percentage form, and retain two decimal places
                ratio = calculator.compute_exclude_nondetermine_in_denominator_ratio() * 100
                formatted_ratio = f"{ratio:.2f}%"  # Format the output
                # Set the corresponding cell in the lower triangle
                ratio_df.at[self.labels[i], self.labels[j]] = formatted_ratio

        # The diagonal is filled with 100% because consistency with itself is always 100%
        np.fill_diagonal(ratio_df.values, "100.00%")
        return ratio_df

