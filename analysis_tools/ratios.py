from scipy import stats
import numpy as np
import pandas as pd

class ValueRatioTableGenerator:
    """
    Calculate the ratios of each raw value in population dataframe and in each col value dataframe
    """
    def __init__(self, df, col, row):
        """
        Initializes the ValueRatioTableGenerator class instance.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        col (str): The name of the column for which you want to calculate ratios (e.g., 'race').
        row (str): The name of the column for which you want to calculate ratios against (e.g., 'assets').
        """

        self.df = df
        self.col = col
        self.row = row

    def generate_ratio_table(self):
        """
        Generate a table containing value ratios for specified columns and specified rows.

        Returns:
        ratio_table (pandas.DataFrame): A table with value ratios for specified columns and specified rows.
        """

        row_values = sorted(self.df[self.row].unique())
        col_values = sorted(self.df[self.col].unique())

        ratio_table = pd.DataFrame(index=row_values, columns=['population'] + col_values)

       # Calculate total population for each race
        col_population = self.df[self.col].value_counts()
        row_population = self.df[self.row].value_counts()

        for i in col_values:
            col_df = self.df[self.df[self.col] == i]
            total_population = col_population[i]

            # Calculate assets value counts for the current race
            row_value_counts = col_df[self.row].value_counts()

            for j in row_values:
                if j in row_value_counts:
                    ratio_table.at[j, 'population'] = row_population[j]/len(self.df)
                    ratio_table.at[j, i] = row_value_counts[j] / total_population

        ratio_table = ratio_table.fillna(0)
        ratio_table = ratio_table.rename_axis(self.row, axis=0)

        return ratio_table

    def calculate_t_values(self, col_data0, col_data1):
        """
        Calculate t-values between two groups of data.

        Parameters:
        col_data0 (pandas.Series): Data for group 0.
        col_data1 (pandas.Series): Data for group 1.

        Returns:
        t_value (float): The calculated t-value.
        p_value (float): The calculated p-value.
        """
        t_stat, p_value = stats.ttest_ind(col_data0, col_data1)
        return t_stat, p_value

    def calculate_significance(self, p_value, alpha_values):
        """
        Calculate significance based on p-value.

        Parameters:
        p_value (float): The p-value.
        alpha_values (list): A list of significance levels to check against.

        Returns:
        significance (dict): A dictionary indicating significance at different alpha levels.
        """
        significance = {}
        for alpha in alpha_values:
            if p_value < alpha:
                significance[f'p<{alpha}'] = 1
            else:
                significance[f'p<{alpha}'] = 0
        return significance

    @property
    def perform_t_test(self):
        """
        Perform t-test to compare each 'col' group against 'population' for each 'row' value.

        Returns:
        t_table (pandas.DataFrame): A table with t-values between 'col' groups under the same row value.
        """
        row_values = sorted(self.df[self.row].unique())
        col_values = sorted(self.df[self.col].unique())
        t_table_data = []

        alpha_values = [0.01, 0.05, 0.10]  # Define your significance levels here

        for j in row_values:
            col_data0 = self.df[self.df[self.col] == 0][self.row]
            col_data1 = self.df[self.df[self.col] == 1][self.row]

            col_data0_modified = col_data0.copy()
            col_data1_modified = col_data1.copy()

            for a in range(len(col_data0_modified)):
                if col_data0.iloc[a] != j:
                    col_data0_modified.iloc[a] = 0

            for a in range(len(col_data1_modified)):
                if col_data1.iloc[a] != j:
                    col_data1_modified.iloc[a] = 0

            t_stat, p_value = self.calculate_t_values(col_data0_modified, col_data1_modified)
            t_stat = t_stat
            p_value = p_value

            significance = self.calculate_significance(p_value, alpha_values)

            t_table_row = {'t_value': t_stat, 'p_value': p_value, **significance}
            t_table_data.append(t_table_row)

        t_table_df = pd.DataFrame(t_table_data, index=row_values, columns=['t_value', 'p_value'] + [f'p<{alpha}' for alpha in alpha_values])

        return t_table_df

