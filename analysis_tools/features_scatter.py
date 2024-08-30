import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataPlotter:
    def __init__(self, dataframe, x_column, y_column):
        """
        Initialize the DataPlotter class.
        Parameters:
        - dataframe (DataFrame): The dataset to analyze.
        - x_column (str): The column name to use as the x-axis.
        - y_column (str): The column name to use as the y-axis.
        """
        self.dataframe = dataframe
        self.x_column = x_column
        self.y_column = y_column

    def plot_scatter(self):
        """
        Plot a scatter plot using the DataFrame and column names defined in the class.
        """
        # Check if the column names exist in the DataFrame
        if self.x_column in self.dataframe.columns and self.y_column in self.dataframe.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=self.x_column, y=self.y_column, data=self.dataframe)
            plt.title(f'Scatter Plot of {self.x_column} vs {self.y_column}')
            plt.xlabel(self.x_column)
            plt.ylabel(self.y_column)
            plt.grid(True)
            plt.show()
        else:
            print(f"Error: Ensure that '{self.x_column}' and '{self.y_column}' are valid column names in the DataFrame.")
