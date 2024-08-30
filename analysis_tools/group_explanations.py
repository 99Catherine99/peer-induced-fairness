import pandas as pd
from scipy.stats import ttest_ind


class GroupExplanations:
    def __init__(self, df1, df2, alpha=0.05):
        """
        Initialize the GroupExplanations class.

        Parameters:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        alpha (float): The significance level, default is 0.05.
        """
        self.df1 = df1
        self.df2 = df2
        self.alpha = alpha

    def perform_t_tests(self, side='two-sided'):
        """
        Perform two-sided t-tests for each column in the two DataFrames.

        Returns:
        pd.DataFrame: A DataFrame containing the t-test statistics, p-values, whether to reject the null hypothesis, and the means of both groups.
        """
        results = []

        for column in self.df1.columns:
            if column in self.df2.columns:
                t_stat, p_value = ttest_ind(self.df1[column].dropna(), self.df2[column].dropna(), alternative=side)
                mean1 = self.df1[column].mean()
                mean2 = self.df2[column].mean()
                results.append({
                    'feature': column,
                    't_stat': round(t_stat, 4),
                    'p_value': round(p_value, 4),
                    'Mean Group 1': round(mean1, 4),
                    'Mean Group 2': round(mean2, 4),
                    'is_significant': p_value < self.alpha
                })

        return pd.DataFrame(results)
