class FairnessCalculator:
    """
    A base class for calculating fairness metrics between protected and non-protected groups.

    Methods:
    __init__(df, group_col, prediction_col, true_col): Initialize the FairnessCalculator instance.
    _split_groups(self): Split the dataset into protected, non-protected, and population groups.
    compute(self): Calculate fairness metrics between protected and non-protected groups, which should be implemented in subclass.

    Attributes:
    df (DataFrame): The input DataFrame containing data and predictions.
    group_col (str): The column representing the group membership.
    prediction_col (str): The column containing prediction, eg:Pr(\hat Y=1).
    true_col (str): The column containing true labels, eg:Pr(Y=1).
    """

    def __init__(self, df, group_col, prediction_col, true_col):
        self.df = df
        self.group_col = group_col
        self.prediction_col = prediction_col
        self.true_col = true_col

    def _split_groups(self):
        protect_df = self.df[self.df[self.group_col] == 0]
        nonprotect_df = self.df[self.df[self.group_col] == 1]
        population_df = self.df

        return protect_df, nonprotect_df, population_df

    def compute(self):
        raise NotImplementedError("Subclasses should implement the 'compute' method.")