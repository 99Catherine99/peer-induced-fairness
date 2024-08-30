class UniqueValueCounter:
    def __init__(self, df, variable_names):
        """
        Initialize the UniqueValueCounter class.

        Parameters:
        - df (DataFrame): Input DataFrame containing the variables.
        - variable_names (list): List of variable names to count unique values for.

        Returns:
        None.
        """
        self.df = df
        self.variable_names = variable_names

    def count_unique_values(self):
        """
        Count unique values and their counts and proportions for each variable in the DataFrame.
        Print the results.

        Returns:
        unique_counts (dict): A dictionary containing variable names as keys and unique value counts as values.
        """
        unique_counts = {}
        for variable_name in self.variable_names:
            unique_values = self.df[variable_name].unique()
            unique_counts[variable_name] = {}

            for value in unique_values:
                count = (self.df[variable_name] == value).sum()
                proportion = count / len(self.df)

                unique_counts[variable_name][value] = {'count': count, 'proportion': proportion}
                print(f'Variable: {variable_name}, Value: {value}, Count: {count}, Proportion: {proportion:.2%}')
