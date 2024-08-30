import pandas as pd

def one_hot_encode_dataframe(df):
    """
    Perform one-hot encoding for all variables in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be one-hot encoded, we could input the dataframe after all the other preprocess.

    Returns:
    encoded_df (pandas.DataFrame): The new DataFrame with one-hot encoded columns.
    """

    encoded_df = pd.get_dummies(df, columns=df.columns)
    return encoded_df

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class FeatureEncoder:
    def __init__(self, df):
        # Initialize the FeatureEncoder with a DataFrame
        self.df = df

    def label_encode(self, columns):
        """
        Encode specified columns using Label Encoding.

        Parameters:
        - columns (list): A list of column names to be label encoded.

        Returns:
        - df (DataFrame): The DataFrame with label encoded columns.
        """
        encoder = LabelEncoder()
        for col in columns:
            self.df[col] = encoder.fit_transform(self.df[col])
        return self.df

    def one_hot_encode(self, columns):
        """
        Perform one-hot encoding on specified columns.

        Parameters:
        - columns (list): A list of column names to be one-hot encoded.

        Returns:
        - df (DataFrame): The DataFrame with one-hot encoded columns.
        """
        return pd.get_dummies(self.df, columns=columns)

    def custom_encode(self, column, mapping):
        """
        Encode a specified column using a custom mapping.

        Parameters:
        - column (str): The name of the column to be custom encoded.
        - mapping (dict): A dictionary containing the mapping for encoding.

        Returns:
        - df (DataFrame): The DataFrame with the column custom encoded.
        """
        self.df[column] = self.df[column].map(mapping)
        return self.df


