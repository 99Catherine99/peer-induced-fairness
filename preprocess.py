import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


def combine_data(file_paths, feature_names_paths, common_feature_path):
    """
    :param file_paths: list of paths
    :param feature_names_paths:  list of paths
    :param common_feature_path: string
    :return:
    """
    # Read data files
    dfs = [pd.read_csv(file_path, delimiter='\t', dtype='object') for file_path in file_paths]

    # Read feature names
    sfs = [pd.read_csv(feature_names_path) for feature_names_path in feature_names_paths]

    # select features
    match_dfs = [df.loc[:, sf['Variables Name']] for df, sf in zip(dfs, sfs)]

    # Get common feature names
    f_df = pd.read_csv(common_feature_path)
    feature_names = f_df['Features Name'].tolist()

    # rename all columns names
    match_dfs = [match_df.rename(columns=dict(zip(match_df.columns, feature_names))) for match_df in match_dfs]

    # fix final outcomes' values in columns.
    match_dfs[2].loc[match_dfs[2]['final outcomes'] == '4', 'final outcomes'] = '1'
    match_dfs[2].loc[match_dfs[2]['final outcomes'] == '5', 'final outcomes'] = '4'
    match_dfs[2].loc[match_dfs[2]['final outcomes'] == '6', 'final outcomes'] = '5'


    match_dfs[0]['annual turnover'] = match_dfs[0]['annual turnover'].astype(int)
    match_dfs[0].loc[(match_dfs[0]['annual turnover'] >= 7) & (match_dfs[0]['annual turnover'] <= 16), 'annual turnover'] -= 1
    match_dfs[0].loc[(match_dfs[0]['credit balance'] == '7'), 'credit balance'] = '6'
    match_dfs[0].loc[(match_dfs[0]['credit balance'] == '8'), 'credit balance'] = '7'
    match_dfs[0].loc[(match_dfs[0]['credit balance'] == '9'), 'credit balance'] = '8'
    match_dfs[0].loc[(match_dfs[0]['credit balance'] == '10'), 'credit balance'] = '9'
    match_dfs[0].loc[(match_dfs[0]['credit balance'] == '11'), 'credit balance'] = '10'
    match_dfs[0].loc[(match_dfs[0]['credit balance'] == '12'), 'credit balance'] = '11'
    match_dfs[0].loc[(match_dfs[0]['credit balance'] == '13'), 'credit balance'] = '12'
    match_dfs[0].loc[(match_dfs[0]['credit balance'] == '14'), 'credit balance'] = '13'
    match_dfs[0].loc[(match_dfs[0]['credit balance'] == '15'), 'credit balance'] = '14'
    match_dfs[0].loc[(match_dfs[0]['credit balance'] == '16'), 'credit balance'] = '15'


    for i in [0,1]:
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '1'), 'credit balance'] = '2'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '2'), 'credit balance'] = '3'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '3'), 'credit balance'] = '4'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '4'), 'credit balance'] = '5'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '5'), 'credit balance'] = '6'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '6'), 'credit balance'] = '7'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '7'), 'credit balance'] = '8'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '8'), 'credit balance'] = '9'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '9'), 'credit balance'] = '10'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '10'), 'credit balance'] = '11'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '11'), 'credit balance'] = '12'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '12'), 'credit balance'] = '13'

        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '12'), 'credit balance'] = '1'
        match_dfs[i].loc[(match_dfs[i]['credit balance'] == '13'), 'credit balance'] = '12'

    df_combined = pd.concat(match_dfs, ignore_index=True)

    return df_combined


def replace_dk_refused(df_combined):
    # Copy the input DataFrame to avoid modifying the original data
    df_replace_dk_refused = df_combined.copy()

    # Define a dictionary of columns and values to replace with NaN
    replace_mapping = {
        'risk': [5.0, 5, '5'],
        'assets': [14.0, 15.0, 14, 15, '14', '15'],
        'liabilities': [14.0, 15.0, 14, 15, '14', '15'],
        'annual turnover': [14.0, 15.0, 14, 15, '14', '15'],
        'finance qualification for manager': [3.0, 3, '3'],
        'turnover growth rate': [5, 6, 5.0, 6.0, '5', '6'],
        'credit balance': [10.0, 11.0, 12.0, 10, 11, 12, '10', '11', '12'],
        'loss or profit': [4.0, 5.0, 4, 5, '4', '5'],
        'age': [5.0, 6.0, 5, 6, '5', '6'],
        'race': [17.0, 18.0, 17, 18, '17', '18'],
        'personal or business account': [3, 3.0, '3'],
        'obstacle to external finance': [11, 11.0, '11']
    }

    # Replace specified values with NaN in the DataFrame
    for col, values in replace_mapping.items():
        if col in df_replace_dk_refused.columns:
            for value in values:
                df_replace_dk_refused[col].replace(value, np.nan, inplace=True)

    return df_replace_dk_refused


def replace_missing_values(df_replace_dk_refused):
    """
    Replace missing values -99.99, '-99.99', ' ' to NaN in the DataFrame.

    Args:
    df_combined (pd.DataFrame): Combined and preprocessed DataFrame.

    Returns:
    replacena_df (pd.DataFrame): DataFrame with replaced missing values.
    """

    df_replace_dk_refused = df_replace_dk_refused.apply(pd.to_numeric, errors='coerce')

    replacena99_df = df_replace_dk_refused.replace(-99.99, np.nan, inplace=False)
    replacena99str_df = replacena99_df.replace('-99.99', np.nan, inplace=False)
    
    replacena_df = replacena99str_df.replace(' ', np.nan, inplace=False)
    return replacena_df


def check_col_missing_ratio(replacena_df, ratio1):
    """
    Filter columns based on the missing value ratio1, we could change the ratio1, if the missing ratio of this feature is larger than ratio1, then delect the feature.

    Args:
    replacena_df (pd.DataFrame): DataFrame with replaced missing values.
    ratio1 (float): Maximum allowed missing value ratio.

    Returns:
    del_hmc_df (pd.DataFrame): DataFrame with selected columns.
    NAN_ratios (pd.Series): Missing value ratios for each column.
    keep_columns (list): List of column names to be retained.
    """

    NAN_ratios = replacena_df.isna().sum() / replacena_df.shape[0]

    keep_columns = NAN_ratios[NAN_ratios <= ratio1].index.tolist()

    if "final outcomes" not in keep_columns:
        keep_columns.append("final outcomes")

    del_hmc_df = replacena_df[keep_columns]

    return del_hmc_df, NAN_ratios, keep_columns


def check_row_missing_ratio(del_hmc_df, ratio2):
    """
    Filter rows based on the missing value ratio2, we could change ratio2, if the missing ratio of features number of data pointis is larger than ratio2, then delect the data point.

    Args:
    del_hmc_df (pd.DataFrame): DataFrame with selected columns.
    ratio2 (float): Maximum allowed missing value ratio per row.

    Returns:
    del_hmr_df (pd.DataFrame): DataFrame with selected rows.
    """

    nan_percent = del_hmc_df.isna().sum(axis=1) / del_hmc_df.shape[1]

    del_hmr_df = del_hmc_df[nan_percent < ratio2]

    return del_hmr_df


def impute_missing_data(del_hmr_df):
    """
    Impute missing values by SimpleImputer in the DataFrame, except 'final outcomes'.

    Args:
    del_hmr_df (pd.DataFrame): DataFrame with selected rows and columns.

    Returns:
    fill_df (pd.DataFrame): DataFrame with imputed missing values.
    """


    final_outcomes_df = pd.DataFrame(del_hmr_df['final outcomes']).reset_index(drop=True)

    remaining_df = del_hmr_df.drop('final outcomes', axis=1).reset_index(drop=True)

    continuous_columns = remaining_df.columns[remaining_df.isna().any()].tolist()

    simple_imputer = SimpleImputer(strategy="most_frequent")
    remaining_df[continuous_columns] = simple_imputer.fit_transform(remaining_df[continuous_columns])

    fill_df = pd.concat([remaining_df, final_outcomes_df], axis=1)

    return fill_df


def convert_to_int(fill_df):
    """
    Convert selected columns to integer data type, except 'final outcomes'.

    Args:
    fill_df (pd.DataFrame): DataFrame with imputed missing values.

    Returns:
    int_df (pd.DataFrame): DataFrame with selected columns converted to integer data type.
    int_df is all dataframe after preprocessing, which contains 'final outcomes' with or without values.
    """

    int_df = fill_df.copy()
    int_columns = fill_df.columns.difference(['final outcomes'])
    int_df[int_columns] = int_df[int_columns].astype(int)
    return int_df


def remove_nan_final_outcomes(int_df):
    """
    Remove rows with missing final outcomes or final outcomes equal to 5.

    Args:
    int_df (pd.DataFrame): DataFrame with selected columns converted to integer data type.

    Returns:
    delfo5_df (pd.DataFrame): DataFrame with rows containing missing values or '5' final outcomes removed.
    delfo5_df is partial dataframe after preprocessing, which contains 'final outcomes' with values
    """


    delfona_df = int_df.loc[int_df['final outcomes'].notna(), :]
    delfo5_df = delfona_df[delfona_df['final outcomes'] != 5]
    delfo5_df = delfo5_df[delfona_df['final outcomes'] != 5.0]
    delfo5_df = delfo5_df[delfona_df['final outcomes'] != '5']

    delfo5_df = delfo5_df.astype(int)

    delfo5_df.reset_index(drop=True, inplace=True)

    return delfo5_df


def merge_final_outcomes(delfona_df, int_df):
    """
    Merge final outcomes and create a binary target column.

    Args:
    delfo5_df (pd.DataFrame): DataFrame with rows containing missing or '5' final outcomes removed.

    Returns:
    mergefo_df (pd.DataFrame): DataFrame with merged final outcomes--binary target column.
    mergefo_df is partial dataframe after preprocessing which contains 'final outcomes' with values and convert 'final outcomes' into Binary final outcomes
    """

    part_mergefo_df = delfona_df.copy()
    all_mergefo_df = int_df.copy()

    part_mergefo_df['Binary Y'] = np.where(part_mergefo_df['final outcomes'].isin([1, 2]), 1,
                                      np.where(part_mergefo_df['final outcomes'].isin([3, 4]), 0,
                                               part_mergefo_df['final outcomes']))

    all_mergefo_df['Binary Y'] = np.where(all_mergefo_df['final outcomes'].isin([1, 2]), 1,
                                      np.where(all_mergefo_df['final outcomes'].isin([3, 4]), 0,
                                               all_mergefo_df['final outcomes']))

    return part_mergefo_df, all_mergefo_df


def merge_attributes(part_mergefo_df, all_mergefo_df):
    """
    Merge and preprocess attributes from two dataframes.

    Parameters:
    - part_merge_df (DataFrame): Partial dataframe to merge.
    - all_merge_df (DataFrame): Full dataframe containing additional data.

    Returns:
    - merge_df (DataFrame): Merged and preprocessed dataframe.
    - int_merge_df (DataFrame): Merged dataframe with NaN values removed.
    """
    merge_df = part_mergefo_df.copy()
    all_merge_df = all_mergefo_df.copy()

    int_merge_df = all_merge_df[pd.isna(all_merge_df['Binary Y'])]
    int_merge_df = int_merge_df.reset_index(drop=True)



    annual_turnover_mapping = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6, 12: 6}

    merge_df['new annual turnover'] = merge_df['annual turnover'].map(annual_turnover_mapping)

    int_merge_df['new annual turnover'] = int_merge_df['annual turnover'].map(
        {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6, 12: 6}).fillna(99).astype(int)

    # workers number
    merge_df['new workers number'] = merge_df['workers number'].map({1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3})
    int_merge_df['new workers number'] = int_merge_df['workers number'].map(
        {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3})

    # funds rejections
    merge_df['new funds injections'] = merge_df['funds injections'].map({1: 1, 2: 2, 3: 2})
    int_merge_df['new funds injections'] = int_merge_df['funds injections'].map({1: 1, 2: 2, 3: 2})

    # sensitive attributes
    # location
    merge_df['new location'] = 0
    merge_df.loc[merge_df['location'].isin([10, 11]), 'new location'] = 1
    int_merge_df['new location'] = 0
    int_merge_df.loc[int_merge_df['location'].isin([10, 11]), 'new location'] = 1

    # gender
    merge_df['new gender'] = np.where(merge_df['gender'] == 2, 0, 1)
    int_merge_df['new gender'] = np.where(int_merge_df['gender'] == 2, 0, 1)

    # establish time
    merge_df['new establish time'] = merge_df['establish time'].map({1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1})
    int_merge_df['new establish time'] = int_merge_df['establish time'].map(
        {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1})

    # race
    merge_df['new race'] = np.where(merge_df['race'] == 1, 1, 0)
    int_merge_df['new race'] = np.where(int_merge_df['race'] == 1, 1, 0)

    # age
    merge_df['new age'] = merge_df['age'].map({1: 0, 2: 0, 3: 1, 4: 1})
    int_merge_df['new age'] = int_merge_df['age'].map({1: 0, 2: 0, 3: 1, 4: 1})

    # size
    merge_df['size'] = 3

    merge_df.loc[(merge_df['workers number'].between(1, 2)) & (merge_df['annual turnover'].between(1, 8)), 'size'] = 0
    merge_df.loc[(merge_df['workers number'] == 3) & (merge_df['annual turnover'].between(9, 10)), 'size'] = 1
    merge_df.loc[(merge_df['workers number'].between(4, 6)) & (merge_df['annual turnover'].between(11, 12)), 'size'] = 2

    int_merge_df['size'] = 3

    int_merge_df.loc[(int_merge_df['workers number'].between(1, 2)) & (int_merge_df['annual turnover'].between(1, 8)), 'size'] = 0
    int_merge_df.loc[(int_merge_df['workers number'] == 3) & (int_merge_df['annual turnover'].between(9, 10)), 'size'] = 1
    int_merge_df.loc[(int_merge_df['workers number'].between(4, 6)) & (int_merge_df['annual turnover'].between(11, 12)), 'size'] = 2

    # size
    merge_df['new size'] = np.where(merge_df['size'] == 0, 0, 1)
    int_merge_df['new size'] = np.where(int_merge_df['size'] == 0, 0, 1)

    merge_df.loc[merge_df['credit purchase'] == 2, 'credit purchase'] = 0
    int_merge_df.loc[int_merge_df['credit purchase'] == 2, 'credit purchase'] = 0

    merge_df.loc[merge_df['previous turndown'] == 2, 'previous turndown'] = 0
    int_merge_df.loc[int_merge_df['previous turndown'] == 2, 'previous turndown'] = 0

    merge_df.loc[merge_df['finance qualification for manager'] == 2, 'finance qualification for manager'] = 0
    int_merge_df.loc[int_merge_df['finance qualification for manager'] == 2, 'finance qualification for manager'] = 0

    return merge_df, int_merge_df

def export_csv(merge_df, int_merge_df):
    """
    Export dataframes to CSV files.
    Thus, we could read the exported files directly and do not need to do the preprocessing agian.

    Parameters:
    - merge_df (DataFrame): Dataframe to export.
    - int_merge_df (DataFrame): Dataframe to export (with NaN removed).
    """
    merge_df.to_csv('merge_csv.csv', index=False)
    int_merge_df.to_csv('int_merge_csv.csv', index=False)

def export_xlsx(merge_df, int_merge_df):
    """
    Export dataframes to Excel files.
    Thus, we could read the exported files directly and do not need to do the preprocessing agian

    Parameters:
    - merge_df (DataFrame): Dataframe to export.
    - int_merge_df (DataFrame): Dataframe to export (with NaN removed).
    """
    merge_df.to_excel('merge_xlsx.xlsx', index=False)
    int_merge_df.to_excel('int_merge_xlsx.xlsx', index=False)



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





rng = np.random.RandomState(42)
regressor = RandomForestRegressor(random_state=0)
N_SPLITS = 4


def get_scores_for_imputer(imputer, X, Y):
    estimator = make_pipeline(imputer, regressor)
    impute_scores = cross_val_score(
        estimator, X, Y, scoring="neg_mean_squared_error", cv=N_SPLITS
    )
    return impute_scores


def impute_zero_score(X, Y):
    imputer = SimpleImputer(
        missing_values=np.nan, add_indicator=True, strategy="constant", fill_value=0
    )
    return imputer.fit_transform(X, Y)


def get_impute_zero_score(X, Y):
    imputer = SimpleImputer(
        missing_values=np.nan, add_indicator=True, strategy="constant", fill_value=0
    )
    zero_impute_scores = get_scores_for_imputer(imputer, X, Y)
    return zero_impute_scores.mean(), zero_impute_scores.std()

def impute_knn_score(X, Y):
    imputer = KNNImputer(missing_values=np.nan, add_indicator=True)
    return imputer.fit_transform(X, Y)


def get_impute_knn_score(X, Y):
    imputer = KNNImputer(missing_values=np.nan, add_indicator=True)
    knn_impute_scores = get_scores_for_imputer(imputer, X, Y)
    return knn_impute_scores.mean(), knn_impute_scores.std()

def impute_mean(X, Y):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean", add_indicator=True)
    return imputer.fit_transform(X, Y)


def get_impute_mean(X, Y):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean", add_indicator=True)
    mean_impute_scores = get_scores_for_imputer(imputer, X, Y)
    return mean_impute_scores.mean(), mean_impute_scores.std()

def impute_iterative(X, Y):
    imputer = IterativeImputer(
        missing_values=np.nan,
        add_indicator=True,
        random_state=0,
        n_nearest_features=10,
        max_iter=1,
        sample_posterior=True,
    )
    return imputer.fit_transform(X, Y)


def get_impute_iterative(X, Y):
    imputer = IterativeImputer(
        missing_values=np.nan,
        add_indicator=True,
        random_state=0,
        n_nearest_features=3,
        max_iter=1,
        sample_posterior=True,
    )
    iterative_impute_scores = get_scores_for_imputer(imputer, X, Y)
    return iterative_impute_scores.mean(), iterative_impute_scores.std()
