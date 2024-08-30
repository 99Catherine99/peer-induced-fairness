import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


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


    # 将 'annual turnover' 列中的字符串转换为整数
    match_dfs[0]['annual turnover'] = match_dfs[0]['annual turnover'].astype(int)
    # 找到 'annual turnover' 列中满足条件的行，并将对应值减去 1
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

    # match_dfs[2].loc[(match_dfs[2]['final outcomes'] == '4'), 'final outcomes'] = '1'
    # match_dfs[2].loc[(match_dfs[2]['final outcomes'] == '5'), 'final outcomes'] = '4'


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


    # print(match_dfs[2]['final outcomes'].unique()) #there is no '6'

    # 将三个DataFrame连接起来
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
    # 替换-99.99为NaN
    replacena99_df = df_replace_dk_refused.replace(-99.99, np.nan, inplace=False)
    replacena99str_df = replacena99_df.replace('-99.99', np.nan, inplace=False)
    # 替换空格为NaN
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

    # 计算每个变量的缺失比例
    NAN_ratios = replacena_df.isna().sum() / replacena_df.shape[0]

    # 选择缺失比例小于等于50%的变量
    keep_columns = NAN_ratios[NAN_ratios <= ratio1].index.tolist()

    # 确保"final outcome"这个变量不会被删除
    if "final outcomes" not in keep_columns:
        keep_columns.append("final outcomes")

    # 保留需要的变量并创建新的DataFrame
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

    # 创建一个包含"final outcomes"列的DataFrame
    final_outcomes_df = pd.DataFrame(del_hmr_df['final outcomes']).reset_index(drop=True)

    # 删除"final outcomes"列，并保留其他列
    remaining_df = del_hmr_df.drop('final outcomes', axis=1).reset_index(drop=True)

    # 获取有缺失值的连续变量列
    continuous_columns = remaining_df.columns[remaining_df.isna().any()].tolist()

    # 使用 SimpleImputer 对象对连续变量进行插补
    simple_imputer = SimpleImputer(strategy="most_frequent")
    remaining_df[continuous_columns] = simple_imputer.fit_transform(remaining_df[continuous_columns])

    # 将插补后的数据与"final outcomes"列合并成一个新的DataFrame
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

    # 删除final outcomes为空的行
    delfona_df = int_df.loc[int_df['final outcomes'].notna(), :]
    delfo5_df = delfona_df[delfona_df['final outcomes'] != 5]
    delfo5_df = delfo5_df[delfona_df['final outcomes'] != 5.0]
    delfo5_df = delfo5_df[delfona_df['final outcomes'] != '5']
    # 将数据类型转换为整数类型
    delfo5_df = delfo5_df.astype(int)
    # 重置索引
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

    # 将final outcomes转换为二进制类别变量
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

    # 筛选出 'Binary Y' 列中值为 NaN 的行
    int_merge_df = all_merge_df[pd.isna(all_merge_df['Binary Y'])]
    # 如果需要重置行索引
    int_merge_df = int_merge_df.reset_index(drop=True)


    # turnover分析的时候需要，预测的时候还是按照多数进行
    # 创建映射字典
    annual_turnover_mapping = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6, 12: 6}
    # 使用map函数创建新列
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
    # 添加'size'列并设置初始值
    merge_df['size'] = 3
    # 根据条件设置'size'的值
    merge_df.loc[(merge_df['workers number'].between(1, 2)) & (merge_df['annual turnover'].between(1, 8)), 'size'] = 0
    merge_df.loc[(merge_df['workers number'] == 3) & (merge_df['annual turnover'].between(9, 10)), 'size'] = 1
    merge_df.loc[(merge_df['workers number'].between(4, 6)) & (merge_df['annual turnover'].between(11, 12)), 'size'] = 2

    int_merge_df['size'] = 3
    # 根据条件设置'size'的值
    int_merge_df.loc[(int_merge_df['workers number'].between(1, 2)) & (int_merge_df['annual turnover'].between(1, 8)), 'size'] = 0
    int_merge_df.loc[(int_merge_df['workers number'] == 3) & (int_merge_df['annual turnover'].between(9, 10)), 'size'] = 1
    int_merge_df.loc[(int_merge_df['workers number'].between(4, 6)) & (int_merge_df['annual turnover'].between(11, 12)), 'size'] = 2

    # size
    merge_df['new size'] = np.where(merge_df['size'] == 0, 0, 1)
    int_merge_df['new size'] = np.where(int_merge_df['size'] == 0, 0, 1)

    # # 将'merge_df'中'business innovation'为1的值改为0，为0的值改为1
    # merge_df['business innovation'] = merge_df['business innovation'].replace({0: 1, 1: 0})
    # int_merge_df['business innovation'] = int_merge_df['business innovation'].replace({0: 1, 1: 0})
    #
    # merge_df['product or service development'] = merge_df['product or service development'].replace({0: 1, 1: 0})
    # int_merge_df['product or service development'] = int_merge_df['product or service development'].replace({0: 1, 1: 0})

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
    # 将DataFrame导出为Excel文件
    merge_df.to_excel('merge_xlsx.xlsx', index=False)
    int_merge_df.to_excel('int_merge_xlsx.xlsx', index=False)





# def combine_data(file_paths, feature_names_paths, common_feature_path):
#     """
#     :param file_paths: list of paths
#     :param feature_names_paths:  list of paths
#     :param common_feature_path: string
#     :return:
#     """
#     # Read data files
#     dfs = [pd.read_csv(file_path, delimiter='\t', dtype='object') for file_path in file_paths]
#
#     # Read feature names
#     sfs = [pd.read_csv(feature_names_path) for feature_names_path in feature_names_paths]
#
#     # select features
#     match_dfs = [df.loc[:, sf['Variables Name']] for df, sf in zip(dfs, sfs)]
#
#     # Get common feature names
#     f_df = pd.read_csv(common_feature_path)
#     feature_names = f_df['Features Name'].tolist()
#
#     # rename all columns names
#     match_dfs = [match_df.rename(columns=dict(zip(match_df.columns, feature_names))) for match_df in match_dfs]
#
#     # fix final outcomes' values in columns.
#     match_dfs[2].loc[match_dfs[2]['final outcomes'] == '4', 'final outcomes'] = '1'
#     match_dfs[2].loc[match_dfs[2]['final outcomes'] == '5', 'final outcomes'] = '4'
#     match_dfs[2].loc[match_dfs[2]['final outcomes'] == '6', 'final outcomes'] = '5'
#
#
#     # 将 'annual turnover' 列中的字符串转换为整数
#     match_dfs[0]['annual turnover'] = match_dfs[0]['annual turnover'].astype(int)
#     # 找到 'annual turnover' 列中满足条件的行，并将对应值减去 1
#     match_dfs[0].loc[(match_dfs[0]['annual turnover'] >= 7) & (match_dfs[0]['annual turnover'] <= 16), 'annual turnover'] -= 1
#     match_dfs[0].loc[(match_dfs[0]['credit balance'] == '7'), 'credit balance'] = '6'
#     match_dfs[0].loc[(match_dfs[0]['credit balance'] == '8'), 'credit balance'] = '7'
#     match_dfs[0].loc[(match_dfs[0]['credit balance'] == '9'), 'credit balance'] = '8'
#     match_dfs[0].loc[(match_dfs[0]['credit balance'] == '10'), 'credit balance'] = '9'
#     match_dfs[0].loc[(match_dfs[0]['credit balance'] == '11'), 'credit balance'] = '10'
#     match_dfs[0].loc[(match_dfs[0]['credit balance'] == '12'), 'credit balance'] = '11'
#     match_dfs[0].loc[(match_dfs[0]['credit balance'] == '13'), 'credit balance'] = '12'
#     match_dfs[0].loc[(match_dfs[0]['credit balance'] == '14'), 'credit balance'] = '13'
#     match_dfs[0].loc[(match_dfs[0]['credit balance'] == '15'), 'credit balance'] = '14'
#     match_dfs[0].loc[(match_dfs[0]['credit balance'] == '16'), 'credit balance'] = '15'
#
#     # match_dfs[2].loc[(match_dfs[2]['final outcomes'] == '4'), 'final outcomes'] = '1'
#     # match_dfs[2].loc[(match_dfs[2]['final outcomes'] == '5'), 'final outcomes'] = '4'
#
#
#     for i in [0,1]:
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '1'), 'credit balance'] = '2'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '2'), 'credit balance'] = '3'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '3'), 'credit balance'] = '4'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '4'), 'credit balance'] = '5'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '5'), 'credit balance'] = '6'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '6'), 'credit balance'] = '7'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '7'), 'credit balance'] = '8'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '8'), 'credit balance'] = '9'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '9'), 'credit balance'] = '10'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '10'), 'credit balance'] = '11'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '11'), 'credit balance'] = '12'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '12'), 'credit balance'] = '13'
#
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '12'), 'credit balance'] = '1'
#         match_dfs[i].loc[(match_dfs[i]['credit balance'] == '13'), 'credit balance'] = '12'
#
#
#     # print(match_dfs[2]['final outcomes'].unique()) #there is no '6'
#
#     # 将三个DataFrame连接起来
#     df_combined = pd.concat(match_dfs, ignore_index=True)
#
#     return df_combined
#
#
# def replace_dk_refused(df_combined):
#     # Copy the input DataFrame to avoid modifying the original data
#     df_replace_dk_refused = df_combined.copy()
#
#     # Define a dictionary of columns and values to replace with NaN
#     replace_mapping = {
#         'risk': [5.0, 5, '5'],
#         'assets': [14.0, 15.0, 14, 15, '14', '15'],
#         'liabilities': [14.0, 15.0, 14, 15, '14', '15'],
#         'annual turnover': [14.0, 15.0, 14, 15, '14', '15'],
#         'finance qualification for manager': [3.0, 3, '3'],
#         'turnover growth rate': [5, 6, 5.0, 6.0, '5', '6'],
#         'credit balance': [10.0, 11.0, 12.0, 10, 11, 12, '10', '11', '12'],
#         'loss or profit': [4.0, 5.0, 4, 5, '4', '5'],
#         'age': [5.0, 6.0, 5, 6, '5', '6'],
#         'race': [17.0, 18.0, 17, 18, '17', '18'],
#         'personal or business account': [3, 3.0, '3'],
#         'obstacle to external finance': [11, 11.0, '11']
#     }
#
#     # Replace specified values with NaN in the DataFrame
#     for col, values in replace_mapping.items():
#         if col in df_replace_dk_refused.columns:
#             for value in values:
#                 df_replace_dk_refused[col].replace(value, np.nan, inplace=True)
#
#     return df_replace_dk_refused
#
#
# def replace_missing_values(df_replace_dk_refused):
#     """
#     Replace missing values -99.99, '-99.99', ' ' to NaN in the DataFrame.
#
#     Args:
#     df_combined (pd.DataFrame): Combined and preprocessed DataFrame.
#
#     Returns:
#     replacena_df (pd.DataFrame): DataFrame with replaced missing values.
#     """
#
#     df_replace_dk_refused = df_replace_dk_refused.apply(pd.to_numeric, errors='coerce')
#     # 替换-99.99为NaN
#     replacena99_df = df_replace_dk_refused.replace(-99.99, np.nan, inplace=False)
#     replacena99str_df = replacena99_df.replace('-99.99', np.nan, inplace=False)
#     # 替换空格为NaN
#     replacena_df = replacena99str_df.replace(' ', np.nan, inplace=False)
#     return replacena_df
#
#
# def check_col_missing_ratio(replacena_df, ratio1):
#     """
#     Filter columns based on the missing value ratio1, we could change the ratio1, if the missing ratio of this feature is larger than ratio1, then delect the feature.
#
#     Args:
#     replacena_df (pd.DataFrame): DataFrame with replaced missing values.
#     ratio1 (float): Maximum allowed missing value ratio.
#
#     Returns:
#     del_hmc_df (pd.DataFrame): DataFrame with selected columns.
#     NAN_ratios (pd.Series): Missing value ratios for each column.
#     keep_columns (list): List of column names to be retained.
#     """
#
#     # 计算每个变量的缺失比例
#     NAN_ratios = replacena_df.isna().sum() / replacena_df.shape[0]
#
#     # 选择缺失比例小于等于50%的变量
#     keep_columns = NAN_ratios[NAN_ratios <= ratio1].index.tolist()
#
#     # 确保"final outcome"这个变量不会被删除
#     if "final outcomes" not in keep_columns:
#         keep_columns.append("final outcomes")
#
#     # 保留需要的变量并创建新的DataFrame
#     del_hmc_df = replacena_df[keep_columns]
#
#     return del_hmc_df, NAN_ratios, keep_columns
#
#
# def check_row_missing_ratio(del_hmc_df, ratio2):
#     """
#     Filter rows based on the missing value ratio2, we could change ratio2, if the missing ratio of features number of data pointis is larger than ratio2, then delect the data point.
#
#     Args:
#     del_hmc_df (pd.DataFrame): DataFrame with selected columns.
#     ratio2 (float): Maximum allowed missing value ratio per row.
#
#     Returns:
#     del_hmr_df (pd.DataFrame): DataFrame with selected rows.
#     """
#
#     nan_percent = del_hmc_df.isna().sum(axis=1) / del_hmc_df.shape[1]
#
#     del_hmr_df = del_hmc_df[nan_percent < ratio2]
#
#     return del_hmr_df
#
#
# def convert_to_int(del_hmr_df):
#     """
#     Convert selected columns to integer data type, except 'final outcomes',
#     while preserving NaN values.
#
#     Args:
#     del_hmr_df (pd.DataFrame): DataFrame with imputed missing values.
#
#     Returns:
#     int_df (pd.DataFrame): DataFrame with selected columns converted to integer data type.
#     int_df is all dataframe after preprocessing, which contains 'final outcomes' with or without values.
#     """
#
#     int_df = del_hmr_df.copy()
#     int_columns = del_hmr_df.columns.difference(['final outcomes'])
#
#     # Convert selected columns to integer type, preserving NaN values
#     for col in int_columns:
#         int_df[col] = pd.to_numeric(int_df[col], errors='coerce').astype(pd.Int64Dtype())
#
#     return int_df
#
#
# def remove_nan_final_outcomes(int_df):
#     """
#     Remove rows with missing final outcomes or final outcomes equal to 5.
#
#     Args:
#     int_df (pd.DataFrame): DataFrame with selected columns converted to integer data type.
#
#     Returns:
#     delfo5_df (pd.DataFrame): DataFrame with rows containing missing values or '5' final outcomes removed.
#     delfo5_df is partial dataframe after preprocessing, which contains 'final outcomes' with values
#     """
#
#     # 删除final outcomes为空的行
#     delfona_df = int_df[int_df['final outcomes'].notna()]
#     # 删除final outcomes等于5的行
#     delfo5_df = delfona_df[(delfona_df['final outcomes'] != 5) &
#                            (delfona_df['final outcomes'] != 5.0) &
#                            (delfona_df['final outcomes'] != '5')]
#
#     # 确保所有列的数据类型为Int64，保留NaN值
#     for col in delfo5_df.columns:
#         if col != 'final outcomes':
#             delfo5_df[col] = pd.to_numeric(delfo5_df[col], errors='coerce').astype(pd.Int64Dtype())
#
#     # 重置索引
#     delfo5_df.reset_index(drop=True, inplace=True)
#
#     return delfo5_df
#
#
#
# def impute_missing_data(delfo5_df, int_df):
#     """
#     Impute missing values by SimpleImputer in the DataFrame, except 'final outcomes'.
#
#     Args:
#     int_df (pd.DataFrame): First DataFrame with missing values.
#     delfo5_df (pd.DataFrame): Second DataFrame with missing values.
#
#     Returns:
#     fill_int_df (pd.DataFrame): First DataFrame with imputed missing values.
#     fill_delfo5_df (pd.DataFrame): Second DataFrame with imputed missing values.
#     """
#     def impute_df(df):
#         # 创建一个包含"final outcomes"列的DataFrame
#         final_outcomes_df = pd.DataFrame(df['final outcomes']).reset_index(drop=True)
#
#         # 删除"final outcomes"列，并保留其他列
#         remaining_df = df.drop('final outcomes', axis=1).reset_index(drop=True)
#
#         # 获取有缺失值的连续变量列
#         continuous_columns = remaining_df.columns[remaining_df.isna().any()].tolist()
#
#         # 使用 SimpleImputer 对象对连续变量进行插补
#         simple_imputer = SimpleImputer(strategy="most_frequent")
#         remaining_df[continuous_columns] = simple_imputer.fit_transform(remaining_df[continuous_columns])
#
#         # 将插补后的数据与"final outcomes"列合并成一个新的DataFrame
#         fill_df = pd.concat([remaining_df, final_outcomes_df], axis=1)
#
#         for col in fill_df.columns:
#             fill_df[col] = pd.to_numeric(fill_df[col], errors='coerce').astype(pd.Int64Dtype())
#
#         return fill_df
#
#     # 对 int_df 和 delfo5_df 分别进行插补
#     fill_int_df = impute_df(int_df)
#     fill_delfo5_df = impute_df(delfo5_df)
#
#     return fill_delfo5_df, fill_int_df
#
# def merge_final_outcomes(fill_delfo5_df, fill_int_df):
#     """
#     Merge final outcomes and create a binary target column.
#
#     Args:
#     delfo5_df (pd.DataFrame): DataFrame with rows containing missing or '5' final outcomes removed.
#
#     Returns:
#     mergefo_df (pd.DataFrame): DataFrame with merged final outcomes--binary target column.
#     mergefo_df is partial dataframe after preprocessing which contains 'final outcomes' with values and convert 'final outcomes' into Binary final outcomes
#     """
#
#     part_mergefo_df = fill_delfo5_df.copy()
#     all_mergefo_df = fill_int_df.copy()
#
#     # 将final outcomes转换为二进制类别变量
#     part_mergefo_df['Binary Y'] = np.where(part_mergefo_df['final outcomes'].isin([1, 2]), 1,
#                                       np.where(part_mergefo_df['final outcomes'].isin([3, 4]), 0,
#                                                part_mergefo_df['final outcomes']))
#
#     all_mergefo_df['Binary Y'] = np.where(all_mergefo_df['final outcomes'].isin([1, 2]), 1,
#                                       np.where(all_mergefo_df['final outcomes'].isin([3, 4]), 0,
#                                                all_mergefo_df['final outcomes']))
#
#     return part_mergefo_df, all_mergefo_df
#
#
#
# def merge_attributes(part_mergefo_df, all_mergefo_df):
#     """
#     Merge and preprocess attributes from two dataframes.
#
#     Parameters:
#     - part_merge_df (DataFrame): Partial dataframe to merge.
#     - all_merge_df (DataFrame): Full dataframe containing additional data.
#
#     Returns:
#     - merge_df (DataFrame): Merged and preprocessed dataframe.
#     - int_merge_df (DataFrame): Merged dataframe with NaN values removed.
#     """
#     merge_df = part_mergefo_df.copy()
#     all_merge_df = all_mergefo_df.copy()
#
#     # 筛选出 'Binary Y' 列中值为 NaN 的行
#     int_merge_df = all_merge_df[pd.isna(all_merge_df['Binary Y'])]
#     # 如果需要重置行索引
#     int_merge_df = int_merge_df.reset_index(drop=True)
#
#
#     # turnover分析的时候需要，预测的时候还是按照多数进行
#     # 创建映射字典
#     annual_turnover_mapping = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6, 12: 6}
#     # 使用map函数创建新列
#     merge_df['new annual turnover'] = merge_df['annual turnover'].map(annual_turnover_mapping)
#
#     int_merge_df['new annual turnover'] = int_merge_df['annual turnover'].map(
#         {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 5, 9: 5, 10: 6, 11: 6, 12: 6}).fillna(99).astype(int)
#
#     # workers number
#     merge_df['new workers number'] = merge_df['workers number'].map({1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3})
#     int_merge_df['new workers number'] = int_merge_df['workers number'].map(
#         {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3})
#
#     # funds rejections
#     merge_df['new funds injections'] = merge_df['funds injections'].map({1: 1, 2: 2, 3: 2})
#     int_merge_df['new funds injections'] = int_merge_df['funds injections'].map({1: 1, 2: 2, 3: 2})
#
#     # sensitive attributes
#     # location
#     merge_df['new location'] = 0
#     merge_df.loc[merge_df['location'].isin([10, 11]), 'new location'] = 1
#     int_merge_df['new location'] = 0
#     int_merge_df.loc[int_merge_df['location'].isin([10, 11]), 'new location'] = 1
#
#     # gender
#     merge_df['new gender'] = np.where(merge_df['gender'] == 2, 0, 1)
#     int_merge_df['new gender'] = np.where(int_merge_df['gender'] == 2, 0, 1)
#
#     # establish time
#     merge_df['new establish time'] = merge_df['establish time'].map({1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1})
#     int_merge_df['new establish time'] = int_merge_df['establish time'].map(
#         {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1})
#
#     # race
#     merge_df['new race'] = np.where(merge_df['race'] == 1, 1, 0)
#     int_merge_df['new race'] = np.where(int_merge_df['race'] == 1, 1, 0)
#
#     # age
#     merge_df['new age'] = merge_df['age'].map({1: 0, 2: 0, 3: 1, 4: 1})
#     int_merge_df['new age'] = int_merge_df['age'].map({1: 0, 2: 0, 3: 1, 4: 1})
#
#     # size
#     # 添加'size'列并设置初始值
#     merge_df['size'] = 3
#     # 根据条件设置'size'的值
#     merge_df.loc[(merge_df['workers number'].between(1, 2)) & (merge_df['annual turnover'].between(1, 8)), 'size'] = 0
#     merge_df.loc[(merge_df['workers number'] == 3) & (merge_df['annual turnover'].between(9, 10)), 'size'] = 1
#     merge_df.loc[(merge_df['workers number'].between(4, 6)) & (merge_df['annual turnover'].between(11, 12)), 'size'] = 2
#
#     int_merge_df['size'] = 3
#     # 根据条件设置'size'的值
#     int_merge_df.loc[(int_merge_df['workers number'].between(1, 2)) & (int_merge_df['annual turnover'].between(1, 8)), 'size'] = 0
#     int_merge_df.loc[(int_merge_df['workers number'] == 3) & (int_merge_df['annual turnover'].between(9, 10)), 'size'] = 1
#     int_merge_df.loc[(int_merge_df['workers number'].between(4, 6)) & (int_merge_df['annual turnover'].between(11, 12)), 'size'] = 2
#
#     # size
#     merge_df['new size'] = np.where(merge_df['size'] == 0, 0, 1)
#     int_merge_df['new size'] = np.where(int_merge_df['size'] == 0, 0, 1)
#
#     # # 将'merge_df'中'business innovation'为1的值改为0，为0的值改为1
#     # merge_df['business innovation'] = merge_df['business innovation'].replace({0: 1, 1: 0})
#     # int_merge_df['business innovation'] = int_merge_df['business innovation'].replace({0: 1, 1: 0})
#     #
#     # merge_df['product or service development'] = merge_df['product or service development'].replace({0: 1, 1: 0})
#     # int_merge_df['product or service development'] = int_merge_df['product or service development'].replace({0: 1, 1: 0})
#
#     merge_df.loc[merge_df['credit purchase'] == 2, 'credit purchase'] = 0
#     int_merge_df.loc[int_merge_df['credit purchase'] == 2, 'credit purchase'] = 0
#
#     merge_df.loc[merge_df['previous turndown'] == 2, 'previous turndown'] = 0
#     int_merge_df.loc[int_merge_df['previous turndown'] == 2, 'previous turndown'] = 0
#
#     merge_df.loc[merge_df['finance qualification for manager'] == 2, 'finance qualification for manager'] = 0
#     int_merge_df.loc[int_merge_df['finance qualification for manager'] == 2, 'finance qualification for manager'] = 0
#
#     return merge_df, int_merge_df
#
# def export_csv(merge_df, int_merge_df):
#     """
#     Export dataframes to CSV files.
#     Thus, we could read the exported files directly and do not need to do the preprocessing agian.
#
#     Parameters:
#     - merge_df (DataFrame): Dataframe to export.
#     - int_merge_df (DataFrame): Dataframe to export (with NaN removed).
#     """
#     merge_df.to_csv('merge_csv.csv', index=False)
#     int_merge_df.to_csv('int_merge_csv.csv', index=False)
#
# def export_xlsx(merge_df, int_merge_df):
#     """
#     Export dataframes to Excel files.
#     Thus, we could read the exported files directly and do not need to do the preprocessing agian
#
#     Parameters:
#     - merge_df (DataFrame): Dataframe to export.
#     - int_merge_df (DataFrame): Dataframe to export (with NaN removed).
#     """
#     # 将DataFrame导出为Excel文件
#     merge_df.to_excel('merge_xlsx.xlsx', index=False)
#     int_merge_df.to_excel('int_merge_xlsx.xlsx', index=False)