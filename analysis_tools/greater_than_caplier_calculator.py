def analysis_matching(matched_df, caplier, comparison_col):
    """
    Perform analysis on matched data to assess the quality of the matching.

    Parameters:
    matched_df (DataFrame): DataFrame after PSM individual matching.
    caplier (float): Caplier calculated by caplier_and_threshold_calculator.
    comparison_col (str): Column needed to make the comparison, eg: Pr(Y=1) difference of treatment group and matched control group.

    Returns:
    max_abs_difference (float): Maximum absolute difference in the specified column--comparison_col.
    counts_greater_than_caplier (int): Number of values of comparison_col greater than the caplier.
    indices_greater_than_caplier (list): List of indices with values of comparison_col greater than the caplier.
    unique_counts_greater_than_caplier (int): Number of unique indices with values of comparison_col greater than the caplier.
    """

    max_abs_difference = matched_df[comparison_col].max()
    counts_greater_than_caplier = (matched_df[comparison_col] > caplier).sum()
    indices_greater_than_caplier = matched_df[matched_df[comparison_col] > caplier]['treatment_index'].tolist()
    unique_counts_greater_than_caplier = len(set(indices_greater_than_caplier))

    return max_abs_difference, counts_greater_than_caplier, indices_greater_than_caplier, unique_counts_greater_than_caplier