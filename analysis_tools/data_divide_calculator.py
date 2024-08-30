


def data_divide(df, group_col=None, binary_col=None, prediction_col=None, ps_col=None):
    """
    Divide the DataFrame into protected and non-protected groups, accepted and rejected groups, and extract specific columns-Pr(S=0),
    which indicates the propensity scores.

    Parameters:
    df (DataFrame): Input DataFrame containing relevant columns.
    group_col (str): Name of the column indicating group membership (0 or 1).
    binary_col (str): Name of the column indicating binary membership (0 or 1).
    prediction_col (str): Name of the column indicating prediction membership (0 or 1).
    ps_col (str): Name of the column indicating propensity scores.

    Returns:
    protect_df (DataFrame): DataFrame containing data for the protected group.
    nonprotect_df (DataFrame): DataFrame containing data for the non-protected group.
    accepted_df (DataFrame): DataFrame containing data for the accepted group.
    rejected_df (DataFrame): DataFrame containing data for the rejected group.
    pred_accepted_df (DataFrame): DataFrame containing data for the accepted group based on prediction (if available).
    pred_rejected_df (DataFrame): DataFrame containing data for the rejected group based on prediction (if available).
    protect_ps (list): List of 'Pr(S=0)' values for the protected group (if available).
    nonprotect_ps (list): List of 'Pr(S=0)' values for the non-protected group (if available).
    """

    # Initialize empty DataFrames
    protect_df = nonprotect_df = accepted_df = rejected_df = pred_accepted_df = pred_rejected_df = None
    protect_ps = nonprotect_ps = None

    # Check if group_col exists and split data if available
    if group_col and group_col in df:
        protect_df = df[df[group_col] == 0]
        nonprotect_df = df[df[group_col] == 1]
        protect_df.reset_index(drop=True, inplace=True)
        nonprotect_df.reset_index(drop=True, inplace=True)

    # Check if binary_col exists and split data if available
    if binary_col and binary_col in df:
        accepted_df = df[df[binary_col] == 1]
        rejected_df = df[df[binary_col] == 0]
        accepted_df.reset_index(drop=True, inplace=True)
        rejected_df.reset_index(drop=True, inplace=True)

    # Check if prediction_col exists and split data if available
    if prediction_col and prediction_col in df:
        pred_accepted_df = df[df[prediction_col] == 1]
        pred_rejected_df = df[df[prediction_col] == 0]
        pred_accepted_df.reset_index(drop=True, inplace=True)
        pred_rejected_df.reset_index(drop=True, inplace=True)

    # Check if ps_col exists
    if ps_col and ps_col in df:
        protect_ps = protect_df[ps_col].tolist() if protect_df is not None else None
        nonprotect_ps = nonprotect_df[ps_col].tolist() if nonprotect_df is not None else None

    return protect_df, nonprotect_df, accepted_df, rejected_df, pred_accepted_df, pred_rejected_df, protect_ps, nonprotect_ps
