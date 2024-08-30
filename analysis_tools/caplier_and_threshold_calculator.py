def caplier_and_threshold(df, group_col, ps_col='Pr(S=0)', proba_col='Pr(Y=1)', caplier_ratio=0.2, threshold_ratio=0.2):
    """
    Calculate Caplier limits and thresholds based on standard deviations.

    Parameters:
    - df: DataFrame containing data.
    - group_col: Column specifying the group (0 or 1).
    - caplier_ratio: Ratio for calculating Caplier limits.
    - threshold_ratio: Ratio for calculating thresholds.

    Returns:
    - protect_caplier: Caplier limit for the protected group.
    - nonprotect_caplier: Caplier limit for the non-protected group.
    - protect_threshold: Threshold for the protected group.
    - nonprotect_threshold: Threshold for the non-protected group.
    """

    protect_df = df[df[group_col] == 0]
    nonprotect_df = df[df[group_col] == 1]

    # Check if ps_col exists in the DataFrame
    if ps_col in df.columns:
        protect_ps_std = protect_df[ps_col].std()
        nonprotect_ps_std = nonprotect_df[ps_col].std()
    else:
        protect_ps_std = 0  # Set a default value if ps_col does not exist
        nonprotect_ps_std = 0  # Set a default value if ps_col does not exist

    protect_y_std = protect_df[proba_col].std()
    nonprotect_y_std = nonprotect_df[proba_col].std()

    protect_caplier = protect_ps_std * caplier_ratio
    nonprotect_caplier = nonprotect_ps_std * caplier_ratio

    protect_threshold = protect_y_std * threshold_ratio
    nonprotect_threshold = nonprotect_y_std * threshold_ratio

    return protect_caplier, nonprotect_caplier, protect_threshold, nonprotect_threshold
