import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau


def correlation(df, correlation_type):
    """
    Compute and visualize the correlation between selected columns in the DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame containing relevant columns.
    correlation_type (str): Type of correlation to compute ('pearson_corr' or 'kendall_corr').

    Returns:
    correlation (DataFrame): DataFrame containing the computed correlation values.
    """

    if correlation_type == 'pearson_corr':

        # Compute Pearson correlation for the entire DataFrame
        correlation_df = df.corr()
        correlation_df.to_csv('pearson_correlation_csv.csv', index=True)

    elif correlation_type == 'kendall_corr':

        # Compute Kendall correlation for selected columns
        cols_of_interest = df.columns
        corr_list = []
        for col1 in cols_of_interest:
            corr_col = []
            for col2 in cols_of_interest:
                corr, _ = kendalltau(df[col1], df[col2])
                corr_col.append(corr)
            corr_list.append(corr_col)

        correlation_df = pd.DataFrame(corr_list, columns=cols_of_interest, index=cols_of_interest)
        correlation_df.to_csv('kendall_correlation_csv.csv', index=True)

    else:
        raise ValueError('Use pearson_corr or kendall_corr for correlation_type')

    plt.figure(figsize=(20, 15))
    # Plot heatmap, setting annot=False to remove value labels
    sns.heatmap(correlation_df, cmap='coolwarm', annot=False)
    # Set font size for axis labels
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Correlation Heatmap', fontsize=25)
    plt.show()

    return correlation_df
