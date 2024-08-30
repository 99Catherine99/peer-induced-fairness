import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

def tsne_visualization_2d(df, group_col, other_reduced_cols):
    """
    Perform t-SNE dimensionality reduction and visualize the data in 2D--dimension reduction and cluster, and assign colours to different groups.

    Parameters:
    df (DataFrame): Input DataFrame containing relevant columns.
    group_col (str): Name of the column indicating group membership (0 or 1).
    other_reduced_cols (list): List of column names to be dropped before t-SNE.

    Returns:
    None, but plot a scat plot.
    """

    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(df.drop(other_reduced_cols, axis=1))

    reduced_df = pd.DataFrame(reduced_data, columns=['t-SNE Component 1', 't-SNE Component 2'])

    # Set the 'group_col' column as the index
    df = df.set_index(group_col)

    protect_mask = df.index == 0
    nonprotect_mask = df.index == 1

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_df.loc[protect_mask, 't-SNE Component 1'],
                reduced_df.loc[protect_mask, 't-SNE Component 2'],
                label='protect', alpha=0.7)

    plt.scatter(reduced_df.loc[nonprotect_mask, 't-SNE Component 1'],
                reduced_df.loc[nonprotect_mask, 't-SNE Component 2'],
                label='nonprotect', alpha=0.7)

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization')
    plt.legend()
    plt.show()

# tsne_visualization(filtered_df, 'new race',
# ['race','Binary Y','final outcomes','propensity score','Binary Prediction with ps','Pr(Y=1)','Binary Prediction'])

def tsne_visualization_3d(df, group_col, other_reduced_cols):
    """
    Perform t-SNE dimensionality reduction and visualize the data in 3D--dimension reduction and cluster, and assign colours to different groups.

    Parameters:
    df (DataFrame): Input DataFrame containing relevant columns.
    group_col (str): Name of the column indicating group membership (0 or 1).
    other_reduced_cols (list): List of column names to be dropped before t-SNE, which is not the features of datapoint, eg:'Pr(Y=1)','Binary Prediction'.

    Returns:
    None, but plot a scat plot.
    """

    tsne = TSNE(n_components=3, random_state=42)
    reduced_data = tsne.fit_transform(df.drop(other_reduced_cols, axis=1))

    reduced_df = pd.DataFrame(reduced_data, columns=['t-SNE Component 1', 't-SNE Component 2', 't-SNE Component 3'])

    # Set the 'group_col' column as the index
    df = df.set_index(group_col)

    protect_mask = df.index == 0
    nonprotect_mask = df.index == 1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(reduced_df.loc[protect_mask, 't-SNE Component 1'],
               reduced_df.loc[protect_mask, 't-SNE Component 2'],
               reduced_df.loc[protect_mask, 't-SNE Component 3'],
               label='protect', alpha=0.7)

    ax.scatter(reduced_df.loc[nonprotect_mask, 't-SNE Component 1'],
               reduced_df.loc[nonprotect_mask, 't-SNE Component 2'],
               reduced_df.loc[nonprotect_mask, 't-SNE Component 3'],
               label='nonprotect', alpha=0.7)

    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    ax.set_title('t-SNE Visualization')
    ax.legend()
    plt.show()