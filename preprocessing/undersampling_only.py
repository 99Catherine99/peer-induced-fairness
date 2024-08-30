import pandas as pd
import numpy as np

class ImbalanceDataSampler:
    def __init__(self, df, only_change_column, base_count, imbalance_ratios, repeats_per_ratio):
        """
        Initialize the ImbalanceDataSampler class.

        Parameters:
        - df: The pandas DataFrame containing the data.
        - only_change_column: The column name used to distinguish between the two groups, assumed to have values 0 or 1.
        - base_count: The total number of data points to sample.
        - imbalance_ratios: An array containing the different imbalance ratios.
        - repeats_per_ratio: The number of repetitions for each ratio.
        """
        self.df = df
        self.only_change_column = only_change_column
        self.base_count = base_count
        self.imbalance_ratios = imbalance_ratios
        self.repeats_per_ratio = repeats_per_ratio
        self.final_adjusted_dfs = []
        self.final_adjusted_info_dfs = []

    def generate_sampled_dataframes(self):
        """
        Generate sampled datasets based on specified imbalance ratios and repetition counts.

        Returns:
        - dataframes: A list of dataframes for each imbalance level.
        """
        group_1 = self.df[self.df[self.only_change_column] == 1]
        group_0 = self.df[self.df[self.only_change_column] == 0]

        for ratio_idx, ratio in enumerate(self.imbalance_ratios):
            for repeat_idx in range(self.repeats_per_ratio):
                # Set the random seed to ensure reproducibility
                seed = 1000 * ratio_idx + repeat_idx
                np.random.seed(seed)
                count_0 = int(self.base_count * ratio)
                count_1 = self.base_count - count_0

                sampled_0 = group_0.sample(n=count_0, replace=False, random_state=seed)
                sampled_1 = group_1.sample(n=count_1, replace=False, random_state=seed)

                df = pd.concat([sampled_0, sampled_1])
                self.final_adjusted_dfs.append(df)

                # Record information about each sampled dataset
                self.final_adjusted_info_dfs.append({
                    'imbalance_ratio': ratio,
                    'repeat_number': repeat_idx + 1,
                    'count_0': count_0,
                    'count_1': count_1,
                    'total_count': count_0 + count_1
                })

        adjusted_info_df = pd.DataFrame(self.final_adjusted_info_dfs)

        return self.final_adjusted_dfs, adjusted_info_df

    def export_adjusted_dfs(self):
        """
        Export the sampled dataframes to CSV files.
        """
        if not self.final_adjusted_dfs:
            print("No Useful Dataframe.")
            return

        for i, adjusted_df in enumerate(self.final_adjusted_dfs):
            file_name = f"dataframe_{i}.csv"
            adjusted_df.to_csv(file_name, index=False)
