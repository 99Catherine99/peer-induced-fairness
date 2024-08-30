import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MultiProportionTrend:
    def __init__(self, result_groups):
        """
        Initialize the MultiProportionTrend class.

        Parameters:
        result_groups (dict): A dictionary where keys are group names and values are lists of DataFrames.
        """
        self.result_groups = result_groups

    def calculate_proportions(self):
        """
        Calculate the proportions of 'True', 'False', and 'Unknown' statuses across the result groups.

        Returns:
        dict: A dictionary containing the proportions of each status for each group.
        """
        proportions = {status: {} for status in ['True', 'False', 'Unknown']}
        for group_name, dfs in self.result_groups.items():
            for status in proportions.keys():
                group_props = []
                for df in dfs:
                    df['is_significant'] = df['is_significant'].astype(str)  # Ensure consistent data type
                    total = len(df)
                    count = df['is_significant'].value_counts().get(status, 0)
                    proportion = count / total if total > 0 else 0
                    group_props.append(proportion)
                proportions[status][group_name] = group_props

        # Convert proportions to a DataFrame for exporting
        proportions_df = pd.DataFrame.from_dict({(i, j): proportions[i][j]
                                                for i in proportions.keys()
                                                for j in proportions[i].keys()},
                                                orient='index')
        # Export to CSV
        proportions_df.to_csv('proportions.csv')
        return proportions

    def calculate_summary_stats(self, proportions):
        """
        Calculate summary statistics (mean and standard error) for each group's proportions.

        Parameters:
        proportions (dict): A dictionary containing the proportions of each status for each group.

        Returns:
        tuple: A tuple containing the summary statistics dictionary and a DataFrame for the summary statistics.
        """
        summary_stats = {status: {} for status in proportions}
        summary_df = pd.DataFrame()
        for status, groups in proportions.items():
            for group_name, props in groups.items():
                mean_prop = np.mean(props)
                stderr_prop = np.std(props, ddof=1) / np.sqrt(len(props)) if len(props) > 1 else 0
                summary_stats[status][group_name] = (mean_prop, stderr_prop)
                # Add to the summary DataFrame
                summary_df.at[group_name, f'{status}_mean'] = mean_prop
                summary_df.at[group_name, f'{status}_stderr'] = stderr_prop
                # Save each proportion to the DataFrame
                for i, prop in enumerate(props):
                    summary_df.at[group_name, f'{status}_prop_{i+1}'] = prop
        # Export summary DataFrame to CSV
        summary_df.to_csv('summary_stats.csv')
        return summary_stats, summary_df

    def plot_trends(self):
        """
        Plot the trends of the proportions with error bars.

        Returns:
        DataFrame: The summary DataFrame containing mean proportions and standard errors.
        """
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        proportions = self.calculate_proportions()
        summary_stats, summary_df = self.calculate_summary_stats(proportions)

        plt.figure(figsize=(7, 6))
        colors = {'True': 'red', 'False': 'blue', 'Unknown': 'gray'}
        label_map = {'True': 'Not equal', 'False': 'Equal', 'Unknown': 'Unknown'}  # Mapping for labels
        for status, stats in summary_stats.items():
            groups = sorted(stats.keys())
            means = [stats[group][0] for group in groups]
            errors = [stats[group][1] for group in groups]
            plt.errorbar(groups, means, yerr=errors, fmt='-o', capsize=5, label=label_map[status], color=colors[status])

        plt.xlabel('Group imbalance level', fontsize=15)
        plt.ylabel('Average proportion', fontsize=15)
        plt.ylim(0, 1)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        plt.savefig('group_imbalance_bar_error.pdf', format='pdf', dpi=300)
        plt.show()
        return summary_df

    def plot_grouped_stacked_bar(self):
        """
        Plot a grouped stacked bar chart to show the proportions of each status across different groups.
        """
        proportions = self.calculate_proportions()
        # Adjust proportions data to ensure it's a single value
        for status in proportions:
            for group in proportions[status]:
                if isinstance(proportions[status][group], list) and len(proportions[status][group]) == 1:
                    proportions[status][group] = proportions[status][group][0]
                elif isinstance(proportions[status][group], list):
                    proportions[status][group] = np.mean(proportions[status][group])  # If there are multiple values, calculate the mean

        # Create a DataFrame
        prop_df = pd.DataFrame(proportions).T.stack().reset_index()
        prop_df.columns = ['Status', 'Group', 'Proportion']

        # Ensure the 'Proportion' column is of float type
        prop_df['Proportion'] = prop_df['Proportion'].astype(float)

        plt.figure(figsize=(7, 5))
        # Use seaborn's barplot to draw the stacked bar chart
        sns.barplot(x='Group', y='Proportion', hue='Status', data=prop_df, palette=['red', 'blue', 'gray'], ci=None)

        plt.xlabel('Group imbalance level', fontsize=15)
        plt.ylabel('Proportion', fontsize=15)

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        plt.show()
