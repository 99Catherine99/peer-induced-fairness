class AverageConsistencyCalculator:
    def __init__(self, calculator_instances):
        """
        Initialize the average consistency calculator.
        :param calculator_instances: A list of MultiConsistencyCalculator instances.
        """
        self.calculator_instances = calculator_instances

    def average_consistency_table(self):
        # Retrieve all consistency tables
        consistency_tables = [calc.consistency_table() for calc in self.calculator_instances]

        # Convert all tables to percentage numbers
        def parse_percentage(val):
            return float(val.strip('%')) if isinstance(val, str) else val

        # Convert percentage strings to numeric values
        tables_numeric = [df.applymap(parse_percentage) for df in consistency_tables]

        # Calculate the average of all tables
        average_table = pd.concat(tables_numeric).groupby(level=0).mean()

        # Convert numeric values back to percentage strings
        average_table = average_table.applymap(lambda x: f"{x:.2f}%")
        # Export to CSV
        average_table.to_csv('average_consistency_table.csv')
        return average_table

    def cv_consistency_table(self):
        """
        Calculate the coefficient of variation (CV) of the consistency ratios between all pairs of DataFrames in the list.
        Returns a DataFrame containing the index combinations of each pair of DataFrames and their CV.
        """
        # Retrieve all consistency tables
        consistency_tables = [calc.consistency_table() for calc in self.calculator_instances]

        # Convert all tables to percentage numbers
        def parse_percentage(val):
            return float(val.strip('%')) if isinstance(val, str) else val

        # Convert percentage strings to numeric values
        tables_numeric = [df.applymap(parse_percentage) for df in consistency_tables]

        # Calculate the standard deviation and mean of all tables
        std_dev_table = pd.concat(tables_numeric).groupby(level=0).std()
        mean_table = pd.concat(tables_numeric).groupby(level=0).mean()

        # Calculate the coefficient of variation, CV = (Standard Deviation / Mean) * 100
        # Avoid division by zero
        cv_table = std_dev_table / mean_table
        cv_table = cv_table.applymap(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
        # Export to CSV
        cv_table.to_csv('cv_consistency_table.csv')

        return cv_table


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RatioLinePlotter:
    def __init__(self, *dataframes):
        """
        Initialize, accepting multiple DataFrames as input.
        :param dataframes: Multiple DataFrame objects.
        """
        self.dataframes = [df.applymap(self._convert_percent_to_float) for df in dataframes]

    def _convert_percent_to_float(self, value):
        """
        Convert to float if the value is a string and contains a percentage sign.
        """
        try:
            return float(value.strip('%')) / 100 if isinstance(value, str) and '%' in value else value
        except ValueError:
            return np.nan

    def plot_individual_lines(self):
        """
        Plot individual line charts for each row and save as separate image files.
        """
        sns.set_context("talk", rc={"lines.linewidth": 2.5})

        for row_index in range(1, len(self.dataframes[0])):  # Start from the second row
            plt.figure(figsize=(7, 6))
            row_values = np.array([df.iloc[row_index].values for df in self.dataframes])
            valid_mask = ~np.isnan(row_values)
            means = np.nanmean(row_values, axis=0)
            errors = np.nanstd(row_values, axis=0) / np.sqrt(valid_mask.sum(axis=0))
            valid_columns = np.where(np.nanmean(valid_mask, axis=0) > 0)[0]

            plt.errorbar(valid_columns + 1, means[valid_columns], yerr=errors[valid_columns], fmt='-o', color='black', capsize=5,
                         label=f'Row {row_index + 1}')
            plt.xlabel('Group imbalance level', fontsize=15)
            plt.ylabel('Invariance rate', fontsize=15)

            # Set x-axis ticks and labels based on the actual number of data points in each row
            x_ticks = valid_columns + 1
            x_labels = [f'{(i + 1) * 10}%' for i in valid_columns]
            plt.xticks(x_ticks, x_labels)

            plt.ylim(0.7, 1)  # Adjust the y-axis range as needed
            # plt.legend(loc='lower right', fontsize=12)
            # plt.grid(True, color='#D3D3D3', alpha=0.5)

            # Save image
            plt.savefig(f'consistency_error_bar_row_{row_index + 1}.png', format='png', dpi=300)
            plt.show()
            # plt.close()  # Close the plot before starting the next one

    def plot(self):
        """
        Plot line charts showing the average and standard error from the second row onwards for each DataFrame, ignoring NaN values.
        """
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        plt.figure(figsize=(10,6))
        # Calculate the average and standard error
        max_columns = 0  # Used to record the maximum number of columns
        # legend_labels = [f'$\omega_i={20 + 10 * i}\%$' for i in range(len(self.dataframes[0]) - 1)]  # Generate LaTeX-style legend labels
        legend_labels = ['$\omega_i=16.33\%$', '$\omega_i=21.33\%$', '$\omega_i=26.33\%$', '$\omega_i=31.33\%$', '$\omega_i=36.33\%$']
        offsets = np.linspace(-0.2, 0.2, len(legend_labels))  # Generate left-right offsets

        for idx, row_index in enumerate(range(1, len(self.dataframes[0]))):  # Start from the second row
            row_values = np.array([df.iloc[row_index].values for df in self.dataframes])
            valid_mask = ~np.isnan(row_values)
            means = np.nanmean(row_values, axis=0)
            errors = np.nanstd(row_values, axis=0) / np.sqrt(valid_mask.sum(axis=0))
            valid_columns = np.where(np.nanmean(valid_mask, axis=0) > 0)[0]
            plt.errorbar(valid_columns + 1 + offsets[idx], means[valid_columns], yerr=errors[valid_columns], fmt='-o',
                         capsize=5,
                         label=legend_labels[idx])  # Use pre-defined legend labels
            if len(valid_columns) > max_columns:
                max_columns = len(valid_columns)  # Update the maximum number of columns

        # Set x-axis ticks from 1 to the maximum number of columns
        x_ticks = np.arange(1, max_columns + 1)
        x_labels = ['11.33%', '16.33%', '21.33%', '26.33%', '31.33%', '36.33%']
        plt.xticks(ticks=x_ticks, labels=x_labels[:max_columns])
        plt.xlabel('$\omega_j$', fontsize=15)
        plt.ylabel('$IOR$', fontsize=15)
        plt.ylim(0.8, 1.0)  # Set y-axis range
        plt.xlim(0.5, max_columns + 0.5)  # Set x-axis range, showing ticks from 1 to the maximum number of columns
        # plt.grid(True, color='#D3D3D3', alpha=0.5)

        # Set y-axis ticks at intervals of 0.1
        y_ticks = np.arange(0.8, 1.05, 0.1)
        plt.yticks(ticks=y_ticks)

        # Format y-axis ticks as percentages
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        # Bold the tick labels
        for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            label.set_fontweight('bold')

        plt.legend(loc='lower right', fontsize=15, frameon=False)
        plt.savefig('consistency_error_bar.pdf', format='pdf', dpi=300)
        plt.show()
