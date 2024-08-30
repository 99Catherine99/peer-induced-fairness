import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class BootstrapSampler:
    def __init__(self, nonprotect_df, protect_df, matched_df, sampling_times=100, draws_per_sample=25, determine=45,
                 alpha=0.05):
        self.matched_df = matched_df
        self.sampling_times = sampling_times
        self.draws_per_sample = draws_per_sample
        self.determine = determine
        self.nonprotect_df = nonprotect_df
        self.protect_df = protect_df
        self.alpha = alpha

        # Set a fixed random seed for reproducibility
        np.random.seed(45)
        random.seed(45)

    def sample(self):
        samples_list = []
        for _, group in self.matched_df.groupby('treatment_index'):
            if len(group) >= self.determine:
                for _ in range(self.sampling_times):
                    sample = group.sample(n=self.draws_per_sample, replace=True)
                    samples_list.append(sample)
            else:
                samples_list.append(group)
        bootstrapped_samples = pd.concat(samples_list, ignore_index=True)
        bootstrapped_samples.to_csv('bootstrapped_samples.csv', index=False)
        return bootstrapped_samples

    def calculate_means(self, bootstrapped_samples):
        mean_values = {}
        grouped_samples = bootstrapped_samples.groupby('treatment_index')
        for treatment_index, samples in grouped_samples:
            original_count = self.matched_df[self.matched_df['treatment_index'] == treatment_index].shape[0]
            if len(samples) == original_count:
                control_indices = self.matched_df[self.matched_df['treatment_index'] == treatment_index][
                    'control_index']
                pr_values = self.nonprotect_df.loc[control_indices, 'Pr(Y=1)']
                mean_values[treatment_index] = [pr_values.mean()]
            else:
                means = []
                num_samples = len(samples) // self.draws_per_sample
                for i in range(num_samples):
                    sample_indices = samples.iloc[i * self.draws_per_sample:(i + 1) * self.draws_per_sample][
                        'control_index']
                    pr_values = self.nonprotect_df.loc[sample_indices, 'Pr(Y=1)']
                    means.append(pr_values.mean())
                mean_values[treatment_index] = means
        return mean_values



    def perform_test(self, mean_values, direction='two-sided'):
        results = []
        for treatment_index, means in mean_values.items():
            protected_value = self.protect_df.loc[treatment_index, 'Pr(Y=1)']
            if len(means) == 1:
                is_significant = "Unknown"
                t_stat = np.nan
                p_value = np.nan
            else:
                alternative = direction
                t_stat, p_value = ttest_1samp(means, protected_value, alternative=alternative)
                is_significant = "True" if p_value < self.alpha else "False"
            results.append({
                'treatment_index': treatment_index,
                't_stat': t_stat,
                'p_value': p_value,
                'mean_protected': protected_value,
                'mean_matched': np.mean(means) if means else np.nan,
                'is_significant': is_significant
            })
            results_df = pd.DataFrame(results)
            # Save the results to CSV
            results_df.to_csv(f'{direction}_test_results.csv', index=False)
        return results_df

    def plot_treatment_comparison(self, mean_values, results):
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        test_results = results.copy()
        test_results.set_index('treatment_index', inplace=True)

        data_to_plot = []
        for treatment_index, values in mean_values.items():
            # obtain Pr(Y=1) from protected group
            protected_value = self.protect_df.loc[treatment_index, 'Pr(Y=1)']
            # calculate the mean of non-protected group
            matched_mean = np.mean(values)
            # get the significance results
            is_significant = test_results.loc[treatment_index, 'is_significant']


            data_to_plot.append({
                'Treatment Index': treatment_index,
                'Protected Pr(Y=1)': protected_value,
                'Matched Mean Pr(Y=1)': matched_mean,
                'is_significant': is_significant
            })

        plot_df = pd.DataFrame(data_to_plot)

        # add colour according to if the data points are at the right or left side of Y=X
        plot_df['comparison'] = plot_df.apply(
            lambda row: 'Higher than' if row['Protected Pr(Y=1)'] > row['Matched Mean Pr(Y=1)'] and row['is_significant'] == 'True' else
            'Lower than' if row['is_significant'] == 'True' else
            'Equal' if row['is_significant'] == 'False' else
            'Unknown', axis=1)

        # filter out the data points labelled Unknown
        plot_df = plot_df[plot_df['comparison'] != 'Unknown']

        # plot the non-blue data point first
        non_blue = plot_df[plot_df['comparison'] != 'Equal']
        blue = plot_df[plot_df['comparison'] == 'Equal']


        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x='Protected Pr(Y=1)',
            y='Matched Mean Pr(Y=1)',
            hue='comparison',  # Using significance results as color classification
            palette={'Lower than': '#B02425', 'Higher than': '#FF6100'},
            data=non_blue,
            legend=None,
            s=50  # data points size
        )

        # then plot the blue data points
        sns.scatterplot(
            x='Protected Pr(Y=1)',
            y='Matched Mean Pr(Y=1)',
            color='blue',
            data=blue,
            legend=None,
            s=70  # data points size
        )

        plt.xlabel('$Pr(\hat Y=1|X=\\mathbf{x}, S=s-)$', fontsize=15)
        plt.ylabel('$E[\\bar{T}]$', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)

        # bold the x,y tick
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        plt.ylim(bottom=0.7)
        # plt.xlim(left=0, right=1)

        # plot dashed black Y=X
        min_value = min(plot_df['Protected Pr(Y=1)'].min(), plot_df['Matched Mean Pr(Y=1)'].min())
        max_value = max(plot_df['Protected Pr(Y=1)'].max(), plot_df['Matched Mean Pr(Y=1)'].max())
        plt.plot([min_value, max_value], [min_value, max_value], 'k--')


        custom_legend_labels = ['Higher than', 'Lower than', 'Equal']
        custom_legend_colors = ['#FF6100', '#B02425', 'blue']
        custom_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in custom_legend_colors]
        plt.legend(custom_legend_handles, custom_legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=13)

        plt.tight_layout()
        plt.savefig('single_double_matched_scatter_plot.pdf', format='pdf', dpi=300)
        plt.show()

        # Save the DataFrame to CSV
        plot_df.to_csv('single_double_treatment_comparison_results.csv', index=False)

        return plot_df

    def try_exp_plot(self, mean_values, selected_treatment_index, results):

        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        test_results = results.copy()
        test_results.set_index('treatment_index', inplace=True)
        data_to_plot = []

        for treatment_index, values in mean_values.items():
            exp_protected_value = np.exp(self.protect_df.loc[treatment_index, 'Pr(Y=1)'])
            exp_matched_mean = np.exp(np.mean(values))
            is_significant = test_results.loc[treatment_index, 'is_significant']

            data_to_plot.append({
                'Treatment Index': treatment_index,
                'Exp(Protected Pr(Y=1))': exp_protected_value,
                'Exp(Matched Mean Pr(Y=1))': exp_matched_mean,
                'is_significant': is_significant
            })

        exp_plot_df = pd.DataFrame(data_to_plot)
        color_dict = {'True': '#B02425', 'False': 'blue', 'Unknown': 'gray'}

        exp_plot_df['is_significant'] = pd.Categorical(exp_plot_df['is_significant'],
                                                       categories=["Unknown", "True", "False"], ordered=True)
        exp_plot_df.sort_values('is_significant', inplace=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        main_plot = sns.scatterplot(
            x='Exp(Protected Pr(Y=1))',
            y='Exp(Matched Mean Pr(Y=1))',
            hue='is_significant',
            palette=color_dict,
            data=exp_plot_df,
            s=50
        )

        highlight_row = exp_plot_df[exp_plot_df['Treatment Index'] == selected_treatment_index]
        plt.scatter(
            highlight_row['Exp(Protected Pr(Y=1))'],
            highlight_row['Exp(Matched Mean Pr(Y=1))'],
            color='blue',
            marker='^',
            edgecolor='black',
            label='Selected $A$',
            s=50
        )


        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)

        plt.xlabel('$exp$(A value)', fontsize=15)
        plt.ylabel('$exp$(peers average)', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)

        plt.ylim(bottom=2)
        plt.xlim(left=1.2, right=2.7)


        # plot dashed black Y=X
        exp_min = min(exp_plot_df['Exp(Protected Pr(Y=1))'].min(), exp_plot_df['Exp(Matched Mean Pr(Y=1))'].min())
        exp_max = max(exp_plot_df['Exp(Protected Pr(Y=1))'].max(), exp_plot_df['Exp(Matched Mean Pr(Y=1))'].max())
        plt.plot([exp_min, exp_max], [exp_min, exp_max], 'k--')


        # Setup for inset
        target_row = highlight_row.iloc[0]
        x1 = target_row['Exp(Protected Pr(Y=1))'] - 0.06
        x2 = target_row['Exp(Protected Pr(Y=1))'] + 0.06
        y1 = target_row['Exp(Matched Mean Pr(Y=1))'] - 0.06
        y2 = target_row['Exp(Matched Mean Pr(Y=1))'] + 0.06

        axins = inset_axes(ax, width="60%", height="60%", loc='center left',
                           bbox_to_anchor=(0.01, 0.01, 0.4, 0.4),
                           bbox_transform=ax.transAxes)
        inset_plot = sns.scatterplot(
            x='Exp(Protected Pr(Y=1))',
            y='Exp(Matched Mean Pr(Y=1))',
            hue='is_significant',
            palette=color_dict,
            data=exp_plot_df,
            ax=axins,
            legend=False,
            s=30
        )
        axins.scatter(
            x=target_row['Exp(Protected Pr(Y=1))'],
            y=target_row['Exp(Matched Mean Pr(Y=1))'],
            color='blue',
            marker='^',  # Triangle marker for selected point
            s=50,  # Size of the triangle marker
            edgecolor='black'
        )
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.set_xlabel('')
        axins.set_ylabel('')
        axins.set_xticks([])
        axins.set_yticks([])

        # 绘制Y=X的黑色虚线
        exp_min = min(exp_plot_df['Exp(Protected Pr(Y=1))'].min(), exp_plot_df['Exp(Matched Mean Pr(Y=1))'].min())
        exp_max = max(exp_plot_df['Exp(Protected Pr(Y=1))'].max(), exp_plot_df['Exp(Matched Mean Pr(Y=1))'].max())
        plt.plot([exp_min, exp_max], [exp_min, exp_max], 'k--')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=5)
        plt.tight_layout()
        plt.savefig('exp_matched_scatter_plot.pdf', format='pdf', dpi=300)
        plt.show()

        return exp_plot_df


    # def try_plot(self, mean_values, selected_treatment_index, results):
    #     sns.set_context("talk", rc={"lines.linewidth": 2.5})
    #     test_results = results.copy()
    #     test_results.set_index('treatment_index', inplace=True)
    #     data_to_plot = []
    #
    #     for treatment_index, values in mean_values.items():
    #         protected_value = self.protect_df.loc[treatment_index, 'Pr(Y=1)']
    #         matched_mean = np.mean(values)
    #         is_significant = test_results.loc[treatment_index, 'is_significant']
    #
    #         data_to_plot.append({
    #             'Treatment Index': treatment_index,
    #             'Protected Pr(Y=1)': protected_value,
    #             'Matched Mean Pr(Y=1)': matched_mean,
    #             'is_significant': is_significant
    #         })
    #
    #     plot_df = pd.DataFrame(data_to_plot)
    #
    #
    #     plot_df['comparison'] = plot_df.apply(
    #         lambda row: 'Higher than' if row['Protected Pr(Y=1)'] > row['Matched Mean Pr(Y=1)'] and row['is_significant'] == 'True' else
    #         'Lower than' if row['is_significant'] == 'True' else
    #         'Equal' if row['is_significant'] == 'False' else
    #         'Unknown', axis=1)
    #
    #
    #     plot_df = plot_df[plot_df['comparison'] != 'Unknown']
    #
    #
    #     non_blue = plot_df[plot_df['comparison'] != 'Equal']
    #     blue = plot_df[plot_df['comparison'] == 'Equal']
    #
    #     fig, ax = plt.subplots(figsize=(7, 5))
    #     sns.scatterplot(
    #         x='Protected Pr(Y=1)',
    #         y='Matched Mean Pr(Y=1)',
    #         hue='comparison',
    #         palette={'Lower than': '#B02425', 'Higher than': '#FF6100'},
    #         data=non_blue,
    #         legend=None,
    #         s=50
    #     )
    #
    #
    #     sns.scatterplot(
    #         x='Protected Pr(Y=1)',
    #         y='Matched Mean Pr(Y=1)',
    #         color='blue',
    #         data=blue,
    #         legend=None,
    #         s=60
    #     )
    #
    #
    #     highlight_row = plot_df[plot_df['Treatment Index'] == selected_treatment_index]
    #     plt.scatter(
    #         highlight_row['Protected Pr(Y=1)'],
    #         highlight_row['Matched Mean Pr(Y=1)'],
    #         color='black',
    #         marker='^',
    #         edgecolor='black',
    #         label='Selected $A$',
    #         s=70
    #     )
    #
    #     plt.xlabel('A value', fontsize=15)
    #     plt.ylabel('Peers value', fontsize=15)
    #     ax.tick_params(axis='both', labelsize=15)
    #
    #     plt.ylim(bottom=0.7)
    #     # plt.xlim(left=0, right=1)
    #
    #
    #     min_value = min(plot_df['Protected Pr(Y=1)'].min(), plot_df['Matched Mean Pr(Y=1)'].min())
    #     max_value = max(plot_df['Protected Pr(Y=1)'].max(), plot_df['Matched Mean Pr(Y=1)'].max())
    #     plt.plot([min_value, max_value], [min_value, max_value], 'k--')
    #
    #
    #     custom_legend_labels = ['Higher than', 'Lower than', 'Equal', 'Selected $A$']
    #     custom_legend_colors = ['#FF6100', '#B02425', 'blue', 'blue']
    #     custom_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in custom_legend_colors]
    #     custom_legend_handles[-1] = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markeredgecolor='black', markersize=10)
    #     plt.legend(custom_legend_handles, custom_legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=10)
    #
    #
    #     plt.tight_layout()
    #     plt.savefig('matched_scatter_plot.pdf', format='pdf', dpi=300)
    #     plt.show()
    #
    #     return plot_df



    def plot_area_scatter(self, mean_values, selected_treatment_index, results, result_type, threshold=0.00000001,
                          comparison_type='less than'):
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        test_results = results.copy()
        test_results.set_index('treatment_index', inplace=True)
        data_to_plot = []

        for treatment_index, values in mean_values.items():
            protected_value = self.protect_df.loc[treatment_index, 'Pr(Y=1)']
            matched_mean = np.mean(values)
            p_value = test_results.loc[treatment_index, 'p_value']

            # Determine the color based on p_value, result type, threshold, and comparison type
            if result_type == 'single_less_sided_results':
                if comparison_type == 'less than':
                    color = '#FF6100' if p_value < threshold else 'gray'
                elif comparison_type == 'greater than':
                    color = '#FF6100' if p_value > threshold else 'gray'
                elif comparison_type == 'between':
                    color = '#FF6100' if threshold[0] < p_value < threshold[1] else 'gray'
            elif result_type == 'single_large_sided_results':
                if comparison_type == 'less than':
                    color = '#B02425' if p_value < threshold else 'gray'
                elif comparison_type == 'greater than':
                    color = '#B02425' if p_value > threshold else 'gray'
                elif comparison_type == 'between':
                    color = '#B02425' if threshold[0] < p_value < threshold[1] else 'gray'
            elif result_type == 'double_sided_results':
                color = 'blue' if p_value > 0.05 else 'gray'
            else:
                if comparison_type == 'less than':
                    color = 'green' if p_value < threshold else 'gray'
                elif comparison_type == 'greater than':
                    color = 'green' if p_value > threshold else 'gray'
                elif comparison_type == 'between':
                    color = 'green' if threshold[0] < p_value < threshold[1] else 'gray'

            data_to_plot.append({
                'Treatment Index': treatment_index,
                'Protected Pr(Y=1)': protected_value,
                'Matched Mean Pr(Y=1)': matched_mean,
                'p_value': p_value,
                'color': color
            })

        plot_df = pd.DataFrame(data_to_plot)

        # Plot gray points first
        gray_df = plot_df[plot_df['color'] == 'gray']
        color_df = plot_df[plot_df['color'] != 'gray']

        # Print the number of colored points
        print(f"Number of colored points: {len(color_df)}")

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            x='Protected Pr(Y=1)',
            y='Matched Mean Pr(Y=1)',
            color='gray',
            data=gray_df,
            legend=None,
            s=50
        )

        # Plot colored points
        sns.scatterplot(
            x='Protected Pr(Y=1)',
            y='Matched Mean Pr(Y=1)',
            hue='color',
            palette={'#FF6100': '#FF6100', '#B02425': '#B02425', 'green': 'green', 'blue': 'blue'},
            data=color_df,
            legend=None,
            s=50
        )

        # Highlight selected point
        highlight_row = plot_df[plot_df['Treatment Index'] == selected_treatment_index]
        plt.scatter(
            highlight_row['Protected Pr(Y=1)'],
            highlight_row['Matched Mean Pr(Y=1)'],
            color='black',
            marker='^',
            edgecolor='black',
            label='Selected $A$',
            s=80
        )

        plt.xlabel('Protected Pr(Y=1)', fontsize=15)
        plt.ylabel('Peers average Pr(Y=1)', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)

        plt.ylim(bottom=0.7)
        # plt.xlim(left=0.2, right=1)

        # 绘制Y=X的黑色虚线
        min_value = min(plot_df['Protected Pr(Y=1)'].min(), plot_df['Matched Mean Pr(Y=1)'].min())
        max_value = max(plot_df['Protected Pr(Y=1)'].max(), plot_df['Matched Mean Pr(Y=1)'].max())
        plt.plot([min_value, max_value], [min_value, max_value], 'k--')

        # 自定义图例
        if result_type == 'single_less_sided_results':
            custom_legend_labels = [f'p_value {comparison_type} {threshold}', 'Others', 'Selected $A$']
            custom_legend_colors = ['#FF6100', 'gray', 'black']
        elif result_type == 'single_large_sided_results':
            custom_legend_labels = [f'p_value {comparison_type} {threshold}', 'Others', 'Selected $A$']
            custom_legend_colors = ['#B02425', 'gray', 'black']
        elif result_type == 'double_sided_results':
            custom_legend_labels = ['p_value > 0.05', 'Others', 'Selected $A$']
            custom_legend_colors = ['blue', 'gray', 'black']
        else:
            custom_legend_labels = [f'p_value {comparison_type} {threshold}', 'Others', 'Selected $A$']
            custom_legend_colors = ['green', 'gray', 'black']

        custom_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for
                                 color in custom_legend_colors]
        custom_legend_handles[-1] = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black',
                                               markeredgecolor='black', markersize=10)
        plt.legend(custom_legend_handles, custom_legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,
                   frameon=False, fontsize=10)

        plt.tight_layout()
        plt.savefig('matched_scatter_plot.pdf', format='pdf', dpi=300)
        plt.show()

        return plot_df

    def plot_density(self, protect_df, mean_values, treatment_index, bw_adjust=2, show_line=True, line_color='blue', density_color='blue'):
        """
        Plot the density of Pr(Y=1) means for a specified treatment index
        and print the kurtosis and standard deviation.
        """
        blue_line_value = protect_df.iloc[treatment_index]['Pr(Y=1)']
        if treatment_index in mean_values:
            data = mean_values[treatment_index]
            data_kurtosis = kurtosis(data)
            data_std = np.std(data)

            # Print the calculated indicators
            print(f"Kurtosis: {data_kurtosis}")
            print(f"Standard Deviation: {data_std}")

            sns.set_context("talk", rc={"lines.linewidth": 2.5})
            plt.figure(figsize=(5, 4))
            sns.kdeplot(data, shade=False, bw_adjust=bw_adjust, color=density_color, linewidth=2.5, label='Peers')
            if show_line:
                plt.axvline(x=blue_line_value, color=line_color, linestyle='--', linewidth=2, label='A')

            plt.xlabel('Mean $Pr(\hat Y=1)$', fontsize=15)
            plt.ylabel('Density', fontsize=15)
            plt.xlim(0.5, 1)
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=11)
            plt.tight_layout()
            plt.grid(False)
            plt.savefig(f'{treatment_index}_after_bootstrapping.pdf', format='pdf', dpi=300)
            plt.show()
        else:
            print("Treatment index not found in the provided mean values.")



    def plot_significance_single_double_pie_chart(self, single_results, double_results):
        sns.set_context("talk", rc={"lines.linewidth": 2.5})


        single_results_copy = single_results.copy()
        double_results_copy = double_results.copy()


        single_results_copy['is_significant'] = single_results_copy['is_significant'].replace(
            {'True': 'Single True', 'False': 'Single False'})
        double_results_copy['is_significant'] = double_results_copy['is_significant'].replace(
            {'True': 'Double True', 'False': 'Double False'})

        single_results_copy.rename(columns={'is_significant': 'single_is_significant'}, inplace=True)
        double_results_copy.rename(columns={'is_significant': 'double_is_significant'}, inplace=True)


        combined_results = pd.merge(single_results_copy, double_results_copy, on='treatment_index')


        combined_results['Final Category'] = combined_results.apply(
            lambda row: 'Unknown' if row['single_is_significant'] == 'Unknown' or row['double_is_significant'] == 'Unknown' else
            'Equal' if row['double_is_significant'] == 'Double False' else
            'Lower than' if row['single_is_significant'] == 'Single True' and row['double_is_significant'] == 'Double True' else
            'Higher than' if row['double_is_significant'] == 'Double True' and row['single_is_significant'] == 'Single False' else
            'Matched',  # Remaining cases
            axis=1
        )


        significance_counts = combined_results['Final Category'].value_counts()
        total_counts = significance_counts.sum()
        significance_proportions = (significance_counts / total_counts * 100).round(2)


        color_mapping = {
            'Lower than': '#B02425',
            'Higher than': '#FF6100',
            'Equal': 'blue',
            'Unknown': 'gray',
            'Matched': 'green'
        }
        plot_colors = [color_mapping[x] for x in significance_counts.index]  # 直接应用颜色映射


        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(significance_counts, labels=significance_counts.index, autopct='%1.1f%%',
                                          startangle=90, colors=plot_colors)
        ax.axis('equal')

        plt.savefig('single_double_bootstrapping_pie_chart.pdf', format='pdf', dpi=300)
        plt.show()


        summary_df = pd.DataFrame({
            'Category': significance_counts.index,
            'Count': significance_counts.values,
            'Proportion (%)': significance_proportions
        })

        summary_df.to_csv('summary_results.csv', index=False)

        return summary_df


    def plot_significance_pie_chart(self, results):
        results['is_significant'] = results['is_significant'].astype(str)
        significance_counts = results['is_significant'].value_counts()
        total_counts = significance_counts.sum()
        significance_proportions = (significance_counts / total_counts * 100).round(2)
        color_mapping = {'Double True': '#B02425', 'Double False': 'blue', 'Unknown': 'gray'}
        plot_colors = [color_mapping.get(x, 'gray') for x in significance_counts.index]
        fig, ax = plt.subplots()
        ax.pie(significance_counts, labels=significance_counts.index, autopct='%1.2f%%', startangle=90, colors=plot_colors)
        ax.axis('equal')
        plt.savefig('bootstrapping_pie_chart.pdf', format='pdf', dpi=300)
        plt.show()
        summary_df = pd.DataFrame({
            'Category': significance_counts.index,
            'Count': significance_counts.values,
            'Proportion (%)': significance_proportions
        })
        return summary_df



    def plot_treatment_single_double_comparison(self, mean_values, results):
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        test_results = results.copy()
        test_results.set_index('treatment_index', inplace=True)

        data_to_plot = []

        for treatment_index, values in mean_values.items():
            exp_protected_value = np.exp(self.protect_df.loc[treatment_index, 'Pr(Y=1)'])
            exp_matched_mean = np.exp(np.mean(values))
            is_significant = test_results.loc[treatment_index, 'is_significant']


            data_to_plot.append({
                'Treatment Index': treatment_index,
                'Exp(Protected Pr(Y=1))': exp_protected_value,
                'Exp(Matched Mean Pr(Y=1))': exp_matched_mean,
                'is_significant': is_significant
            })

        exp_plot_df = pd.DataFrame(data_to_plot)


        exp_plot_df['comparison'] = exp_plot_df.apply(
            lambda row: 'Higher than' if row['Exp(Protected Pr(Y=1))'] > row['Exp(Matched Mean Pr(Y=1))'] and row['is_significant'] == 'True' else
            'Lower than' if row['is_significant'] == 'True' else
            'Equal' if row['is_significant'] == 'False' else
            'Unknown', axis=1)


        exp_plot_df = exp_plot_df[exp_plot_df['comparison'] != 'Unknown']


        non_blue = exp_plot_df[exp_plot_df['comparison'] != 'Equal']
        blue = exp_plot_df[exp_plot_df['comparison'] == 'Equal']


        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x='Exp(Protected Pr(Y=1))',
            y='Exp(Matched Mean Pr(Y=1))',
            hue='comparison',
            palette={'Lower than': '#B02425', 'Higher than': '#FF6100'},
            data=non_blue,
            legend=None,
            s=50
        )


        sns.scatterplot(
            x='Exp(Protected Pr(Y=1))',
            y='Exp(Matched Mean Pr(Y=1))',
            color='blue',
            data=blue,
            legend=None,
            s=60
        )

        plt.xlabel('$exp$(A value)', fontsize=15)
        plt.ylabel('$exp$(Overall mean of all peers)', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)

        plt.ylim(bottom=1.8)
        plt.xlim(left=1.2, right=2.7)


        exp_min = min(exp_plot_df['Exp(Protected Pr(Y=1))'].min(), exp_plot_df['Exp(Matched Mean Pr(Y=1))'].min())
        exp_max = max(exp_plot_df['Exp(Protected Pr(Y=1))'].max(), exp_plot_df['Exp(Matched Mean Pr(Y=1))'].max())
        plt.plot([exp_min, exp_max], [exp_min, exp_max], 'k--')


        custom_legend_labels = ['Higher than', 'Lower than', 'Equal']
        custom_legend_colors = ['#FF6100', '#B02425', 'blue']
        custom_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in custom_legend_colors]
        plt.legend(custom_legend_handles, custom_legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=13)

        plt.tight_layout()
        plt.savefig('single_double_exp_matched_scatter_plot.pdf', format='pdf', dpi=300)
        plt.show()

        # Save the DataFrame to CSV
        exp_plot_df.to_csv('single_double_treatment_comparison_results.csv', index=False)

        return exp_plot_df

