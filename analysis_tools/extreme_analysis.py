import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import kurtosis, t

class DensityScatterPlotter:
    def __init__(self, protect_df, nonprotect_df, bootstrapped_samples, result_type, comparison_type='less than'):
        self.protect_df = protect_df
        self.nonprotect_df = nonprotect_df
        self.bootstrapped_samples = bootstrapped_samples
        self.result_type = result_type
        self.comparison_type = comparison_type

    def calculate_p_value(self, t_statistic, freedom=99):
        """
        Calculate p-value for a given t-statistic and degrees of freedom.
        """
        p_value = t.sf(np.abs(t_statistic), freedom)  # sf is the survival function (1 - cdf)
        return p_value

    def plot_area_scatter(self, mean_values, selected_treatment_index, results, threshold=0.00000001):
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        mean_values = {k: v for k, v in mean_values.items() if len(v) > 1}

        # test_results = results.copy()
        test_results = results[results['is_significant'] != 'Unknown']
        test_results.set_index('treatment_index', inplace=True)
        data_to_plot = []

        for treatment_index, values in mean_values.items():
            protected_value = self.protect_df.loc[treatment_index, 'Pr(Y=1)']
            matched_mean = np.mean(values)
            p_value = test_results.loc[treatment_index, 'p_value']

            # Determine the color based on p_value, result type, threshold, and comparison type
            if self.result_type == 'single_less_sided_results':
                if self.comparison_type == 'less than':
                    color = '#FF6100' if p_value <= threshold else 'gray'
                elif self.comparison_type == 'greater than':
                    color = '#FF6100' if p_value > threshold else 'gray'
                elif self.comparison_type == 'between':
                    color = '#FF6100' if threshold[0] < p_value < threshold[1] else 'gray'
            elif self.result_type == 'single_greater_sided_results':
                if self.comparison_type == 'less than':
                    color = '#B02425' if p_value <= threshold else 'gray'
                elif self.comparison_type == 'greater than':
                    color = '#B02425' if p_value > threshold else 'gray'
                elif self.comparison_type == 'between':
                    color = '#B02425' if threshold[0] < p_value < threshold[1] else 'gray'
            elif self.result_type == 'double_sided_results':
                color = 'blue' if p_value >= 0.05 else 'gray'
            else:
                if self.comparison_type == 'less than':
                    color = '#84BA84' if p_value < threshold else 'gray'
                elif self.comparison_type == 'greater than':
                    color = '#84BA84' if p_value > threshold else 'gray'
                elif self.comparison_type == 'between':
                    color = '#84BA84' if threshold[0] < p_value < threshold[1] else 'gray'

            data_to_plot.append({
                'treatment_index': treatment_index,
                'protected Pr(Y=1)': protected_value,
                'matched mean Pr(Y=1)': matched_mean,
                'p_value': p_value,
                'color': color,
                'is_selected': 'Yes' if treatment_index == selected_treatment_index else 'No'
            })

        plot_df = pd.DataFrame(data_to_plot)

        # Plot gray points first
        gray_df = plot_df[plot_df['color'] == 'gray']
        color_df = plot_df[plot_df['color'] != 'gray']

        # Print the number of colored points
        print(f"Number of colored points: {len(color_df)}")

        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(
            x='protected Pr(Y=1)',
            y='matched mean Pr(Y=1)',
            color='gray',
            data=gray_df,
            legend=None,
            s=50,
            alpha=0.3  # Set the transparency of the data points
        )

        # Determine alpha for colored points based on comparison type
        # alpha_value = 0.6 if self.comparison_type == 'between' else 1.0
        alpha_value=1

        # Plot colored points
        sns.scatterplot(
            x='protected Pr(Y=1)',
            y='matched mean Pr(Y=1)',
            hue='color',
            palette={'#FF6100': '#FF6100', '#B02425': '#B02425', '#84BA84': '#84BA84', 'blue': 'blue'},
            data=color_df,
            legend=None,
            s=70,
            alpha=alpha_value
        )

        # Highlight selected point
        highlight_row = plot_df[plot_df['treatment_index'] == selected_treatment_index]
        highlight_color = highlight_row['color'].values[0]
        plt.scatter(
            highlight_row['protected Pr(Y=1)'],
            highlight_row['matched mean Pr(Y=1)'],
            color=highlight_color,
            marker='^',
            edgecolor='black',
            label='Selected micro-firm',
            s=90
        )

        plt.xlabel('$Pr(\hat Y=1|X=\\mathbf{x}, S=s-)$', fontsize=15)
        plt.ylabel('$E[\\bar{T}]$', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        plt.ylim(bottom=0.7)
        # plt.xlim(left=0.5, right=1)

        min_value = min(plot_df['protected Pr(Y=1)'].min(), plot_df['matched mean Pr(Y=1)'].min())
        max_value = max(plot_df['protected Pr(Y=1)'].max(), plot_df['matched mean Pr(Y=1)'].max())
        plt.plot([min_value, max_value], [min_value, max_value], 'k--')

        if self.result_type == 'single_less_sided_results':
            custom_legend_labels = [f'p_value {self.comparison_type} {threshold}', 'Others', 'Selected micro-firm']
            custom_legend_colors = ['#FF6100', 'gray', highlight_color]
        elif self.result_type == 'single_greater_sided_results':
            custom_legend_labels = [f'p_value {self.comparison_type} {threshold}', 'Others', 'Selected micro-firm']
            custom_legend_colors = ['#B02425', 'gray', highlight_color]
        elif self.result_type == 'double_sided_results':
            custom_legend_labels = ['p_value > 0.05', 'Others', 'Selected micro-firm']
            custom_legend_colors = ['blue', 'gray', highlight_color]
        else:
            custom_legend_labels = [f'p_value {self.comparison_type} {threshold}', 'Others', 'Selected micro-firm']
            custom_legend_colors = ['#84BA84', 'gray', highlight_color]

        custom_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in custom_legend_colors]
        custom_legend_handles[-1] = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=highlight_color, markeredgecolor='black', markersize=10)
        plt.legend(custom_legend_handles, custom_legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False, fontsize=12)

        plt.tight_layout()
        plt.savefig(f'{self.result_type}_{self.comparison_type}_scatter.pdf', format='pdf', dpi=300)
        plt.show()

        data_ratio = color_df.shape[0]/test_results.shape[0]
        print(f'the data ratio is:{data_ratio}')

        # Save the dataframes to CSV
        plot_df.to_csv(f'{self.result_type}_{self.comparison_type}_plot_df.csv', index=False)
        gray_df.to_csv(f'{self.result_type}_{self.comparison_type}_gray_df.csv', index=False)
        color_df.to_csv(f'{self.result_type}_{self.comparison_type}_color_df.csv', index=False)

        return plot_df, gray_df, color_df

    def plot_group_density(self, dataframe, column1, column2, color, density_color='#84BA84'):
        """
        绘制有颜色数据点的两个指定数据列的密度图，并在同一图中显示，并打印均值和标准差。
        """
        # # 设置全局参数，使边框和刻度线变粗
        # plt.rcParams['axes.linewidth'] = 10  # 坐标轴边框宽度
        # plt.rcParams['xtick.major.width'] = 4  # x轴刻度线宽度
        # plt.rcParams['ytick.major.width'] = 4  # y轴刻度线宽度
        # plt.rcParams['xtick.major.size'] = 3  # x轴刻度线长度
        # plt.rcParams['ytick.major.size'] = 3  # y轴刻度线长度

        # 设置Seaborn情境，确保所有线条更粗
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        plt.figure(figsize=(6, 5))

        data1 = dataframe[column1]
        data2 = dataframe[column2]
        mean1 = data1.mean()
        mean2 = data2.mean()
        std1 = data1.std()
        std2 = data2.std()

        # 打印均值和标准差
        print(f"{color} Protected Pr(Y=1) Mean: {mean1}, Standard Deviation: {std1}")
        print(f"{color} Matched Mean Pr(Y=1) Mean: {mean2}, Standard Deviation: {std2}")

        # 绘制密度图
        ax = sns.kdeplot(data1, shade=False, color=color, linewidth=7, label='Micro-firms', linestyle='--')
        sns.kdeplot(data2, shade=False, color=density_color, linewidth=7, label='Peers group')

        # 设置标签和标题
        plt.xlabel('$Pr(\hat Y=1|X=\\mathbf{x})$', fontsize=15)
        plt.ylabel('Density', fontsize=15)
        plt.xlim(0.6, 1)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=15)
        plt.tight_layout()
        plt.grid(False)

        plt.xticks(np.linspace(0.6, 1, 5))  # 例如，设置5个刻度点

        # 加粗刻度标签
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        # 保存和显示图形
        plt.savefig(f'{self.result_type}_{self.comparison_type}_group_density.pdf', format='pdf', dpi=300)
        plt.show()
        # Save the data to CSV
        density_df = pd.DataFrame({
            f'{column1}_mean': [mean1],
            f'{column2}_mean': [mean2]
        })
        density_df.to_csv(f'{self.result_type}_{self.comparison_type}_density_stats.csv', index=False)

    def plot_scatter(self, mean_values, selected_treatment_index, results, threshold_type='fixed', floating_ratio=0.1, mean_diff_threshold=0.03, p_value_threshold=0.05):
        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        mean_values = {k: v for k, v in mean_values.items() if len(v) > 1}

        # 过滤掉'is_significant'为'Unknown'的结果
        test_results = results[results['is_significant'] != 'Unknown']
        test_results.set_index('treatment_index', inplace=True)
        data_to_plot = []

        for treatment_index, values in mean_values.items():
            protected_value = self.protect_df.loc[treatment_index, 'Pr(Y=1)']
            matched_mean = np.mean(values)
            mean_diff = matched_mean - protected_value
            reverse_mean_diff = protected_value - matched_mean
            p_value = test_results.loc[treatment_index, 'p_value']
            is_significant = test_results.loc[treatment_index, 'is_significant']

            # 确定阈值
            if threshold_type == 'floating':
                actual_mean_diff_threshold = floating_ratio * protected_value
            else:
                actual_mean_diff_threshold = mean_diff_threshold

            color = 'gray'
            # Determine the color based on mean_diff and p_value thresholds
            if is_significant == 'True':
                if self.result_type == 'single_less_sided_results':
                    if self.comparison_type == 'less than':
                        color = '#FF6100' if reverse_mean_diff > actual_mean_diff_threshold else 'gray'
                    elif self.comparison_type == 'between':
                        color = '#FF6100' if reverse_mean_diff <= actual_mean_diff_threshold else 'gray'

                elif self.result_type == 'single_greater_sided_results':
                    if self.comparison_type == 'less than':
                        color = '#B02425' if mean_diff > actual_mean_diff_threshold else 'gray'
                    elif self.comparison_type == 'between':
                        color = '#B02425' if mean_diff <= actual_mean_diff_threshold else 'gray'

            elif self.result_type == 'double_sided_results':
                color = 'blue' if p_value > p_value_threshold else 'gray'

            data_to_plot.append({
                'treatment_index': treatment_index,
                'protected Pr(Y=1)': protected_value,
                'matched mean Pr(Y=1)': matched_mean,
                'mean_diff': mean_diff,
                'p_value': p_value,
                'color': color,
                'is_selected': 'Yes' if treatment_index == selected_treatment_index else 'No'
            })

        plot_df = pd.DataFrame(data_to_plot)

        # Plot gray points first
        gray_df = plot_df[plot_df['color'] == 'gray']
        color_df = plot_df[plot_df['color'] != 'gray']

        # Print the number of colored points
        print(f"Number of colored points: {len(color_df)}")

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            x='protected Pr(Y=1)',
            y='matched mean Pr(Y=1)',
            color='gray',
            data=gray_df,
            legend=None,
            s=50,
            alpha=0.3  # 设置灰色点的透明度
        )

        alpha_value = 1

        # Plot colored points
        sns.scatterplot(
            x='protected Pr(Y=1)',
            y='matched mean Pr(Y=1)',
            hue='color',
            palette={'#FF6100': '#FF6100', '#B02425': '#B02425', '#84BA84': '#84BA84', 'blue': 'blue'},
            data=color_df,
            legend=None,
            s=70,
            alpha=alpha_value
        )

        # Highlight selected point
        highlight_row = plot_df[plot_df['treatment_index'] == selected_treatment_index]
        highlight_color = highlight_row['color'].values[0]
        plt.scatter(
            highlight_row['protected Pr(Y=1)'],
            highlight_row['matched mean Pr(Y=1)'],
            color=highlight_color,
            marker='^',
            edgecolor='black',
            label='Selected micro-firm',
            s=90
        )

        plt.xlabel('$Pr(\hat Y=1|X=\\mathbf{x}, S=s-)$', fontsize=15)
        plt.ylabel('$E[\\bar{T}]$', fontsize=15)
        ax.tick_params(axis='both', labelsize=15)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        plt.ylim(bottom=0.7)

        # 绘制Y=X的黑色虚线
        min_value = min(plot_df['protected Pr(Y=1)'].min(), plot_df['matched mean Pr(Y=1)'].min())
        max_value = max(plot_df['protected Pr(Y=1)'].max(), plot_df['matched mean Pr(Y=1)'].max())
        plt.plot([min_value, max_value], [min_value, max_value], 'k--')

        # # 自定义图例
        # if self.result_type == 'single_less_sided_results':
        #     custom_legend_labels = [f'mean_diff {self.comparison_type} {mean_diff_threshold}', 'Others', 'Selected micro-firm']
        #     custom_legend_colors = ['#FF6100', 'gray', highlight_color]
        # elif self.result_type == 'single_greater_sided_results':
        #     custom_legend_labels = [f'mean_diff {self.comparison_type} {mean_diff_threshold}', 'Others', 'Selected micro-firm']
        #     custom_legend_colors = ['#B02425', 'gray', highlight_color]
        # elif self.result_type == 'double_sided_results':
        #     custom_legend_labels = ['p_value > 0.05', 'Others', 'Selected micro-firm']
        #     custom_legend_colors = ['blue', 'gray', highlight_color]
        # else:
        #     custom_legend_labels = [f'mean_diff {self.comparison_type} {mean_diff_threshold}', 'Others', 'Selected micro-firm']
        #     custom_legend_colors = ['#84BA84', 'gray', highlight_color]

        # 自定义图例
        if self.result_type == 'single_less_sided_results':
            if self.comparison_type == 'less than':
                custom_legend_labels = ['EP', 'Others', 'Selected micro-firm']
                custom_legend_colors = ['#FF6100', 'gray', highlight_color]
            elif self.comparison_type == 'between':
                custom_legend_labels = ['SP', 'Others', 'Selected micro-firm']
                custom_legend_colors = ['#FF6100', 'gray', highlight_color]
        elif self.result_type == 'single_greater_sided_results':
            if self.comparison_type == 'less than':
                custom_legend_labels = ['ED', 'Others', 'Selected micro-firm']
                custom_legend_colors = ['#B02425', 'gray', highlight_color]
            elif self.comparison_type == 'between':
                custom_legend_labels = ['SD', 'Others', 'Selected micro-firm']
                custom_legend_colors = ['#B02425', 'gray', highlight_color]
        elif self.result_type == 'double_sided_results':
            custom_legend_labels = ['FT', 'Others', 'Selected micro-firm']
            custom_legend_colors = ['blue', 'gray', highlight_color]
        else:
            custom_legend_labels = ['Others', 'Selected micro-firm']
            custom_legend_colors = ['gray', highlight_color]

        custom_legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in custom_legend_colors]
        custom_legend_handles[-1] = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor=highlight_color, markeredgecolor='black', markersize=10)
        plt.legend(custom_legend_handles, custom_legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.17), ncol=3, frameon=False, fontsize=12)

        plt.tight_layout()
        plt.savefig(f'{self.result_type}_{self.comparison_type}_scatter.pdf', format='pdf', dpi=300)
        plt.show()

        data_ratio = color_df.shape[0] / test_results.shape[0]
        print(f'the data ratio is: {data_ratio}')

        # Save the dataframes to CSV
        plot_df.to_csv(f'{self.result_type}_{self.comparison_type}_plot_df.csv', index=False)
        gray_df.to_csv(f'{self.result_type}_{self.comparison_type}_gray_df.csv', index=False)
        color_df.to_csv(f'{self.result_type}_{self.comparison_type}_color_df.csv', index=False)

        return plot_df, gray_df, color_df

    def group_ground_truth(self, color_df,label='sigle_greater_between'):
        """
        根据 color_df['treatment_index'] 到 bootstrapped_samples['treatment_index'] 寻找 'control_index'，
        然后在 nonprotect_df 的 index 中寻找相应的 Binary Y，
        计算 Binary Y 为 0 和 1 的个数以及比例。

        参数：
        - color_df (pd.DataFrame): 包含有颜色数据点的 DataFrame。

        返回：
        - pd.DataFrame: 包含所有 treatment_index 的 Binary Y 为 0 和 1 的个数以及比例的 DataFrame。
        """
        all_control_indices = []
        protect_binary_y_values = []

        treatment_indices = color_df['treatment_index']

        for treatment_index in treatment_indices:
            control_indices = self.bootstrapped_samples.loc[
                self.bootstrapped_samples['treatment_index'] == treatment_index, 'control_index']
            all_control_indices.extend(control_indices.explode().tolist())
            protect_binary_y_values.append(self.protect_df.loc[treatment_index, 'Binary Y'])

        binary_y_values = self.nonprotect_df.loc[all_control_indices, 'Binary Y']
        count_0 = (binary_y_values == 0).sum()
        count_1 = (binary_y_values == 1).sum()
        total = len(binary_y_values)
        proportion_0 = count_0 / total
        proportion_1 = count_1 / total

        protect_count_0 = (np.array(protect_binary_y_values) == 0).sum()
        protect_count_1 = (np.array(protect_binary_y_values) == 1).sum()
        protect_total = len(protect_binary_y_values)
        protect_proportion_0 = protect_count_0 / protect_total
        protect_proportion_1 = protect_count_1 / protect_total

        data = {
            'Type': ['Nonprotect', 'Nonprotect', 'Protect', 'Protect'],
            'Binary Y': [0, 1, 0, 1],
            'Count': [count_0, count_1, protect_count_0, protect_count_1],
            'Proportion': [proportion_0, proportion_1, protect_proportion_0, protect_proportion_1]
        }

        df = pd.DataFrame(data)

        sns.set_context("talk", rc={"lines.linewidth": 2.5})
        # 绘制堆积柱形图
        fig, ax = plt.subplots(figsize=(5,4))

        # Nonprotect组的比例
        nonprotect_0 = df[(df['Type'] == 'Nonprotect') & (df['Binary Y'] == 0)]['Proportion'].values[0]
        nonprotect_1 = df[(df['Type'] == 'Nonprotect') & (df['Binary Y'] == 1)]['Proportion'].values[0]

        # Protect组的比例
        protect_0 = df[(df['Type'] == 'Protect') & (df['Binary Y'] == 0)]['Proportion'].values[0]
        protect_1 = df[(df['Type'] == 'Protect') & (df['Binary Y'] == 1)]['Proportion'].values[0]

        bar_width = 0.2
        x = np.array([1.6, 2])  # 调整x位置，使柱子更近

        # 绘制堆积柱形图
        bars1 = ax.bar(x[0], nonprotect_0, bar_width, color='#B02425')
        bars2 = ax.bar(x[0], nonprotect_1, bar_width, bottom=nonprotect_0, color='blue')

        bars3 = ax.bar(x[1], protect_0, bar_width, color='#B02425')
        bars4 = ax.bar(x[1], protect_1, bar_width, bottom=protect_0, color='blue')

        ax.set_xlabel('Group', fontsize=15)
        ax.set_ylabel('Percentage', fontsize=15)
        # ax.set_title('Proportion of Binary Y for Protect and Nonprotect Groups')
        ax.set_xticks(x)
        ax.set_xticklabels(['Peers', 'Micro-firms'], fontsize=15)

        # 设置图例，仅对两个色块进行解释
        ax.legend(['Rejected', 'Accepted'], loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2,
                  frameon=False, fontsize=15)

        plt.tight_layout()
        plt.savefig(f'{label}_proportion_stacked_bar_plot.pdf', format='pdf', dpi=300)
        plt.show()
        # Save the dataframe to CSV
        df.to_csv(f'{label}_group_ground_truth.csv', index=False)

        return df

    def try_1(self, matched_df, color_df, pr_y_means, label='sigle_greater_between'):
        """
        根据 color_df['treatment_index'] 到 bootstrapped_samples['treatment_index'] 寻找 'control_index'，
        然后在 nonprotect_df 的 index 中寻找相应的 Binary Y，
        计算 Binary Y 为 0 和 1 的个数以及比例，并将 pr_y_means 字典中的值作为 E[T] 加入结果。

        参数：
        - color_df (pd.DataFrame): 包含有颜色数据点的 DataFrame。
        - pr_y_means (dict): 包含 treatment_index 对应的 E[T] 值的字典。

        返回：
        - pd.DataFrame: 包含所有 treatment_index 的 Binary Y 为 0 和 1 的个数、比例及 E[T] 的 DataFrame。
        """
        all_control_indices = []
        protect_binary_y_values = []

        treatment_indices = color_df['treatment_index']

        results = []

        for treatment_index in treatment_indices:
            control_indices = matched_df.loc[
                matched_df['treatment_index'] == treatment_index, 'control_index']
            all_control_indices.extend(control_indices.explode().tolist())

            binary_y_values = self.nonprotect_df.loc[control_indices.explode().tolist(), 'Binary Y']
            count_0 = (binary_y_values == 0).sum()
            count_1 = (binary_y_values == 1).sum()
            total = len(binary_y_values)
            proportion_0 = count_0 / total if total > 0 else 0
            proportion_1 = count_1 / total if total > 0 else 0

            # 计算 E[T] 值
            e_t_values = pr_y_means.get(treatment_index, [])
            e_t_mean = np.mean(e_t_values) if e_t_values else np.nan

            results.append({
                'treatment_index': treatment_index,
                'Binary Y 0 Count': count_0,
                'Binary Y 1 Count': count_1,
                'Proportion 0': proportion_0,
                'Proportion 1': proportion_1,
                'Y': self.protect_df.loc[treatment_index, 'Binary Y'],  # 新增一列标识Y
                'Pr(Y=1)': self.protect_df.loc[treatment_index, 'Pr(Y=1)'],  # 新增 Pr(Y=1)
                # 'Binary Prediction': self.protect_df.loc[treatment_index, 'Binary Prediction'],  # 新增 Binary Prediction
                'E[T]': e_t_mean  # 新增 E[T]
            })

            protect_binary_y_values.append(self.protect_df.loc[treatment_index, 'Binary Y'])

        protect_count_0 = (np.array(protect_binary_y_values) == 0).sum()
        protect_count_1 = (np.array(protect_binary_y_values) == 1).sum()
        protect_total = len(protect_binary_y_values)
        protect_proportion_0 = protect_count_0 / protect_total
        protect_proportion_1 = protect_count_1 / protect_total

        protect_data = {
            'Type': ['Protect', 'Protect'],
            'Binary Y': [0, 1],
            'Count': [protect_count_0, protect_count_1],
            'Proportion': [protect_proportion_0, protect_proportion_1]
        }

        results_df = pd.DataFrame(results)
        protect_df = pd.DataFrame(protect_data)

        # results_df.to_csv(f'{label}_group_ground_truth.csv', index=False)
        # protect_df.to_csv(f'{label}_protect_group_ground_truth.csv', index=False)

        return results_df, protect_df


    def try_2(self, color_df, pr_y_means, label='sigle_greater_between'):
        """
        根据 color_df['treatment_index'] 到 bootstrapped_samples['treatment_index'] 寻找 'control_index'，
        然后在 nonprotect_df 的 index 中寻找相应的 Binary Y，
        计算 Binary Y 为 0 和 1 的个数以及比例，并将 pr_y_means 字典中的值作为 E[T] 加入结果。

        参数：
        - color_df (pd.DataFrame): 包含有颜色数据点的 DataFrame。
        - pr_y_means (dict): 包含 treatment_index 对应的 E[T] 值的字典。

        返回：
        - pd.DataFrame: 包含所有 treatment_index 的 Binary Y 为 0 和 1 的个数、比例及 E[T] 的 DataFrame。
        """
        all_control_indices = []
        protect_binary_y_values = []

        treatment_indices = color_df['treatment_index']

        results = []

        for treatment_index in treatment_indices:
            control_indices = self.bootstrapped_samples.loc[
                self.bootstrapped_samples['treatment_index'] == treatment_index, 'control_index']
            all_control_indices.extend(control_indices.explode().tolist())

            binary_y_values = self.nonprotect_df.loc[control_indices.explode().tolist(), 'Binary Y']
            count_0 = (binary_y_values == 0).sum()
            count_1 = (binary_y_values == 1).sum()
            total = len(binary_y_values)
            proportion_0 = count_0 / total if total > 0 else 0
            proportion_1 = count_1 / total if total > 0 else 0

            # 计算 E[T] 值
            e_t_values = pr_y_means.get(treatment_index, [])
            e_t_mean = np.mean(e_t_values) if e_t_values else np.nan

            results.append({
                'treatment_index': treatment_index,
                'Binary Y 0 Count': count_0,
                'Binary Y 1 Count': count_1,
                'Proportion 0': proportion_0,
                'Proportion 1': proportion_1,
                'Y': self.protect_df.loc[treatment_index, 'Binary Y'],  # 新增一列标识Y
                'Pr(Y=1)': self.protect_df.loc[treatment_index, 'Pr(Y=1)'],  # 新增 Pr(Y=1)
                # 'Binary Prediction': self.protect_df.loc[treatment_index, 'Binary Prediction'],  # 新增 Binary Prediction
                'E[T]': e_t_mean  # 新增 E[T]
            })

            protect_binary_y_values.append(self.protect_df.loc[treatment_index, 'Binary Y'])

        protect_count_0 = (np.array(protect_binary_y_values) == 0).sum()
        protect_count_1 = (np.array(protect_binary_y_values) == 1).sum()
        protect_total = len(protect_binary_y_values)
        protect_proportion_0 = protect_count_0 / protect_total
        protect_proportion_1 = protect_count_1 / protect_total

        protect_data = {
            'Type': ['Protect', 'Protect'],
            'Binary Y': [0, 1],
            'Count': [protect_count_0, protect_count_1],
            'Proportion': [protect_proportion_0, protect_proportion_1]
        }

        results_df = pd.DataFrame(results)
        protect_df = pd.DataFrame(protect_data)

        # results_df.to_csv(f'{label}_group_ground_truth.csv', index=False)
        # protect_df.to_csv(f'{label}_protect_group_ground_truth.csv', index=False)

        return results_df, protect_df



    def plot_case_density(self, protect_df, mean_values, treatment_index, bw_adjust=2, show_line=True,
                          line_color='#B02425',
                          density_color='#84BA84'):
        """
        Plot the density of Pr(Y=1) means for a specified treatment index
        and print the kurtosis, standard deviation, and density mean.
        """
        line_value = protect_df.iloc[treatment_index]['Pr(Y=1)']
        if treatment_index in mean_values:
            data = mean_values[treatment_index]
            data_kurtosis = kurtosis(data)
            data_std = np.std(data)
            data_mean = np.mean(data)

            # Print the calculated indicators
            print(f"Kurtosis: {data_kurtosis}")
            print(f"Standard Deviation: {data_std}")
            print(f"Density Mean: {data_mean}")
            print(f"Line Value (A): {line_value}")

            sns.set_context("talk", rc={"lines.linewidth": 2.5})
            plt.figure(figsize=(6,5))
            ax = sns.kdeplot(data, shade=False, bw_adjust=bw_adjust, color=density_color, linewidth=7, label='Peers')
            if show_line:
                plt.axvline(x=line_value, color=line_color, linestyle='--', linewidth=7, label='Selected micro-firm')

            plt.xlabel('$Pr(\hat Y=1|X=\\mathbf{x})$', fontsize=15)
            plt.ylabel('Density', fontsize=15)
            plt.xlim(0.5, 1)
            # 加粗刻度标签
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,
                       frameon=False, fontsize=15)
            plt.tight_layout()
            plt.grid(False)
            plt.savefig(f'{self.result_type}_{self.comparison_type}_{treatment_index}_case_density.pdf', format='pdf', dpi=300)
            plt.show()
            # Save the data to CSV
            density_df = pd.DataFrame({
                'Peers Value': data
            })
            density_df['Line Value (A)'] = line_value
            density_df.to_csv(f'{self.result_type}_{self.comparison_type}_{treatment_index}_case_density_stats.csv',
                              index=False)
        else:
            print("Treatment index not found in the provided mean values.")

    def case_ground_truth(self, specified_treatment_index):
        """
        根据指定的 treatment_index 在 bootstrapped_samples 中寻找 'control_index'，
        然后在 nonprotect_df 的 index 中寻找相应的 Binary Y，
        计算 Binary Y 为 0 和 1 的个数以及比例。

        参数：
        - specified_treatment_index (int): 指定的 treatment_index 值。

        返回：
        - pd.DataFrame: 包含指定 treatment_index 的 Binary Y 值及其对应的 control_index 中 Binary Y 为 0 和 1 的个数及比例的 DataFrame。
        """
        protect_binary_y = self.protect_df.loc[specified_treatment_index, 'Binary Y']
        control_indices = self.bootstrapped_samples.loc[
            self.bootstrapped_samples['treatment_index'] == specified_treatment_index, 'control_index']
        control_indices = control_indices.explode().tolist()
        control_binary_y_values = self.nonprotect_df.loc[control_indices, 'Binary Y']
        count_0 = (control_binary_y_values == 0).sum()
        count_1 = (control_binary_y_values == 1).sum()
        total = len(control_binary_y_values)
        proportion_0 = count_0 / total
        proportion_1 = count_1 / total

        data = {
            'Specified Treatment Index': [specified_treatment_index],
            'Binary Y': [protect_binary_y],
            'Control Count 0': [count_0],
            'Control Count 1': [count_1],
            'Control Proportion 0': [proportion_0],
            'Control Proportion 1': [proportion_1]
        }

        return pd.DataFrame(data)

    def counts_peers(self, color_df, matched_df):
        """
        计算 color_df['treatment_index'] 在 matched_df['treatment_index'] 中出现的次数。

        参数:
        color_df (pd.DataFrame): 包含 'treatment_index' 列的 DataFrame。
        matched_df (pd.DataFrame): 包含 'treatment_index' 列的 DataFrame。

        返回:
        pd.DataFrame: 在 color_df 基础上增加了一列 'count'，表示每个 'treatment_index' 在 matched_df 中出现的次数。
        """
        # 计算每个 treatment_index 在 matched_df 中出现的次数
        counts = matched_df['treatment_index'].value_counts()

        # 将 counts 转换为 DataFrame，以便与 color_df 进行合并
        counts_df = counts.reset_index()
        counts_df.columns = ['treatment_index', 'count']

        # 将 counts_df 与 color_df 合并，以便将出现次数添加到 color_df
        peers_num_df = pd.merge(color_df, counts_df, on='treatment_index', how='left')

        # 将合并后的 DataFrame 中的 NaN 值替换为 0（即那些在 matched_df 中未出现的 treatment_index）
        peers_num_df['count'] = peers_num_df['count'].fillna(0).astype(int)

        plt.figure(figsize=(10, 6))
        sns.kdeplot(peers_num_df['count'], shade=True)
        plt.title('Density Plot of Peers Number')
        plt.xlabel('Peers Number')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()

        return peers_num_df

