import pandas as pd
from metrics.bacc_calculator import BalancedAccuracyCalculator
from metrics.fnr_calculator import FNRCalculator
from metrics.fpr_calculator import FPRCalculator
from metrics.npv_calculator import NPVCalculator
from metrics.ppv_calculator import PPVCalculator
from metrics.proportion_calculator import ProportionCalculator
from metrics.sp_calculator import StatisticalParityCalculator
from metrics.tnr_calculator import TNRCalculator
from metrics.tpr_calculator import TPRCalculator


class FairnessMetricsCalculator:
    def __init__(self, df, group_col, prediction_col, true_col):
        """
        Initialize the FairnessMetricsCalculator.

        Parameters:
        - df (DataFrame): The input DataFrame containing the data.
        - group_col (str): The column specifying the protected group.
        - prediction_col (str): The column containing model predictions.
        - true_col (str): The column containing true labels.
        """
        self.df = df
        self.group_col = group_col
        self.prediction_col = prediction_col
        self.true_col = true_col

    def calculate_metrics(self, file_prefix):
        """
        Calculate fairness metrics and store them in a CSV file.

        Parameters:
        - file_prefix (str): Prefix for the output CSV file name.

        Returns:
        - metrics_df (DataFrame): DataFrame containing fairness metrics.
        """
        metrics = []

        def calculate_and_store(metric_calculator, metric_name, **kwargs):
            """
            Calculate a fairness metric using a specified calculator and store the result in the metrics list.

            Parameters:
            - metric_calculator (class): The class responsible for calculating the fairness metric.
            - metric_name (str): The name of the fairness metric.
            - **kwargs: Additional keyword arguments to pass to the metric calculator.

            This function initializes a metric calculator with the given data and calculates the specified fairness metric.
            The calculated metric result is then appended to the metrics list along with its name.

            Returns:
            None
            """
            calculator = metric_calculator(self.df, self.group_col, self.prediction_col, self.true_col)
            result = calculator.compute(**kwargs)
            metrics.append([metric_name] + list(result))

        calculate_and_store(ProportionCalculator, 'Proportion')
        calculate_and_store(StatisticalParityCalculator, 'Statistical Parity')
        calculate_and_store(TPRCalculator, 'TPR', true_value=1)
        calculate_and_store(FPRCalculator, 'FPR', true_value=0)
        # calculate_and_store(TNRCalculator, 'TNR', true_value=0)
        # calculate_and_store(FNRCalculator, 'FNR', true_value=1)
        # calculate_and_store(BalancedAccuracyCalculator, 'Balanced Accuracy')
        calculate_and_store(PPVCalculator, 'PPV', prediction_value=1)
        # calculate_and_store(NPVCalculator, 'NPV', prediction_value=0)
        # calculate_and_store(DiscriminationRatioCalculator, 'Discrimination Ratio')
        # calculate_and_store(CVCalculator, 'CV')

        # Create a DataFrame from the metrics list
        metrics_df = pd.DataFrame(metrics, columns=['Metric', 'Protected', 'Nonprotected', 'Population', 'Difference'])
        output_filename = f'{file_prefix}_fairness_metrics.csv'
        metrics_df.to_csv(output_filename, index=True)

        return metrics_df