from metrics.fairness_calculator import FairnessCalculator

class FPRCalculator(FairnessCalculator):
    def compute(self, true_value=0):
        protect_df, nonprotect_df, population_df = self._split_groups()

        protect_y_df = protect_df[protect_df[self.true_col] == true_value]
        nonprotect_y_df = nonprotect_df[nonprotect_df[self.true_col] == true_value]
        population_y_df = population_df[population_df[self.true_col] == true_value]

        protect_fpr = protect_y_df[self.prediction_col].value_counts()[1] / len(protect_y_df) if len(protect_y_df) > 0 else 0
        nonprotect_fpr = nonprotect_y_df[self.prediction_col].value_counts()[1] / len(nonprotect_y_df) if len(nonprotect_y_df) > 0 else 0
        population_fpr = population_y_df[self.prediction_col].value_counts()[1] / len(population_y_df) if len(population_y_df) > 0 else 0
        diff_fpr = nonprotect_fpr - protect_fpr

        # print('protect_fpr: ', protect_fpr)
        # print('nonprotect_fpr: ', nonprotect_fpr)
        # print('population_fpr: ', population_fpr)
        # print('diff_fpr: ', diff_fpr)

        return protect_fpr, nonprotect_fpr, population_fpr, diff_fpr