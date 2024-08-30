from metrics.fairness_calculator import FairnessCalculator


class TPRCalculator(FairnessCalculator):
    def compute(self, true_value=1):
        protect_df, nonprotect_df, population_df = self._split_groups()

        protect_y_df = protect_df[protect_df[self.true_col] == true_value]
        nonprotect_y_df = nonprotect_df[nonprotect_df[self.true_col] == true_value]
        population_y_df = population_df[population_df[self.true_col] == true_value]

        protect_tpr = protect_y_df[self.prediction_col].value_counts()[1] / len(protect_y_df) if len(protect_y_df) > 0 else 0
        nonprotect_tpr = nonprotect_y_df[self.prediction_col].value_counts()[1] / len(nonprotect_y_df) if len(nonprotect_y_df) > 0 else 0
        population_tpr = population_y_df[self.prediction_col].value_counts()[1] / len(population_y_df) if len(population_y_df) > 0 else 0
        diff_tpr = nonprotect_tpr - protect_tpr

        # print('protect_tpr: ', protect_tpr)
        # print('nonprotect_tpr: ', nonprotect_tpr)
        # print('population_tpr: ', population_tpr)
        # print('diff_tpr: ', diff_tpr)

        return protect_tpr, nonprotect_tpr, population_tpr, diff_tpr