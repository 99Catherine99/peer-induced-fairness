from metrics.fairness_calculator import FairnessCalculator

class TNRCalculator(FairnessCalculator):
    def compute(self, true_value=0):
        protect_df, nonprotect_df, population_df = self._split_groups()

        protect_y_df = protect_df[protect_df[self.true_col] == true_value]
        nonprotect_y_df = nonprotect_df[nonprotect_df[self.true_col] == true_value]
        population_y_df = population_df[population_df[self.true_col] == true_value]

        protect_tnr = protect_y_df[self.prediction_col].value_counts()[0] / len(protect_y_df) if len(protect_y_df) > 0 else 0
        nonprotect_tnr = nonprotect_y_df[self.prediction_col].value_counts()[0] / len(nonprotect_y_df) if len(nonprotect_y_df) > 0 else 0
        population_tnr = population_y_df[self.prediction_col].value_counts()[0] / len(population_y_df) if len(population_y_df) > 0 else 0
        diff_tnr = nonprotect_tnr-protect_tnr

        # print('protect_tnr: ', protect_tnr)
        # print('nonprotect_tnr: ', nonprotect_tnr)
        # print('population_tnr: ', population_tnr)
        # print('diff_tnr: ', diff_tnr)

        return protect_tnr, nonprotect_tnr, population_tnr, diff_tnr