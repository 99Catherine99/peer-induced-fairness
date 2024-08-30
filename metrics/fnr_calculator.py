from metrics.fairness_calculator import FairnessCalculator

class FNRCalculator(FairnessCalculator):
    def compute(self, true_value=1):
        protect_df, nonprotect_df, population_df = self._split_groups()

        protect_y_df = protect_df[protect_df[self.true_col] == true_value]
        nonprotect_y_df = nonprotect_df[nonprotect_df[self.true_col] == true_value]
        population_y_df = population_df[population_df[self.true_col] == true_value]

        protect_fnr = protect_y_df[self.prediction_col].value_counts()[0] / len(protect_y_df) if len(protect_y_df) > 0 else 0
        nonprotect_fnr = nonprotect_y_df[self.prediction_col].value_counts()[0] / len(nonprotect_y_df) if len(nonprotect_y_df) > 0 else 0
        population_fnr = population_y_df[self.prediction_col].value_counts()[0] / len(population_y_df) if len(population_y_df) > 0 else 0
        diff_fnr = nonprotect_fnr - protect_fnr

        # print('protect_fnr: ', protect_fnr)
        # print('nonprotect_fnr: ', nonprotect_fnr)
        # print('population_fnr: ', population_fnr)
        # print('diff_fnr: ', diff_fnr)

        return protect_fnr, nonprotect_fnr, population_fnr, diff_fnr
