from metrics.fairness_calculator import FairnessCalculator


class NPVCalculator(FairnessCalculator):
    def compute(self, prediction_value=0):
        protect_df, nonprotect_df, population_df = self._split_groups()

        protect_yhat_df = protect_df[protect_df[self.prediction_col] == prediction_value]
        nonprotect_yhat_df = nonprotect_df[nonprotect_df[self.prediction_col] == prediction_value]
        population_yhat_df = population_df[population_df[self.prediction_col] == prediction_value]

        protect_npv = protect_yhat_df[self.true_col].value_counts()[0] / len(protect_yhat_df)
        nonprotect_npv = nonprotect_yhat_df[self.true_col].value_counts()[0] / len(nonprotect_yhat_df)
        population_npv = population_yhat_df[self.true_col].value_counts()[0] / len(population_yhat_df)
        diff_npv = nonprotect_npv - protect_npv

        # print('protect_bacc: ', protect_npv)
        # print('nonprotect_bacc: ', nonprotect_npv)
        # print('population_bacc: ', population_npv)
        # print('diff_npv: ', diff_npv)

        return protect_npv, nonprotect_npv, population_npv, diff_npv