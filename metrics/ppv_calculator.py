from metrics.fairness_calculator import FairnessCalculator


class PPVCalculator(FairnessCalculator):
    def compute(self, prediction_value=1):
        protect_df, nonprotect_df, population_df = self._split_groups()

        protect_yhat_df = protect_df[protect_df[self.prediction_col] == prediction_value]
        nonprotect_yhat_df = nonprotect_df[nonprotect_df[self.prediction_col] == prediction_value]
        population_yhat_df = population_df[population_df[self.prediction_col] == prediction_value]

        protect_ppv = protect_yhat_df[self.true_col].value_counts()[1] / len(protect_yhat_df)
        nonprotect_ppv = nonprotect_yhat_df[self.true_col].value_counts()[1] / len(nonprotect_yhat_df)
        population_ppv = population_yhat_df[self.true_col].value_counts()[1] / len(population_yhat_df)
        diff_ppv = nonprotect_ppv - protect_ppv

        # print('protect_bacc: ', protect_ppv)
        # print('nonprotect_bacc: ', nonprotect_ppv)
        # print('population_bacc: ', population_ppv)
        # print('diff_ppv: ', diff_ppv)

        return protect_ppv, nonprotect_ppv, population_ppv, diff_ppv