from metrics.fairness_calculator import FairnessCalculator


class StatisticalParityCalculator(FairnessCalculator):
    def compute(self):
        protect_df, nonprotect_df, population_df = self._split_groups()

        protect_sp = protect_df[self.prediction_col].value_counts()[1] / len(protect_df)
        nonprotect_sp = nonprotect_df[self.prediction_col].value_counts()[1] / len(nonprotect_df)
        population_sp = population_df[self.prediction_col].value_counts()[1] / len(population_df)
        diff_sp = nonprotect_sp - protect_sp

        # print('protect_sp: ', protect_sp)
        # print('nonprotect_sp: ', nonprotect_sp)
        # print('population_sp: ', population_sp)
        # print('diff_sp: ', diff_sp)

        return protect_sp, nonprotect_sp, population_sp, diff_sp