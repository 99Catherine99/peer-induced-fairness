from metrics.fairness_calculator import FairnessCalculator

class ProportionCalculator(FairnessCalculator):
    def compute(self):
        protect_df, nonprotect_df, population_df = self._split_groups()

        protect_pro = protect_df[self.true_col].value_counts()[1] / len(protect_df)
        nonprotect_pro = nonprotect_df[self.true_col].value_counts()[1] / len(nonprotect_df)
        population_pro = population_df[self.true_col].value_counts()[1] / len(population_df)
        diff_pro = nonprotect_pro-protect_pro

        # print('protect_pro: ', protect_pro)
        # print('nonprotect_pro: ', nonprotect_pro)
        # print('population_pro: ', population_pro)
        # print('diff_pro: ', diff_pro)

        return protect_pro, nonprotect_pro, population_pro, diff_pro