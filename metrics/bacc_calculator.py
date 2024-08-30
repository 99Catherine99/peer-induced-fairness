from metrics.fairness_calculator import FairnessCalculator
from metrics.tnr_calculator import TNRCalculator
from metrics.tpr_calculator import TPRCalculator


class BalancedAccuracyCalculator(FairnessCalculator):
    def compute(self):
        # Calculate TPR for both protected and non-protected groups
        tpr_calculator = TPRCalculator(self.df, self.group_col, self.prediction_col, self.true_col)
        protect_tpr, nonprotect_tpr, population_tpr, diff_tpr = tpr_calculator.compute(true_value=1)  # No need to pass true_value

        # Calculate TNR for both protected and non-protected groups
        tnr_calculator = TNRCalculator(self.df, self.group_col, self.prediction_col, self.true_col)
        protect_tnr, nonprotect_tnr, population_tnr, diff_tnr = tnr_calculator.compute(true_value=0)  # No need to pass true_value

        # Calculate Balanced Accuracy
        protect_bacc = (protect_tpr + protect_tnr) / 2
        nonprotect_bacc = (nonprotect_tpr + nonprotect_tnr) / 2
        population_bacc = (population_tpr + population_tnr) / 2
        diff_bacc = nonprotect_bacc - protect_bacc

        # print('protect_bacc: ', protect_bacc)
        # print('nonprotect_bacc: ', nonprotect_bacc)
        # print('population_bacc: ', population_bacc)
        # print('diff_bacc: ', diff_bacc)

        return protect_bacc, nonprotect_bacc, population_bacc, diff_bacc