from metrics.fairness_calculator import FairnessCalculator
from metrics.sp_calculator import StatisticalParityCalculator


class CVCalculator(FairnessCalculator):
    """
    A class for calculating CV Score (CV) and Absolute CV scores for protected and non-protected groups.

    This class inherits from the FairnessCalculator class.

    Methods:
    compute(self): Calculate CV and Absolute CV scores for protected and non-protected groups.
                   Returns the CV and Absolute CV scores.
    """

    def compute(self):
        """
        Calculate CV Score (CV) and Absolute CV scores for protected and non-protected groups.

        Returns:
        cv_score (float): CV score representing the difference in Statistical Parity
                          between protected and non-protected groups.
        abs_cv_score (float): Absolute CV score representing the absolute difference in Statistical Parity
                              between protected and non-protected groups.
        """

        statistical_parity_calculator = StatisticalParityCalculator(self.df, self.group_col, self.prediction_col,
                                                                    self.true_col)
        protect_sp, nonprotect_sp, population_sp = statistical_parity_calculator.compute()

        cv_score = nonprotect_sp - protect_sp
        abs_cv_score = abs(cv_score)

        print('cv_score: ', cv_score)
        print('abs_cv_score: ', abs_cv_score)

        return cv_score, abs_cv_score