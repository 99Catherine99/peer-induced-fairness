from metrics.fairness_calculator import FairnessCalculator
from metrics.sp_calculator import StatisticalParityCalculator


class DiscriminationRatioCalculator(FairnessCalculator):
    """
    A class for calculating Discrimination Ratio for protected and non-protected groups.

    This class inherits from the FairnessCalculator class.

    Methods:
    compute(self): Calculate Discrimination Ratio between protected and non-protected groups.
                   Returns the Discrimination Ratio.
    """

    def compute(self):
        """
        Calculate Discrimination Ratio between protected and non-protected groups.

        Returns:
        discrimination_ratio (float): Discrimination Ratio representing the ratio of Statistical Parity
                                     between protected and non-protected groups.
        """

        statistical_parity_calculator = StatisticalParityCalculator(self.df, self.group_col, self.prediction_col,
                                                                    self.true_col)
        protect_sp, nonprotect_sp, population_sp = statistical_parity_calculator.compute()

        discrimination_ratio = protect_sp / nonprotect_sp

        print('discrimination_ratio: ', discrimination_ratio)

        return discrimination_ratio