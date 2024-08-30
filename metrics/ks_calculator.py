from scipy.stats import ks_2samp
from metrics.distance_calculator import DistanceCalculatorBase


class KS_DistanceCalculator(DistanceCalculatorBase):
    """
    A class for calculating the Kolmogorov-Smirnov (KS) distance between two datasets.

    Methods:
    compute(self): Calculate and return the KS distance and p-value between the two datasets.

    Inherits from:
    DistanceCalculatorBase: A base class providing common methods for distance calculators.

    Attributes:
    data1 (numpy.ndarray): The first dataset for which to calculate KS distance.
    data2 (numpy.ndarray): The second dataset for which to calculate KS distance.
    """

    def compute(self):
        """
        Calculate and return the KS distance and p-value between the two datasets.
        Returns: ks_statistic (float): The calculated KS distance between the two datasets, ks_p_value (float): The p-value associated with the KS test.
        """

        ks_statistic, ks_p_value = ks_2samp(self.data1, self.data2)
        print("KS Distance:", ks_statistic)
        print('KS P Value:', ks_p_value)
        return ks_statistic, ks_p_value