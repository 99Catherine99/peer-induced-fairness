from metrics.distance_calculator import DistanceCalculatorBase


class KL_DivergenceCalculator(DistanceCalculatorBase):
    """
    A class for calculating the Kullback-Leibler (KL) divergence between two datasets.

    Methods:
    compute(self): Calculate and return the KL divergence between the two datasets.

    Inherits from:
    DistanceCalculatorBase: A base class providing common methods for distance calculators.

    Attributes:
    data1 (numpy.ndarray): The first dataset for which to calculate KL divergence.
    data2 (numpy.ndarray): The second dataset for which to calculate KL divergence.
    """

    def compute(self):
        """
        Calculate and return the KL divergence between the two datasets.
        Returns: kl_divergence (float): The calculated KL divergence between the two datasets.
        """

        kl_divergence = self.kl(self.data1, self.data2)
        print("KL Divergence:", kl_divergence)
        return kl_divergence
