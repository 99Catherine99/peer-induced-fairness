from scipy.stats import wasserstein_distance
from metrics.distance_calculator import DistanceCalculatorBase


class Wasserstein_DistanceCalculator(DistanceCalculatorBase):
    """
    A class for calculating the Wasserstein distance (Earth Mover's distance, EMD) between two datasets.

    Methods:
    compute(self): Calculate and return the Wasserstein distance between the two datasets.

    Inherits from:
    DistanceCalculatorBase: A base class providing common methods for distance calculators.

    Attributes:
    data1 (numpy.ndarray): The first dataset for which to calculate the Wasserstein distance.
    data2 (numpy.ndarray): The second dataset for which to calculate the Wasserstein distance.
    """

    def compute(self):
        """
        Calculate and return the Wasserstein distance between the two datasets.
        Returns: wasserstein (float): The calculated Wasserstein distance between the two datasets.
        """

        wasserstein = wasserstein_distance(self.data1, self.data2)
        print("Wasserstein Distance:", wasserstein)
        return wasserstein