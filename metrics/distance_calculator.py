import numpy as np

class DistanceCalculatorBase:
    """
    A base class for calculating distances between two sets of data.

    Methods:
    kl(p, q): Define the Kullback-Leibler (KL) divergence between two probability distributions.
    __init__(data1, data2): Initialize the DistanceCalculatorBase instance with two sets of data.
    compute(self): Calculate the distance between the two sets of data, which should be implemented in subclass.
    """

    @staticmethod
    def kl(p, q):
        """
        Calculate the Kullback-Leibler (KL) divergence between two probability distributions.

        Args:
        p (array-like): The first probability distribution.
        q (array-like): The second probability distribution.

        Returns:
        kl_divergence (float): The KL divergence between the two distributions.
        """

        return np.sum(np.where(p != 0, p * np.log(p / q), 0))

    def __init__(self, data1, data2):
        """
        Initialize the DistanceCalculatorBase instance with two sets of data.

        Args:
        data1 (array-like): The first set of data.
        data2 (array-like): The second set of data.
        """

        self.data1 = data1
        self.data2 = data2

    def compute(self):
        """
        Calculate the distance between the two sets of data.

        This method should be implemented in subclasses.

        Raises:
        NotImplementedError: This method should be implemented in subclasses.
        """

        raise NotImplementedError("Subclasses should implement the 'compute' method.")