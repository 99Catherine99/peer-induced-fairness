import numpy as np
from sklearn.neighbors import KernelDensity

class CoefficientofVariationCalculator:
    def __init__(self, data, random_seed=None):
        """
        Initialize a CoefficientofVariationCalculator object.

        Parameters:
        - data: Input data for coefficient of variation calculation.
        - random_seed: Random seed for reproducibility.
        """
        self.data = data
        self.random_seed = random_seed

    def calculate_coefficient_of_variation(self):
        """
        Calculate the coefficient of variation for the input data.

        Returns:
        - coefficient_of_variation: Coefficient of variation in percentage.
        """
        if len(self.data) < 2:
            raise ValueError("Data should have at least two values for calculation.")

        mean_value = np.mean(self.data)
        std_deviation = np.std(self.data)

        coefficient_of_variation = (std_deviation / mean_value) * 100  # 以百分比表示
        print(f"Real Data - Coefficient of Variation: {coefficient_of_variation:.2f}%")
        return coefficient_of_variation

    def calculate_kde_coefficient_of_variation(self, bandwidth=0.01, n_samples=10000):
        """
        Calculate the coefficient of variation using KDE-generated samples.

        Parameters:
        - bandwidth: Bandwidth for Kernel Density Estimation (KDE).
        - n_samples: Number of KDE-generated samples.

        Returns:
        - coefficient_of_variation: Coefficient of variation in percentage.
        """
        # 使用 KDE 生成随机样本
        if self.random_seed is not None:
            np.random.seed(self.random_seed)  # 设置随机种子
        kde = KernelDensity(bandwidth=bandwidth)
        # 转换Series对象为NumPy数组，然后进行reshape
        data_array = np.array(self.data).reshape(-1, 1)
        kde.fit(data_array)  # Reshape data to a column vector
        kde_samples = kde.sample(n_samples)

        if len(kde_samples) < 2:
            raise ValueError("KDE samples should have at least two values for calculation.")

        mean_value = np.mean(kde_samples)
        std_deviation = np.std(kde_samples)

        coefficient_of_variation = (std_deviation / mean_value) * 100  # 以百分比表示
        print(f"KDE Samples - Coefficient of Variation: {coefficient_of_variation:.2f}%")
        return coefficient_of_variation
