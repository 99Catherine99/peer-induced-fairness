import pandas as pd
import np as numpy
import matplotlib.pyplot as plt


class MatchCountVisualizer:
    def __init__(self, data):
        self.data = data

    def calculate_match_counts(self):
        """
        Calculate the number of matches for each treatment_index.
        """
        self.match_counts = self.data.groupby('treatment_index').size()
        return self.match_counts

    def plot_frequency_distribution(self, bin_width=35):
        """
        Plot the frequency distribution of the match counts.

        Parameters:
        - bin_width (int): The width of each bin in the histogram.
        """
        # Calculate the bins for the histogram
        bins = np.arange(0, self.match_counts.max() + bin_width, bin_width)

        # Plot the frequency distribution
        plt.figure(figsize=(7, 6))
        plt.hist(self.match_counts, bins=bins, density=False, alpha=0.75, edgecolor='black')
        plt.xlabel('Peers Number', fontsize=15)
        plt.ylabel('Frequency', fontsize=15)
        plt.xticks(bins)
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig('frequency_bar.png')

        # Display the plot
        plt.show()

    def export_data(self):
        """
        Export the match counts data to a CSV file.
        """
        self.match_counts.to_csv('frequency.csv')

    def calculate_small_frequencies(self):
        """
        Calculate the frequency of match counts that are less than 35.
        """
        count_less_than = self.match_counts[self.match_counts < 35].count()
        print(f"Number of less than 35 peers: {count_less_than}")
        return count_less_than

    def analyze_unique_treatment_indices(self):
        """
        Calculate the number of unique treatment indices in the data.
        """
        unique_count = self.data['treatment_index'].nunique()
        print(f"Number of unique treatment indices: {unique_count}")
        return unique_count
