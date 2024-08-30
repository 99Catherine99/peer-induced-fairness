import pandas as pd

class Categories:
    def __init__(self, double_sided_results, protect_df, single_less_less_than_color_df):
        self.double_sided_results = double_sided_results
        self.protect_df = protect_df
        self.single_less_less_than_color_df = single_less_less_than_color_df

    def calculate_and_save_ratios(self, binary_y_values, output_csv=False):
        ratios = []
        for binary_y_value in binary_y_values:
            # Filter out rows where the 'is_significant' column is not 'Unknown'
            df = self.double_sided_results[self.double_sided_results['is_significant'] != 'Unknown']
            # Filter rows in protect_df where Binary Y equals the specified value
            protect_accept_df = self.protect_df[self.protect_df['Binary Y'] == binary_y_value]
            # Get the indices of protect_accept_df
            protect_accept_indices = protect_accept_df.index

            # Get the 'treatment_index' column from double_sided_results
            df_index = df['treatment_index']

            # Find the intersection of protect_accept_indices and double_sided_treatment_indices
            matching_indices = protect_accept_indices[protect_accept_indices.isin(df_index)]

            # Get the 'treatment_index' column from color_df
            treatment_indices = self.single_less_less_than_color_df['treatment_index']

            # Calculate the number of times matching_indices appear in treatment_indices
            count = matching_indices.isin(treatment_indices).sum()
            ratio = count / len(matching_indices) if len(matching_indices) > 0 else 0

            ratios.append(ratio)

        # Create a DataFrame and optionally save it to a CSV file
        data = {
            'Binary Y': binary_y_values,
            'Ratio': ratios
        }
        df = pd.DataFrame(data)

        if output_csv:
            df.to_csv('ratios.csv', index=False)

        return df
