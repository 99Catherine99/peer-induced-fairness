import pandas as pd

class MatchingDescribe:
    def __init__(self, matched_df, treatment_df, control_df, threshold):
        """
        Initialize the MatchingDescribe class.

        Parameters:
        - matched_df (DataFrame): DataFrame containing information about matched treatment and control pairs.
        - treatment_df (DataFrame): DataFrame representing the treatment group.
        - control_df (DataFrame): DataFrame representing the control group.
        - threshold (float): A numerical threshold used for defining a boundary around the mean difference.
        """
        self.matched_df = matched_df
        # treatment_df and control_df could change following the matching method
        self.treatment_df = treatment_df
        self.control_df = control_df
        self.threshold = threshold

    def describe(self, file_prefix):
        """
        Generate a summary description of matched treatment and control groups.

        Parameters:
        - file_prefix (str): A prefix used for naming the output CSV file.

        Returns:
        - matching_describe_df (DataFrame): DataFrame containing matching group descriptions.
        """
        matching_describe = []

        for treatment_index in self.matched_df['treatment_index'].unique():
            control_select_index = self.matched_df[self.matched_df['treatment_index'] == treatment_index]['control_index'].tolist()
            control_select_data_df = self.control_df[self.control_df.index.isin(control_select_index)]
            control_mean_pr = control_select_data_df['Pr(Y=1)'].mean()

            treatment_select_index = self.matched_df[self.matched_df['treatment_index'] == treatment_index]['treatment_index'].tolist()
            treatment_select_data_df = self.treatment_df[self.treatment_df.index.isin(treatment_select_index)]
            treatment_mean_pr = treatment_select_data_df['Pr(Y=1)'].mean()

            mean_difference = control_mean_pr - treatment_mean_pr
            lower_boundary = treatment_mean_pr - self.threshold
            upper_boundary = treatment_mean_pr + self.threshold

            check = 0 if control_mean_pr < lower_boundary else (
                1 if lower_boundary <= control_mean_pr <= upper_boundary else 2)

            matching_describe.append({
                'Treatment_Group_Index': treatment_index,
                'Treatment_Group_Mean_Pr': treatment_mean_pr,
                'Control_Group_Mean_Pr': control_mean_pr,
                'Mean_Difference': mean_difference,
                'Lower_Boundary': lower_boundary,
                'Upper_Boundary': upper_boundary,
                'Check': check
            })

        matching_describe_df = pd.DataFrame(matching_describe)
        output_filename = f'{file_prefix}_matching_describe.csv'
        matching_describe_df.to_csv(output_filename, index=True)

        # Calculate counts and proportions
        counts = matching_describe_df['Check'].value_counts()
        proportions = counts / counts.sum()
        # Create a DataFrame for counts and proportions
        counts_proportions_df = pd.DataFrame({'Counts': counts, 'Proportions': proportions})
        counts_proportions_df.index.name = 'Check Value'
        # Print counts and proportions in one row
        print(counts_proportions_df)

        return matching_describe_df
