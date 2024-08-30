import numpy as np
import pandas as pd

class CaplierMatching:
    def __init__(self, df, group_col, K=15, caplier_ratio=0.2):
        self.df = df
        self.group_col = group_col
        self.K = K
        self.caplier_ratio = caplier_ratio
        self.protect_df, self.nonprotect_df, self.protect_pr, self.nonprotect_pr, self.protect_ps, self.nonprotect_ps = self.preparation_for_psm()
        self.weighted_protect_df, self.weighted_nonprotect_df, self.weighted_protect_ps, self.weighted_nonprotect_ps = self.calculate_weighted_ps()

    def preparation_for_psm(self):
        protect_df = self.df[self.df[self.group_col] == 0]
        nonprotect_df = self.df[self.df[self.group_col] == 1]
        self.protect_pr = protect_df['Pr(Y=1)'].tolist()
        self.nonprotect_pr = nonprotect_df['Pr(Y=1)'].tolist()
        protect_ps = protect_df['Pr(S=0)'].tolist()
        nonprotect_ps = nonprotect_df['Pr(S=0)'].tolist()
        return protect_df, nonprotect_df, self.protect_pr, self.nonprotect_pr, protect_ps, nonprotect_ps

    def calculate_weighted_ps(self):
        total_count = len(self.df)
        protect_ratio = len(self.protect_df) / total_count
        nonprotect_ratio = len(self.nonprotect_df) / total_count
        weighted_protect_ps = [ps / protect_ratio for ps in self.protect_ps]
        weighted_nonprotect_ps = [(1-ps) / nonprotect_ratio for ps in self.nonprotect_ps]
        return self.protect_df, self.nonprotect_df, weighted_protect_ps, weighted_nonprotect_ps


    def caplier_matching(self):
        weighted_protect_ps_np = np.array(self.weighted_protect_ps)
        weighted_nonprotect_ps_np = np.array(self.weighted_nonprotect_ps)
        caplier = self.caplier_ratio * np.std(weighted_protect_ps_np)
        matched_data_points = []

        for protect_index, weighted_protect_ps in enumerate(weighted_protect_ps_np):
            # calculate the distance
            differences = np.abs(weighted_nonprotect_ps_np - weighted_protect_ps)
            mask = differences <= caplier
            matching_indices = np.where(mask)[0]

            # If there are matching points, sort the matching indexes by the difference size
            if matching_indices.size > 0:
                sorted_indices = matching_indices[np.argsort(differences[matching_indices])]

                for match_index in sorted_indices:
                    control_ps = weighted_nonprotect_ps_np[match_index]
                    matched_data_points.append({
                        'treatment_index': protect_index,
                        'treatment_ps': weighted_protect_ps,
                        'control_index': match_index,
                        'control_ps': control_ps,
                        'abs_difference': differences[match_index]  # 已经计算的差异
                    })

        return pd.DataFrame(matched_data_points)


    def individual_caplier_matching(self):
        weighted_protect_ps_np = np.array(self.weighted_protect_ps)
        weighted_nonprotect_ps_np = np.array(self.weighted_nonprotect_ps)
        caplier = self.caplier_ratio * np.std(weighted_protect_ps_np)
        matched_data_points = []

        for protect_index, weighted_protect_ps in enumerate(weighted_protect_ps_np):
            pr = self.protect_pr[protect_index]
            # individual_caplier = caplier / math.exp(pr)
            individual_caplier = caplier / weighted_protect_ps
            differences = np.abs(weighted_nonprotect_ps_np - weighted_protect_ps)
            mask = differences <= individual_caplier
            matching_indices = np.where(mask)[0]

            # Sort the matches that meet the conditions and get the sorted index
            if matching_indices.size > 0:
                sorted_indices = matching_indices[np.argsort(differences[matching_indices])]

                for match_index in sorted_indices:
                    control_ps = weighted_nonprotect_ps_np[match_index]
                    matched_data_points.append({
                        'treatment_index': protect_index,
                        'treatment_ps': weighted_protect_ps,
                        'control_index': match_index,
                        'control_ps': control_ps,
                        'abs_difference': differences[match_index]  # 使用已计算的差异
                    })

        return pd.DataFrame(matched_data_points)

    def filter_matches(self, matched_df):
        # Count the number of original matches
        initial_count = len(self.protect_df)

        # Ensure that each protected point has at least K matches
        sufficient_matches_df = matched_df.groupby('treatment_index').filter(lambda x: len(x) >= self.K)

        # Select the top K matches with the smallest differences from the ones that meet the conditions
        filtered_df = (sufficient_matches_df
                        .sort_values(by=['treatment_index', 'abs_difference'])
                        .groupby('treatment_index')
                        .head(self.K))

        # Calculate statistics after filtering
        remaining_count = filtered_df['treatment_index'].nunique()
        removed_count = initial_count - remaining_count


        print(f"Removed {removed_count} data points.")
        print(f"Remaining {remaining_count} data points after filtering.")

        return filtered_df



