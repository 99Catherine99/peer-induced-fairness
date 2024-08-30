import pandas as pd
from analysis.ratios import ValueRatioTableGenerator


class EmpiricalDataExporter:
    def __init__(self, df_list, col_list):
        """
        Initialize an EmpiricalDataExporter object.

        Parameters:
        - df_list: List of DataFrames to analyze and export.
        - col_list: List of column names for analysis of ratios.
        """
        self.df_list = df_list
        self.col_list = col_list

    def export_empirical_data(self, file_prefix):
        """
        Export empirical data to CSV files for each combination of DataFrame and columns.

        Parameters:
        - file_prefix: Prefix for the output file names.
        """
        for index, df in enumerate(self.df_list):
            combined_emperical_df = pd.DataFrame()

            for col in self.col_list:
                combined_row_emperical_df = pd.DataFrame()

                for row in df.columns:
                    generator = ValueRatioTableGenerator(df, col=col, row=row)
                    ratio_table = generator.generate_ratio_table()
                    t_table = generator.perform_t_test

                    emperical_df = pd.concat([ratio_table, t_table], axis=1)

                    combined_row_emperical_df = pd.concat([combined_row_emperical_df, emperical_df], ignore_index=False)

                combined_emperical_df = pd.concat([combined_emperical_df, combined_row_emperical_df], axis=1,
                                                  ignore_index=False)

            output_filename = f'{file_prefix}_Matching_{index+1}.csv'
            combined_emperical_df.to_csv(output_filename, index=True)


