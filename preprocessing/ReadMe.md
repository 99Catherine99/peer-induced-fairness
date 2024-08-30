# Preprocessing

This section covers the preprocessing steps applied to the SMEs dataset, where different questionnaires treat features differently. The preprocessing ensures that the dataset is standardized, clean, and ready for analysis.

## Files and Their Functions

### data preprocess

It is designed to handle the comprehensive preprocessing of the dataset. Key tasks include:

- **Standardization**: Ensuring that feature formats are consistent across the dataset.
- **Variable Selection**: Identifying and selecting relevant variables for analysis.
- **Missing Value Handling**: Deleting features with high missing ratio. Addressing missing data points using appropriate imputation methods.
- **Feature and Label Merging**: Combining features and labels into a unified dataset.
- **Dataset Splitting**: Separating the dataset into datasets with labels and without labels.
- **Exporting**: Exporting the preprocessed dataset in CSV format for easy access in subsequent steps.



> **Output Files**: 
> - `int_merge_csv.csv` / `int_merge_xlsx.xlsx`: Preprocessed data without labels.
> - `merge_csv.csv` / `merge_xlsx.xlsx`: Preprocessed data with labels.

### Encoding

It manages the encoding of features, ensuring that the data is prepared for machine learning models:

- **Label Encoding**: Applied to ordered variables to maintain the natural order in categorical data.
- **One-Hot Encoding**: Used for unordered variables to accurately represent categorical data without implying any order.

### Imputer

It includes methods for handling missing values in the dataset:

- **Missing Value Strategy**: In this preprocessing step, we use `strategy=frequent`, which fills in missing values with the most frequent value in the feature.

### Undersampling

It is used to process data by resampling it to achieve different imbalance levels, which will be used in the following imbalance folder.

---

By following the preprocessing steps outlined in these scripts, the SMEs dataset is transformed into a standardized and ready-to-use format, facilitating accurate and efficient analysis in the subsequent stages.
