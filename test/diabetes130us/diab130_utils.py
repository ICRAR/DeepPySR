import pandas as pd
import numpy as np
import os

def analyze_missingness(df, label="Missingness Analysis"):
    # Calculate missing values
    missing_count = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # Combine into a summary dataframe
    missing_summary = pd.DataFrame({
        'Missing Values': missing_count,
        'Percentage (%)': missing_percentage
    })

    # Sort by missing values descending
    missing_summary = missing_summary.sort_values(by='Missing Values', ascending=False)

    # Print variables with missing values
    print(f"{label} (top 15 variables):")
    print(missing_summary.head(15))

    print("\nTotal rows:", len(df))

def load_and_clean_data(threshold=10.0):
    # Load the data
    file_path = os.path.join(os.path.dirname(__file__), '../../test_data/Health/diabetes130us/diabetes130us.csv')
    df = pd.read_csv(file_path, na_values=['?', 'None','Unknown/Invalid'], low_memory=False)

    # Calculate missing percentage
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # Identify columns to drop
    cols_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
    print(f"\nDropping columns with > {threshold}% missingness: {cols_to_drop}")

    # Drop the columns
    df_cleaned = df.drop(columns=cols_to_drop)

    # Remove data entries with missing values in gender, diag_1, diag_2, or diag_3
    cols_to_check = ['gender', 'diag_1', 'diag_2', 'diag_3']
    # Only check columns that still exist in the dataframe
    cols_to_check = [col for col in cols_to_check if col in df_cleaned.columns]
    
    print(f"\nRemoving rows with missing values in: {cols_to_check}")
    df_cleaned = df_cleaned.dropna(subset=cols_to_check)

    # Impute missing values in the 'race' column using the mode
    if 'race' in df_cleaned.columns and df_cleaned['race'].isnull().any():
        race_mode = df_cleaned['race'].mode()[0]
        print(f"\nImputing missing values in 'race' with mode: {race_mode}")
        df_cleaned['race'] = df_cleaned['race'].fillna(race_mode)

    print(f"Original shape: {df.shape}")
    print(f"Cleaned shape: {df_cleaned.shape}")

    df_cleaned = preprocess_diabetes_data(df_cleaned)
    # Show missingness after cleaning
    analyze_missingness(df_cleaned, label="Missingness Analysis After Processing")

    # Show rows with NaN values
    rows_with_nan = df_cleaned[df_cleaned.isnull().any(axis=1)]
    print(f"\nRows with any NaN values after processing ({len(rows_with_nan)} rows):")
    if not rows_with_nan.empty:
        print("patient_nbr, column_name_with_nan")
        for idx, row in rows_with_nan.iterrows():
            cols_with_nan = row.index[row.isnull()].tolist()
            for col in cols_with_nan:
                print(f"{row['patient_nbr']}, {col}")
    else:
        print("No rows with NaN values found.")

    print("\nSample of processed data (first 10 rows, selected columns):")
    return df_cleaned['encounter_id'], df_cleaned.drop(columns = ['encounter_id', 'patient_nbr', 'readmitted']), df_cleaned['readmitted']


def preprocess_diabetes_data(df):
    """
    Apply specific preprocessing to the cleaned diabetes dataset.
    """
    df_proc = df.copy()

    # for race; one hot code
    if 'race' in df_proc.columns:
        df_proc = pd.get_dummies(df_proc, columns=['race'], prefix='race')

    # in gender, treat female as 0 and male as 1
    if 'gender' in df_proc.columns:
        # Some rows might have 'Unknown/Invalid' if not handled by load_and_clean_data
        # But load_and_clean_data handles 'Unknown/Invalid' as NaN and drops them.
        gender_map = {'Female': 0, 'Male': 1}
        df_proc['gender'] = df_proc['gender'].map(gender_map)

    # bin the age as it shows now
    if 'age' in df_proc.columns:
        # Age is like [0-10), [10-20), etc. 
        # The request "bin the age as it shows now" usually means converting to numeric codes.
        age_map = {
            '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
            '[50-60)': 5, '[60-70)': 6, '[70-70)': 7, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
        }
        df_proc['age'] = df_proc['age'].map(age_map)

    # one hot code for admission_type_id
    if 'admission_type_id' in df_proc.columns:
        df_proc = pd.get_dummies(df_proc, columns=['admission_type_id'], prefix='admission_type')

    # don't floor the diag_1, diag_2, and diag_3
    # for these 3 variables, any value starts with a 'E' replace it to be 10, 
    # and replace V with 11
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df_proc.columns:
            def get_diag_group(val):
                if pd.isnull(val): return val
                val = str(val)
                if val.startswith('E'):
                    return float(val.replace('E', ''))
                if val.startswith('V'):
                    return float(val.replace('V', '10', 1))
                try:
                    return float(val)
                except ValueError:
                    return 0.0 # Return a default numeric value for any other strings
            df_proc[col] = df_proc[col].apply(get_diag_group)

    # medication mapping
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 
        'glipizide-metformin', 'glimepiride-pioglitazone', 
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    med_map = {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}
    for col in med_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].map(med_map)

    # change, 0 for No, 1 for Ch
    if 'change' in df_proc.columns:
        df_proc['change'] = df_proc['change'].map({'No': 0, 'Ch': 1})

    # diabetesMed, use 0 for No, and 1 for Yes
    if 'diabetesMed' in df_proc.columns:
        df_proc['diabetesMed'] = df_proc['diabetesMed'].map({'No': 0, 'Yes': 1})

    # readmitted, use 1 for <30, and 0 for >30 and NO
    if 'readmitted' in df_proc.columns:
        df_proc['readmitted'] = df_proc['readmitted'].map({'NO': 0, '<30': 1, '>30': 0})

    return df_proc


if __name__ == "__main__":
    file_path = '/home/00101787/Projects/DeepPySR/test_data/Health/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv'
    # !Clean data
    df_cleaned = load_and_clean_data(file_path)
    
