import pandas as pd
import os

def aggregate_bmi_data():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../test_data/Health/bmi'))
    ages = [8, 10, 13, 16, 20, 23, 26]
    age_mapping = {13: 14, 16: 17, 26: 27}
    all_data = []

    for age in ages:
        file_path = os.path.join(data_dir, f"rawdata_yr{age}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            target_col = f"y{age}bmi"
            if target_col in df.columns:
                df = df.rename(columns={target_col: "target_bmi"})
            exact_age = age_mapping.get(age, age)
            df = df.copy()
            df['age'] = exact_age
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()

    aggregated_df = pd.concat(all_data, ignore_index=True)
    return aggregated_df

def clean_bmi_data(df, threshold=0.1):
    if df.empty:
        return df
    cols_to_drop = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    cols_to_drop = cols_to_drop + ['mother_id','preg_no','cohab_0']
    df_cleaned = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    nan_rates = df_cleaned.isnull().mean()
    cols_to_drop_nan = nan_rates[nan_rates > threshold].index.tolist()
    df_cleaned = df_cleaned.drop(columns=cols_to_drop_nan)
    df_cleaned = df_cleaned.dropna()
        
    return df_cleaned

def load_bmi_agg_data(age=None):
    df = aggregate_bmi_data()
    df = clean_bmi_data(df)
    if age is not None:
        df = df[df['age'] == age]

    ids = df["child_id"].values
    X = df.drop(columns=['child_id','target_bmi'])
    y = df['target_bmi'].values
    return ids, X, y
