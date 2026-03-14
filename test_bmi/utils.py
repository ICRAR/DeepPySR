import pandas as pd
import os

def load_data(path: str, year: int = 8):
    data = pd.read_csv(path)
    dataid = data[["child_id"]].T.drop_duplicates().T.values.reshape(1, -1)[0]
    data.columns = data.columns.str.replace(',','_')
    params = {}
    if 'occupcode_m_0' in data.columns:
        data['occupcode_m_0'] = data['occupcode_m_0']+1
        params['occupcode_m_0'] = 10
    if 'occupcode_f1_1' in data.columns:
        data['occupcode_f1_1'] = data['occupcode_f1_1']+1
        params['occupcode_f1_1'] = 10
    if 'hhincome_0' in data.columns:
        params['hhincome_0'] = 4
    if 'prepreg_smk' in data.columns:
        data['prepreg_smk'] = data['prepreg_smk']+1
        params['prepreg_smk'] = 2
    if 'prepreg_cig' in data.columns:
        data['prepreg_cig'] = data['prepreg_cig']+1
        params['prepreg_cig'] = 3

    datain = data.drop(columns=['child_id',f'y{year}bmi',f'pred_y{year}bmi'])
    dataout = data[[f'y{year}bmi']]

    return dataid,datain,dataout

def aggregate_data():
    data_dir = "/home/00101787/Projects/pgs/data/data_all/my1y5data/"
    ages = [8, 10, 13, 16, 20, 23, 26]
    age_mapping = {13: 14, 16: 17, 26: 27}
    all_data = []

    for age in ages:
        file_path = os.path.join(data_dir, f"rawdata_yr{age}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Rename y{age}bmi to target_bmi
            target_col = f"y{age}bmi"
            if target_col in df.columns:
                df = df.rename(columns={target_col: "target_bmi"})
            
            # Add age column with specified exact age value
            exact_age = age_mapping.get(age, age)
            df['age'] = exact_age
            
            all_data.append(df)
        else:
            print(f"Warning: File {file_path} not found.")

    if not all_data:
        return pd.DataFrame()

    # Aggregate all data
    aggregated_df = pd.concat(all_data, ignore_index=True)
    return aggregated_df

def clean_data(threshold: float = 0.1):
    df = aggregate_data()

    # Get columns that have only one unique value (excluding NaNs)
    cols_to_drop = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    cols_to_drop = cols_to_drop + ['mother_id','preg_no','cohab_0',]
    df_cleaned = df.drop(columns=cols_to_drop)
    
    # Drop columns with more than 'threshold' fraction of missing values
    # to maximize sample size while keeping the most informative variables.
    nan_rates = df_cleaned.isnull().mean()
    cols_to_drop_nan = nan_rates[nan_rates > threshold].index.tolist()
    #     {'threshold': 0.0, 'n_cols': 61, 'n_rows': 5867}
    # {'threshold': 0.05, 'n_cols': 116, 'n_rows': 4062}
    # {'threshold': 0.1, 'n_cols': 127, 'n_rows': 3235} choose this
    # {'threshold': 0.15, 'n_cols': 133, 'n_rows': 1937}
    # {'threshold': 0.2, 'n_cols': 151, 'n_rows': 802}
    # {'threshold': 0.25, 'n_cols': 153, 'n_rows': 706}
    # {'threshold': 0.3, 'n_cols': 154, 'n_rows': 682}
    df_cleaned = df_cleaned.drop(columns=cols_to_drop_nan)
    
    # Drop rows with any remaining missing values
    df_cleaned = df_cleaned.dropna()
        
    return df_cleaned

def load_agg_data(age=None):
    data = clean_data()
    if age is not None:
        data = data[data['age'] == age]

    dataid = data[["child_id"]].T.drop_duplicates().T.values.reshape(1, -1)[0]
    datain = data.drop(columns=['child_id','target_bmi'])
    dataout = data[['target_bmi']]
    return dataid,datain,dataout

data = clean_data()