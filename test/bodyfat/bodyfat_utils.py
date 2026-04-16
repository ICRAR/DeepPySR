import os
import pandas as pd


def load_bodyfat_data(file_path=None):
    if file_path is None:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../test_data/Health/bodyfat.csv'))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"BodyFat data file not found at {file_path}")

    df = pd.read_csv(file_path)
    if 'BodyFat' not in df.columns:
        raise ValueError("Expected a 'BodyFat' target column in the dataset")

    X = df.drop(columns=['BodyFat', 'Density'])
    y = df['BodyFat']
    return X, y
