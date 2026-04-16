import os
import pandas as pd


def load_heart_cleveland_data(file_path=None, binary=True):
    if file_path is None:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../test_data/Health/heart+disease/processed.cleveland.data'))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Heart Cleveland dataset file not found at {file_path}")

    cols = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]

    df = pd.read_csv(file_path, header=None, names=cols)
    df = df.replace('?', pd.NA)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    y = df['num'].astype(int)
    if binary:
        y = (y > 0).astype(int)

    X = df.drop(columns=['num'])
    return X, y
