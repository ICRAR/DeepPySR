import pandas as pd
import os

def load_diabetes_brfss_data(file_path=None):
    if file_path is None:
        # Default path relative to project root
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../test_data/Health/diabetes_012_health_indicators_BRFSS2015.csv'))
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Target is Diabetes_012 (0 = no diabetes, 1 = prediabetes, 2 = diabetes)
    y = df['Diabetes_012']
    X = df.drop(columns=['Diabetes_012'])
    
    return X, y
