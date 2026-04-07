import pandas as pd
import numpy as np
import os

def load_wine_data(wine_type='red'):
    """
    Load and preprocess the wine quality dataset.
    wine_type: 'red' or 'white'
    """
    if wine_type == 'red':
        file_path = '/home/00101787/Projects/DeepPySR/test_data/Wine/wine+quality/winequality-red.csv'
    elif wine_type == 'white':
        file_path = '/home/00101787/Projects/DeepPySR/test_data/Wine/wine+quality/winequality-white.csv'
    else:
        raise ValueError("wine_type must be 'red' or 'white'")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # The dataset uses ';' as delimiter
    df = pd.read_csv(file_path, sep=';')
    
    # Cleaning: check for missing values (though UCI wine quality usually has none)
    if df.isnull().any().any():
        print(f"Warning: Missing values found in {wine_type} wine data. Dropping rows.")
        df = df.dropna()
        
    return df

if __name__ == "__main__":
    df_red = load_wine_data('red')
    print(f"Red wine shape: {df_red.shape}")
    print(df_red.head())
    
    df_white = load_wine_data('white')
    print(f"White wine shape: {df_white.shape}")
    print(df_white.head())
