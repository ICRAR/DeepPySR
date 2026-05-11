import numpy as np
import pandas as pd
import os

# Physical constants
G = 6.67430e-11
c = 299792458

# Selected Feynman equations for testing
equations = {
    'I.6.2a': {
        'formula': 'exp(-theta**2/2)/sqrt(2*pi)',
        'vars': ['theta'],
        'output': 'f'
    },
    'I.8.14': {
        'formula': 'sqrt((x2-x1)**2+(y2-y1)**2)',
        'vars': ['x1', 'x2', 'y1', 'y2'],
        'output': 'd'
    },
    'I.13.4': {
        'formula': '1/2*m*(v**2+u**2+w**2)',
        'vars': ['m', 'v', 'u', 'w'],
        'output': 'K'
    },
    'I.9.18': {
        'formula': 'G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)',
        'vars': ['m1', 'm2', 'G', 'x1', 'x2', 'y1', 'y2', 'z1', 'z2'],
        'output': 'F'
    },
    'I.32.17': {
        'formula': '(1/2*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)',
        'vars': ['epsilon', 'c', 'Ef', 'r', 'omega', 'omega_0'],
        'output': 'Pwr'
    },
}

def load_feynman_data(eq_name, n_samples=1000):
    """
    Load Feynman data from file for testing.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, 'test_data', 'Feynman', 'Feynman_with_units', eq_name)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file for {eq_name} not found at {data_path}")
        
    # Read the data, handling potential double spaces or leading/trailing whitespace
    data = pd.read_csv(data_path, sep=r'\s+', header=None, engine='python')
    
    # Take up to n_samples
    if n_samples and len(data) > n_samples:
        data = data.sample(n_samples, random_state=42).reset_index(drop=True)
    
    X = data.iloc[:, :-1]
    y_raw = data.iloc[:, -1].values.astype(float)

    if eq_name in equations:
        X.columns = equations[eq_name]['vars']
        y = pd.Series(y_raw, name=equations[eq_name]['output'])
    else:
        X.columns = [f'x{i}' for i in range(X.shape[1])]
        y = pd.Series(y_raw, name='y')
        
    return X, y

if __name__ == "__main__":
    # Test data loading
    for eq_name in equations.keys():
        print(f"Loading data for {eq_name}...")
        try:
            X, y = load_feynman_data(eq_name, n_samples=100)
            print(f"  Shape: X={X.shape}, y={y.shape}")
            print(f"  X columns: {list(X.columns)}")
            print(f"  y name: {y.name}")
            print(f"  y range: [{y.min():.3e}, {y.max():.3e}]")
        except Exception as e:
            print(f"  Error loading {eq_name}: {e}")
        print()
