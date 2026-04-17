import numpy as np
import pandas as pd
import os

# Selected Feynman equations for testing
equations = {
    'I.6.2a': {
        'formula': 'np.exp(-theta**2/2)/np.sqrt(2*np.pi)',
        'vars': {'theta': (1, 3)}
    },
    'I.8.14': {
        'formula': 'np.sqrt((x2-x1)**2 + (y2-y1)**2)',
        'vars': {'x1': (1, 5), 'x2': (1, 5), 'y1': (1, 5), 'y2': (1, 5)}
    },
    'I.13.4': {
        'formula': '0.5 * m * (v**2 + u**2 + w**2)',
        'vars': {'m': (1, 5), 'v': (1, 5), 'u': (1, 5), 'w': (1, 5)}
    },
    'I.9.18': {
        'formula': 'G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)',
        'vars': {'m1': (1, 2), 'm2': (1, 2), 'G': (1, 2), 'x1': (3, 4), 'x2': (1, 2), 'y1': (3, 4), 'y2': (1, 2), 'z1': (3, 4), 'z2': (1, 2)}
    },
    'I.32.17': {
        'formula': '(1/2*epsilon*c*Ef**2)*(8*np.pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)',
        'vars': {'epsilon': (1, 2), 'c': (1, 2), 'Ef': (1, 2), 'r': (1, 2), 'omega': (1, 2), 'omega_0': (3, 5)}
    },
}

def generate_feynman_data(eq_name, n_samples=1000, random_state=42):
    """
    Generate synthetic data for a Feynman equation.
    
    Parameters:
    - eq_name: Name of the equation (key in equations dict)
    - n_samples: Number of samples to generate
    - random_state: Random seed for reproducibility
    
    Returns:
    - X: DataFrame of input variables
    - y: Array of output values
    """
    if eq_name not in equations:
        raise ValueError(f"Equation {eq_name} not found in equations dict")
    
    np.random.seed(random_state)
    
    eq = equations[eq_name]
    formula = eq['formula']
    vars_dict = eq['vars']
    
    # Generate random samples for each variable
    data = {}
    for var, (low, high) in vars_dict.items():
        data[var] = np.random.uniform(low, high, n_samples)
    
    X = pd.DataFrame(data)
    
    # Compute output using the formula
    y = []
    for i in range(n_samples):
        env = {var: X.iloc[i][var] for var in data}
        try:
            y_val = eval(formula, {"__builtins__": None, "np": np}, env)
            y.append(y_val)
        except Exception as e:
            print(f"Error evaluating formula for sample {i}: {e}")
            y.append(np.nan)
    
    y = np.array(y)
    
    # Remove any NaN values
    valid_idx = ~np.isnan(y)
    X = X[valid_idx].reset_index(drop=True)
    y = y[valid_idx]
    
    return X, y

def load_feynman_data(eq_name, n_samples=1000):
    """
    Load or generate Feynman data for testing.
    """
    return generate_feynman_data(eq_name, n_samples)

if __name__ == "__main__":
    # Test data generation
    for eq_name in equations.keys():
        print(f"Generating data for {eq_name}...")
        X, y = generate_feynman_data(eq_name, n_samples=100)
        print(f"  Shape: X={X.shape}, y={y.shape}")
        print(f"  X columns: {list(X.columns)}")
        print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
        print()