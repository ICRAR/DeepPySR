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
    'I.14.3': {
        'formula': 'm * g * z',
        'vars': {'m': (1, 5), 'g': (1, 5), 'z': (1, 5)}
    },
    'I.25.13': {
        'formula': 'q / C',
        'vars': {'q': (1, 5), 'C': (1, 5)}
    }
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