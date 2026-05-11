import numpy as np
import pandas as pd
from deeppysr import DeepPySR
import os

def test_deeppysr_constant_multilayer():
    # 1. Generate synthetic data
    # Equations:
    # x2 = 4 * (x1 + x4)
    # y = x1 + c * x2 * x3
    # c = 2.5 (constant value to fit)
    
    np.random.seed(42)
    n_samples = 500
    x1 = np.random.uniform(1, 5, n_samples)
    x3 = np.random.uniform(1, 5, n_samples)
    x4 = np.random.uniform(1, 5, n_samples)
    
    c_true = 299792458 # Speed of light in m/s
    x2 = 4 * (x1 + x4)
    y_true = x1 + c_true * x2 * x3
    
    # Add some small noise
    # Since c_true is very large, noise should be relative or we use smaller values for x
    # Let's keep x in [1, 5] but maybe y will be huge. 
    # That's fine for symbolic regression if it finds the constant.
    y = y_true + np.random.normal(0, 0.01, n_samples)
    
    # Input features for DeepPySR: x1, x3, x4
    # Note: x2 is an intermediate variable that DeepPySR should ideally discover or use
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4
    })
    
    print(f"Generated data with {n_samples} samples.")
    print(f"Target y range: [{y.min():.3e}, {y.max():.3e}]")

    # 2. Configure DeepPySR
    # We want 2 layers and to fit a constant 'c'
    output_dir = "test_output_constant"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model = DeepPySR(
        max_layers=2,
        extra_constants=['c'],
        output_dir=output_dir,
        niterations=100,
        populations=20,
        population_size=100,
        binary_operators=["+", "*", "/"],
        unary_operators=[],
        verbosity=1,
        pareto_r2_weight=[1.0, 10.0],
        pareto_lambda=[0.001, 0.01]
    )

    # 3. Fit the model
    print("Fitting DeepPySR model (2 layers)...")
    model.fit(X, y)

    # 4. Verify results
    print("\nFitting completed.")
    print("Relationships found:")
    try:
        rel = model._get_mapped_relationships()
        print(rel)
    except Exception as e:
        print(f"Could not get mapped relationships: {e}")

    print("\nSympy representation:")
    try:
        print(model.sympy())
    except Exception as e:
        print(f"Could not get sympy representation: {e}")

    # Check predictions
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred)**2)
    print(f"MSE: {mse:.6f}")

if __name__ == "__main__":
    test_deeppysr_constant_multilayer()
