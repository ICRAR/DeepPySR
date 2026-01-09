import numpy as np
from deepPySR.deepPySR.regressor import DeepPySRRegressor
import os

def generate_sample_data(n_samples=500, n_vars=5, seed=42):
    """
    Generates synthetic data with hidden hierarchical relationships.
    y = x0 * exp(x1 - 1)
    x1 = x2 + sin(x3)
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.1, 1.0, size=(n_samples, n_vars))
    
    # Hidden relationship Layer 2
    X[:, 1] = X[:, 2] + np.sin(X[:, 3])
    
    # Hidden relationship Layer 1
    y = X[:, 0] * np.exp(X[:, 1] - 1.0)
    
    return X, y

def main():
    # 1. Prepare data
    print("Generating synthetic data...")
    X, y = generate_sample_data()

    # 2. Initialize the DeepPySRRegressor
    # It now inherits from PySRRegressor, so you can use any PySR parameters!
    # Default operators are now:
    # binary: ["+", "-", "*", "/"]
    # unary: ["sin", "cos", "exp", "log", "sqrt", "tanh", "square", "asin", "acos", "atanh"]
    regressor = DeepPySRRegressor(
        max_layers=2,           # DeepPySR specific: Depth of the symbolic hierarchy
        score_threshold=2.0,    # DeepPySR specific: Minimum score to continue exploring
        output_dir="demo_outputs",
        
        # PySR parameters (inherited)
        niterations=30,
        population_size=500,
        model_selection="best",
        random_state=42,
        progress=True # Main progress bar
    )

    # 3. Fit the model
    # This will recursively find relationships: 
    # y = f(x_i) -> x_i = f(x_j) -> ...
    print("Fitting DeepPySRRegressor (this may take a few minutes)...")
    regressor.fit(X, y)

    # 4. Access discovered relationships
    print("\nDiscovered Relationships:")
    for rel in regressor.relationships_:
        print(f"Layer {rel['layer']} | {rel['target']} = {rel['formula']} (Score: {rel['score']:.2f})")

    # 5. Visualize the hierarchy
    # This creates a PNG file with the graph and a summary table
    plot_path = "symbolic_hierarchy_demo.png"
    regressor.plot(plot_path)
    
    print(f"\nDemo completed! Check '{os.path.join('demo_outputs', plot_path)}' for the visualization.")

if __name__ == "__main__":
    main()
