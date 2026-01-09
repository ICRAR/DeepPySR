import numpy as np
from DeepPySR.DeepPySR.regressor import DeepPySRRegressor
import os
import time

def generate_data(n_samples: int = 1000, n_vars: int = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_vars))

    # Construct some hidden symbolic relationships among variables
    # Layer 3 (can be enabled if n_layers >= 3 is used)
    X[:,4] = np.cos(X[:,8]+X[:,7])
    X[:,5] = X[:,8] + X[:,9]

    # Layer 2
    X[:,1] = X[:,4] + np.sin(X[:,5]/(X[:,6]+1))
    X[:,2] = X[:,5] * np.exp(X[:,7]-1)

    # Layer 1
    y = X[:,0] * np.exp(X[:,1]-1) + X[:,2]*np.sin(X[:,3])
    return X, y

def main():
    # 1. Prepare data
    print("Generating synthetic data...")
    X, y = generate_data()

    # 2. Initialize the DeepPySRRegressor
    # It now inherits from PySRRegressor, so you can use any PySR parameters!
    # Default operators are now:
    # binary: ["+", "-", "*", "/"]
    # unary: ["sin", "cos", "exp", "log", "sqrt", "tanh", "square", "asin", "acos", "atanh"]
    regressor = DeepPySRRegressor(
        max_layers=4,           # DeepPySR specific: Depth of the symbolic hierarchy
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
    start_time = time.time()
    regressor.fit(X, y)
    duration = time.time() - start_time
    print(f"Fitting completed in {duration//60:.2f} minutes.")

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
