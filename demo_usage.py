import numpy as np
from DeepPySR.DeepPySR.regressor import DeepPySRRegressor
import os
import time

def generate_l3_v11(n_samples: int = 1000, n_vars: int = 11, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, n_vars))

    # Construct some hidden symbolic relationships among variables
    # Layer 3 (can be enabled if n_layers >= 3 is used)
    X[:,4] = np.tan(X[:,7]+X[:,8])
    X[:,6] = X[:,8] * X[:,9]

    # Layer 2
    X[:,1] = np.exp(X[:,4]+X[:,5]) * 2
    X[:,2] = np.cos(X[:,3]+X[:,6])

    # Layer 1
    y = X[:,0] * np.sin(X[:,1]+3) + X[:,2] * X[:,3]
    return X, y

def generate_l3_v15_easy(n_samples: int = 1000, n_vars: int = 15, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_samples, n_vars))

    # Construct some hidden symbolic relationships among variables
    # Layer 3 (can be enabled if n_layers >= 3 is used)
    X[:,5] = np.tan(X[:,10]+X[:,11])
    X[:,7] = -5*X[:,12] + 8

    # Layer 2
    X[:,0] = X[:,4] + X[:,5]
    X[:,1] = np.exp(X[:,6]+2)
    X[:,2] = X[:,7]*X[:,8]*np.exp(X[:,9])

    # Layer 1
    y = X[:,0] * np.sin(X[:,1]) + X[:,2] * X[:,3]
    return X, y

def main():
    # 1. Prepare data
    print("Generating synthetic data...")
    X, y = generate_l3_v15_easy()

    # 2. Initialize the DeepPySRRegressor
    # It now inherits from PySRRegressor, so you can use any PySR parameters!
    # Default operators are now:
    # binary: ["+", "-", "*", "/"]
    # unary: ["sin", "cos", "exp", "log", "sqrt", "tanh", "square", "asin", "acos", "atanh"]
    regressor = DeepPySRRegressor(
        max_layers=4,           # DeepPySR specific: Depth of the symbolic hierarchy
        output_dir="l3_v15_easy",
        stopping_score=0.1,     # DeepPySR specific: Stop recursion if loss is below this
        
        # PySR parameters (inherited)
        select_k_features = None,
        niterations=50,
        population_size=500,
        model_selection="best",
        early_stop_condition="f(loss, complexity) = (loss < 0.0001) && (complexity < 10)",
        verbosity = 0,
        denoise=True
    )

    # 3. Fit the model
    # This will recursively find relationships: 
    # y = f(x_i) -> x_i = f(x_j) -> ...
    print("Fitting DeepPySRRegressor (this may take a few minutes)...")
    start_time = time.time()
    regressor.fit(X, y)
    duration = time.time() - start_time
    print(f"Fitting completed in {duration/60:.2f} minutes.")

    # 4. Access discovered relationships
    print("\nDiscovered Relationships:")
    for rel in regressor.relationships_:
        print(f"Layer {rel['layer']} | {rel['target']} = {rel['formula']} (Score: {rel['score']:.2f})")

    # 5. Visualize the hierarchy
    # This creates a PNG file with the graph and a summary table
    plot_path = "symbolic_hierarchy_demo.png"
    regressor.plot(plot_path)
    regressor.plot_circle(filename="circle_demo.png")
    print(f"\nDemo completed! Check '{os.path.join('demo_outputs', plot_path)}' for the visualization.")

if __name__ == "__main__":
    main()
