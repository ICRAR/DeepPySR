import numpy as np
from DeepPySR.regressor import DeepPySRRegressor
import os,torch
import time
import kan as KAN
from sklearn.metrics import r2_score

def generate_l3_v11(n_samples: int = 1000, n_vars: int = 11, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n_samples, n_vars))

    # Construct some hidden symbolic relationships among variables
    # Layer 3 (can be enabled if n_layers >= 3 is used)
    X[:,4] = np.sin(X[:,7]+X[:,8])
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

def run_pysr():
    # 1. Prepare data
    print("Generating synthetic data...")
    X, y = generate_l3_v15_easy()

    # 2. Initialize the DeepPySRRegressor
    # It now inherits from PySRRegressor, so you can use any PySR parameters!
    # Default operators are now:
    # binary: ["+", "-", "*", "/"]
    # unary: ["sin", "cos", "exp", "log", "sqrt", "tanh", "square", "asin", "acos", "atanh"]
    pysr_kwargs = {
        "select_k_features": None,
        "niterations": 100,
        "population_size": 50,
        "model_selection": "best",
        "early_stop_condition": "f(loss, complexity) = (loss < 0.0001) && (complexity < 10)",
        "verbosity": 0,
        "denoise": True,
        "procs": os.cpu_count()-1
    }
    regressor = DeepPySRRegressor(
        max_layers=4,           # DeepPySR specific: Depth of the symbolic hierarchy
        output_dir="results/l3_v15_test",
        stopping_score=0.1,     # DeepPySR specific: Stop recursion if loss is below this
        **pysr_kwargs
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

    y_pred = regressor.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"PySR R2 score: {r2:.2f}")

def run_kan(width=[15,5,1]):
    X, y = generate_l3_v15_easy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = {}
    dataset["train_input"] = (
        torch.from_numpy(X).float().to(device)
    )
    dataset["test_input"] = (
        torch.from_numpy(X).float().to(device)
    )
    dataset["train_label"] = (
        torch.from_numpy(y).float().to(device)
    )
    dataset["test_label"] = (
        torch.from_numpy(y).float().to(device)
    )
    kan_dir=f"./results/l3_v15_kan{''.join([str(i) for i in width])}"
    model = KAN.KAN(
        width=width,
        # grid=grid,
        k=3,
        seed=0,
        # scale_base_mu=scale_base_mu,
        # scale_base_sigma=scale_base_sigma,
        ckpt_path=kan_dir,
        device=device,
    ).to(device)
    model.fit(
        dataset,
        # save_fig=True,
        # opt=opt,
        # steps=step,
        # lamb=lamb,
        # lamb_entropy=lamb_entropy,
        # lamb_l1=lamb_l1,
        # update_grid=False,
        # patience = 500
        # lr=lr,
    )
    # model.prune()
    model(dataset['train_input'])
    model.plot(folder=kan_dir)
    print(f"KAN plot saved to {kan_dir}/kan_plot.png")
    model.auto_symbolic()
    formula = model.symbolic_formula()[0][0]
    print('formula',formula)

    y_pred = model(dataset["test_input"]).detach().cpu().numpy().reshape(-1, 1)
    r2 = r2_score(dataset["test_label"].to('cpu'), y_pred)
    print(f"KAN R2 score: {r2:.2f}")

if __name__ == "__main__":
    run_pysr() # r2 1 for var15, 0.47 for var11
    # run_kan(width=[15,[5,1],1]) # r2 0 formula 0.0166442945192102
    # run_kan(width=[15,5,2,1])