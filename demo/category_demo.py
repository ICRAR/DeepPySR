import numpy as np
import pandas as pd
import time, torch
import kan as KAN
from pysr import TemplateExpressionSpec
from DeepPySR.regressor import DeepPySRRegressor
from sklearn.metrics import r2_score

# 1. Generate Synthetic Data
# Create data with 2 features and 3 categories
X = np.random.uniform(-3, 3, (1000, 2))
category = np.random.randint(0, 3, 1000)
y = np.zeros(category.shape[0])
for i in range(category.shape[0]):
    if category[i] == 0:
        y[i] = np.sin(X[i,0])
    elif category[i] == 1:
        y[i] = X[i,0]**2 - 1
    else:
        y[i] = np.sin(X[i,0]-1)


category_p_one = category + 1
X_with_category = np.column_stack([X, category])
X_with_category = pd.DataFrame(X_with_category,columns=["x0","x1","c"])
def run_deeppysr():
    # Create feature DataFrame (excluding the category column)
    template = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["x0", "x1", "c"],
        parameters={"p": 3},  # One parameter per category
        combine="f(x0, x1, p[c])"
    )

    regressor = DeepPySRRegressor(
        # max_layers=2,           # DeepPySR specific: Depth of the symbolic hierarchy
        output_dir="./results/category",
        stopping_score=2,     # DeepPySR specific: Stop recursion if loss is below this

        # PySR parameters (inherited)
        unary_operators=["sin","asin"],
        expression_spec=template,
        select_k_features = None,
        # niterations=50,
        # population_size=500,
        maxsize=10,
        model_selection="best",
        # elementwise_loss="L2DistLoss()",
        early_stop_condition="f(loss, complexity) = (loss < 0.0001) && (complexity < 20)",
        verbosity = 0,
        denoise=True
    )

    # 3. Fit the model
    # This will recursively find relationships:
    # y = f(x_i) -> x_i = f(x_j) -> ...
    print("Fitting DeepPySRRegressor (this may take a few minutes)...")
    start_time = time.time()
    regressor.fit(X_with_category, y)
    duration = time.time() - start_time
    print(f"Fitting completed in {duration/60:.2f} minutes.")
    y_pred = regressor.predict(X_with_category)
    # 4. Access discovered relationships
    print("\nDiscovered Relationships:")
    for rel in regressor.relationships_:
        print(f"Layer {rel['layer']} | {rel['target']} = {rel['formula']} (Score: {rel['score']:.2f})")

    # 5. Visualize the hierarchy
    # This creates a PNG file with the graph and a summary table
    plot_path = "symbolic_hierarchy_demo.png"
    regressor.plot(plot_path)
    regressor.plot_circle(filename="circle_demo.png")
    # print(f"\nDemo completed! Check '{os.path.join('demo_outputs', plot_path)}' for the visualization.")

    r2 = r2_score(y, y_pred)
    print(f"PySR R2 score: {r2:.2f}")

def run_kan(width=[3,1,1]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = {}
    dataset["train_input"] = (
        torch.from_numpy(X_with_category.values).float().to(device)
    )
    dataset["test_input"] = (
        torch.from_numpy(X_with_category.values).float().to(device)
    )
    dataset["train_label"] = (
        torch.from_numpy(y).float().to(device)
    )
    dataset["test_label"] = (
        torch.from_numpy(y).float().to(device)
    )
    kan_dir=f"./results/category_kan{''.join([str(i) for i in width])}"
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
    model.plot(folder=kan_dir,in_vars=["x1", "x2", "category"])

    model.auto_symbolic()
    formula = model.symbolic_formula(var=["x1", "x2", "category"])[0][0]
    print('formula',formula)

    y_pred = model(dataset["test_input"]).detach().cpu().numpy().reshape(-1, 1)
    r2 = r2_score(dataset["test_label"].to('cpu'), y_pred)
    print(f"KAN R2 score: {r2:.2f}")

run_deeppysr() #r2 1 formula y = (x0*x0 - 0.12)*sin(1.57*x2) + 1.0000013*sin(x0 + x2*(-1.0*x0 - 0.07) - 1.0*x2)
# run_kan(width=[3,1,1]) #r2 0 formula 0.707158565521240
# run_kan(width=[3,2,1]) #r2 0 formula 0.665530741214752
# run_kan(width=[3,2,1,1]) #r2 0 formula 0.612724974751472