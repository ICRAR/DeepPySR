import numpy as np
import sympy
import os
import time
import pandas as pd
from sklearn.metrics import r2_score
from DeepPySR.regressor import DeepPySRRegressor
from utils import load_agg_data

sympy_cond = lambda x, y: sympy.Piecewise((y, x > 0), (0, True))
# template = TemplateExpressionSpec(expressions=["f", "g", "h", "k"],combine="f + g + h + k")
pysr_kwargs = {
                "parallelism": "multithreading",
                # "select_k_features": 10,
                # "expression_spec": template,
                # "complexity_of_variables":0.01,
                "maxsize": 40,
                "binary_operators": ["+", "*", "/", "-","cond(x,y) = x > 0 ? y : y*0"],
                # "binary_operators": ["+", "*", "/", "-"],
                "extra_sympy_mappings":{'cond': sympy_cond},
                # "unary_operators": ["sin", "cos", "exp", "log","sqrt","abs"],
                "unary_operators": ["exp", "log"],
                "parsimony": 0.001,#0.001,
                # "niterations": 500,
                "populations": 20,#15,
                "population_size": 100,
                "ncycles_per_iteration": 200,
                "adaptive_parsimony_scaling": 50.0,#50.0,
                "variable_prune_start": 50,#40,50, # new defined
                "variable_prune_ramp": 150,#150,80 # new defined
                "variable_prune_max": 0.7,#0.6,0.7 # new defined
                # "model_selection":"accuracy", # score, best, accuracy
                # "early_stop_condition": "f(loss, complexity) = (loss < 0.00001) && (complexity < 20)",
                "verbosity":1,
                "denoise":True,
                "turbo": True,
                "procs": max(1, (os.cpu_count() or 2) - 1),
                # "loss_function":'''
                #             function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
                #             prediction, flag = eval_tree_array(tree, dataset.X, options)
                #             if !flag
                #             return L(Inf)
                #             end
                #             # log(cosh(x)) acts like x^2/2 for small x and |x| for large x
                #             return sum(log.(cosh.(prediction .- dataset.y))) / dataset.n
                #             end
                #             ''',
}

def run_deeppysr(X, y, outdir, r2w=1.0, l=1.0, pysr_overrides=None):

    # 2. Initialize the DeepPySRRegressor
    # Merge overrides with base kwargs if provided
    final_kwargs = dict(pysr_kwargs)
    if pysr_overrides:
        final_kwargs.update(pysr_overrides)

    deeppysr = DeepPySRRegressor(
        max_layers=1,           # DeepPySR specific: Depth of the symbolic hierarchy
        output_dir=outdir,
        pareto_lambda=l,
        pareto_r2_weight=r2w,
        stopping_score=0.001,     # DeepPySR specific: Stop recursion if loss is below this
        model_provider='pypysr',
        **final_kwargs,
    )

    # 3. Fit the model
    # This will recursively find relationships:
    # y = f(x_i) -> x_i = f(x_j) -> ...
    print(f"Fitting DeepPySRRegressor (this may take a few minutes)...")
    start_time = time.time()
    deeppysr.fit(X, y)
    duration = time.time() - start_time
    print(f"Fitting completed in {duration/60:.2f} minutes.")

    y_pred = deeppysr.predict(X)
    # Handle NaNs and Infs in predictions
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
    r2 = r2_score(y, y_pred)
    print(f"R2 score: {r2:.2f}")

    # 4. Access discovered relationships
    print("\nDiscovered Relationships:")
    for rel in deeppysr.relationships_:
        print(f"Layer {rel['layer']} | {rel['target']} = {rel['formula']} (Score: {rel['score']:.2f})")

    # 5. Visualize the hierarchy
    # This creates a PNG file with the graph and a summary table
    plot_path = "symbolic_hierarchy_demo.png"
    deeppysr.plot(plot_path)
    deeppysr.plot_circle(filename="circle_demo.png")
    print(f"\nDeeppysr run completed! Check output directory for visualizations.")

def main():
    is_longitudal = True

    # Define four argument configurations
    arg_configs = {
        # Standard SR (no pruning, no adaptive parsimony scaling)
        "stdsr": {
            "adaptive_parsimony_scaling": 0.0,
            "variable_prune_max": 0.0,
            "variable_prune_start": 0,
            "variable_prune_ramp": 0,
        },
        # SR with pruning only
        "srprn": {
            "adaptive_parsimony_scaling": 0.0,
            "variable_prune_start": 50,
            "variable_prune_ramp": 150,
            "variable_prune_max": 0.7,
        },
        # SR with parsimony scaling only
        "srpsm": {
            "adaptive_parsimony_scaling": 1040.0,
            "variable_prune_max": 0.0,
            "variable_prune_start": 0,
            "variable_prune_ramp": 0,
        },
        # Full SR (use current/default arguments)
        "fullsr": {
            "adaptive_parsimony_scaling": 1040.0,
            "variable_prune_start": 50,
            "variable_prune_ramp": 150,
            "variable_prune_max": 0.7,
        },
    }

    age = [8,10,14,17,20,23,27]
    # r2w = [1, 1.5, 2]
    # l = [0.001, 0.005, 0.01, 0.1]
    r2w = [1]
    l= [0.001]
    if is_longitudal:
        id, X, y = load_agg_data()
        for cfg_name, cfg_overrides in arg_configs.items():
            # Prepare naming values based on overrides (fallback to base defaults)
            parsimony = pysr_kwargs["parsimony"]
            population = pysr_kwargs["populations"]
            pop_size = pysr_kwargs["population_size"]
            parsimony_scaling = cfg_overrides.get("adaptive_parsimony_scaling", pysr_kwargs.get("adaptive_parsimony_scaling"))
            prune_start = cfg_overrides.get("variable_prune_start", pysr_kwargs.get("variable_prune_start"))
            prune_ramp = cfg_overrides.get("variable_prune_ramp", pysr_kwargs.get("variable_prune_ramp"))
            prune_max = cfg_overrides.get("variable_prune_max", pysr_kwargs.get("variable_prune_max"))

            for r2w_ in r2w:
                for l_ in l:
                    outdir = (f"./results_bmi/deeppysr_longitudinal/cfg{cfg_name}_par{parsimony}_pop{population}_popsz{pop_size}_"
                              f"scl{parsimony_scaling}_prnst{prune_start}_ramp{prune_ramp}_max{prune_max}_"
                              f"r2w{r2w_}_lambda{l_}")
                    csv_path = os.path.join(outdir, "relationships.csv")
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        if df["layer"].max() >= 2:
                            print(f"[{cfg_name}] Relationships already exist for r2w {r2w_}, lambda {l_}. Skipping.")
                            continue

                    print(f"[{cfg_name}] Running DeepPySRRegressor for r2w {r2w_}, lambda {l_}.")
                    run_deeppysr(X, y, outdir, r2w=r2w_, l=l_, pysr_overrides=cfg_overrides)
    else:
        for age_ in age:
            id, X, y = load_agg_data(age=age_)
            for cfg_name, cfg_overrides in arg_configs.items():
                parsimony = pysr_kwargs["parsimony"]
                population = pysr_kwargs["populations"]
                pop_size = pysr_kwargs["population_size"]
                parsimony_scaling = cfg_overrides.get("adaptive_parsimony_scaling", pysr_kwargs.get("adaptive_parsimony_scaling"))
                prune_start = cfg_overrides.get("variable_prune_start", pysr_kwargs.get("variable_prune_start"))
                prune_ramp = cfg_overrides.get("variable_prune_ramp", pysr_kwargs.get("variable_prune_ramp"))
                prune_max = cfg_overrides.get("variable_prune_max", pysr_kwargs.get("variable_prune_max"))

                for r2w_ in r2w:
                    for l_ in l:
                        outdir = (f"./results_bmi/deeppysr_age/age{age_}_cfg{cfg_name}_par{parsimony}_pop{population}_popsz{pop_size}_"
                                  f"scl{parsimony_scaling}_prnst{prune_start}_ramp{prune_ramp}_max{prune_max}_"
                                  f"r2w{r2w_}_lambda{l_}")
                        csv_path = os.path.join(outdir, "relationships.csv")
                        if os.path.exists(csv_path):
                            df = pd.read_csv(csv_path)
                            if df["layer"].max() >= 1:
                                print(f"[{cfg_name}] Relationships already exist for age{age_} r2w {r2w_}, lambda {l_}. Skipping.")
                                continue

                        print(f"[{cfg_name}] Running DeepPySRRegressor for age{age_} r2w {r2w_}, lambda {l_}.")
                        run_deeppysr(X, y, outdir, r2w=r2w_, l=l_, pysr_overrides=cfg_overrides)


if __name__ == "__main__":
    main()
