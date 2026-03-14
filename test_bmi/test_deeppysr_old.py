import numpy as np
import sympy
import os
import time,torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from DeepPySR.regressor import DeepPySRRegressor
from DeepPySR.kan_regressor import KANPySRRegressor
from kan import KAN
from utils import load_data

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
                "parsimony": 0.001,
                # "niterations": 500,
                "populations": 15,
                "population_size": 100,
                "ncycles_per_iteration": 200,
                "adaptive_parsimony_scaling": 50.0,
                # "variable_prune_start": 50, # new defined
                # "variable_prune_ramp": 150, # new defined
                # "variable_prune_max": 0.6, # new defined
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




def run_deeppysr(X, y, year: int = 8,type='cluster', r2w = 1.,l = 1.,model_provider='pysr'):
    # 2. Initialize the DeepPySRRegressor
    deeppysr = DeepPySRRegressor(
        max_layers=1,           # DeepPySR specific: Depth of the symbolic hierarchy
        output_dir=f"results_bmi/deeppysr_old/yr{year}_{type}_{model_provider}_r2w{r2w}_lambda{l}",
        pareto_lambda = l,
        pareto_r2_weight = r2w,
        stopping_score=0.001,     # DeepPySR specific: Stop recursion if loss is below this
        model_provider=model_provider,
        **pysr_kwargs,
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

def run_kansr(X, y, year: int = 8, type='cluster', r2w = 1.,l = 1.,model_provider='pysr'):
    kansr = KANPySRRegressor(
        kan_width=[X.shape[1], 5, 1], # Simple architecture for demo
        kan_steps=100,
        output_dir=f"./results_bmi/kansr/yr{year}_{type}_{model_provider}_r2w{r2w}_lambda{l}",
        pareto_lambda = l,
        pareto_r2_weight = r2w,
        model_provider=model_provider,
        **pysr_kwargs
    )

    print("Fitting KANPySRRegressor...")
    start_time = time.time()
    kansr.fit(X, y)
    duration = time.time() - start_time
    print(f"Fitting completed in {duration/60:.2f} minutes.")

    y_pred = kansr.predict(X)
    # Handle NaNs and Infs in predictions
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
    r2 = r2_score(y, y_pred)
    print(f"R2 score: {r2:.2f}")

    # Visualize the hierarchy
    plot_path = "kansr_hierarchy.png"
    kansr.plot(plot_path)
    kansr.plot_circle(filename="kansr_circle.png")
    print(f"\nKANSR Demo completed! Check output directory for visualizations.")

def run_kan(X, y, year: int=8, type='single'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = {}
    dataset['train_input'] = torch.from_numpy(X.values).float().to(device)
    dataset['train_label'] = torch.from_numpy(y.values).float().to(device)
    dataset['test_input'] = torch.from_numpy(X.values).float().to(device)
    dataset['test_label'] = torch.from_numpy(y.values).float().to(device)
    save_path = './results_bmi/kan/yr'+str(year)+'_'+type+'/'
    model = KAN(
        width=[X.shape[1],10,1], grid=5, k=3, seed=0, device=device,
        ckpt_path=save_path
    )

    model.fit(dataset,steps =100, lamb=0.1,update_grid=False, lamb_l1=5,lamb_entropy=1,loss_fn=torch.nn.MSELoss(),)
    # model = model.prune()
    model.auto_symbolic()
    y_pred = model(dataset['train_input']).detach().cpu().numpy()
    # Handle NaNs and Infs in predictions
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
    model.plot(folder=save_path,in_vars=X.columns.tolist(),out_vars=['y'+str(year)+'bmi'])
    r2 = r2_score(y, y_pred)
    print(f"R2 score: {r2:.2f}")

    sp_files = os.listdir(save_path)
    for file in sp_files:
        if file.startswith('sp'):
            os.remove(save_path+file)

    weights = torch.matmul(model.acts_scale[0].T, model.acts_scale[1].T).detach().cpu().numpy()
    weight_df = pd.DataFrame()
    weight_df["feature"] = X.columns.tolist()
    weight_df["weight"] = 100 * weights/weights.sum()
    weight_df.to_csv(save_path+'weights.csv',index=False)

    formula, variables = model.symbolic_formula(var=X.columns.tolist())
    file_path = os.path.join(save_path,"formula_raw.txt")
    with open(file_path, "w") as file:
        file.write(str(formula[0]))

def main():
    years = [8,10,13,16,20,23,26]
    # years = [8]
    type = 'single'
    r2w = [1,1.5,2]
    l = [0.001,0.005,0.01,0.1]
    # r2w = [1.5]
    # l = [0.005]
    model_provider = ['pypysr']
    for year in years:
        for model_provider_ in model_provider:
            if type == 'single':
                id, X, y = load_data(f"/home/00101787/Projects/pgs/analysis/analysis_prsmochl_kan/plots_paper/yr{year}_kan/data_single.csv",year=year)
            else:
                id, X, y = load_data(f"/home/00101787/Projects/pgs/analysis/analysis_prsmochl_kan/plots_paper/yr{year}_kan/data.csv",year=year)

            if model_provider_ == "pypysr":
                pysr_kwargs["variable_prune_start"] = 50
                pysr_kwargs["variable_prune_ramp"] = 150
                pysr_kwargs["variable_prune_max"] = 0.6

            for r2w_ in r2w:
                for l_ in l:
                    csv_path = f"results_bmi/deeppysr_old/yr{year}_{type}_{model_provider_}_r2w{r2w_}_lambda{l_}/relationships.csv"
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        if df["layer"].max() >= 3:
                            print(f"Relationships already exist for year {year}, type {type}, r2w {r2w_}, lambda {l_}, model_provider {model_provider_}. Skipping.")
                            continue
                    
                    print(f"Running DeepPySRRegressor for year {year}, type {type}, r2w {r2w_}, lambda {l_}, model_provider {model_provider_}.")
                    run_deeppysr(X, y, year, type,r2w = r2w_,l=l_, model_provider=model_provider_)
                    # run_deeppysr(X, y, year, type,r2w = r2w_,l=l_, model_provider=model_provider_)
                # OR
                #     run_kansr(X, y, year, type, r2w=r2w_,l=l_,model_provider=model_provider_)
                # OR
        # run_kan(X, y, year, type)

if __name__ == "__main__":
    main()
