import numpy as np
import sympy
import os
import time,torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score


sympy_cond = lambda x, y: sympy.Piecewise((y, x > 0), (0, True))
# template = TemplateExpressionSpec(expressions=["f", "g", "h", "k"],combine="f + g + h + k")
pysr_kwargs = {
                "select_k_features": 10,
                # "expression_spec": template,
                "complexity_of_variables":0.01,
                "maxsize":30,
                "binary_operators": ["+", "*","cond(x,y) = x > 0 ? y : y*0"],
                "extra_sympy_mappings":{'cond': sympy_cond},
                "unary_operators": ["sin","cos","exp", "log","sqrt","abs"],
                "parsimony":0.001,
                "niterations":500,
                "populations": 30,
                "model_selection":"accuracy", # score, best, accuracy
                # "early_stop_condition": "f(loss, complexity) = (loss < 0.00001) && (complexity < 20)",
                "verbosity":0,
                "denoise":True,
                "turbo": True,
                "procs": max(1, (os.cpu_count() or 2) - 1),
                "loss_function":'''
                            function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
                            prediction, flag = eval_tree_array(tree, dataset.X, options)
                            if !flag
                            return L(Inf)
                            end
                            # log(cosh(x)) acts like x^2/2 for small x and |x| for large x
                            return sum(log.(cosh.(prediction .- dataset.y))) / dataset.n
                            end
                            ''',}


def load_data(path: str, year: int = 8):
    from pysr import TemplateExpressionSpec
    data = pd.read_csv(path)
    dataid = data[["child_id"]].T.drop_duplicates().T.values.reshape(1, -1)[0]
    data.columns = data.columns.str.replace(',','_')
    params = {}
    if 'occupcode_m_0' in data.columns:
        data['occupcode_m_0'] = data['occupcode_m_0']+1
        params['occupcode_m_0'] = 10
    if 'occupcode_f1_1' in data.columns:
        data['occupcode_f1_1'] = data['occupcode_f1_1']+1
        params['occupcode_f1_1'] = 10
    if 'hhincome_0' in data.columns:
        params['hhincome_0'] = 4
    if 'prepreg_smk' in data.columns:
        data['prepreg_smk'] = data['prepreg_smk']+1
        params['prepreg_smk'] = 2
    if 'prepreg_cig' in data.columns:
        data['prepreg_cig'] = data['prepreg_cig']+1
        params['prepreg_cig'] = 3

    datain = data.drop(columns=['child_id',f'y{year}bmi',f'pred_y{year}bmi'])
    dataout = data[[f'y{year}bmi']]

    var_names = datain.columns.tolist()
    combine_str = ''
    for var_name in var_names:
        if var_name in params:
            combine_str += f'{var_name}[{var_name}], '
        else:
            combine_str += f'{var_name}, '
    combine_str = 'f(' + combine_str[:-2] +')'

    template = TemplateExpressionSpec(
        expressions=["f"],
        variable_names=var_names,
        parameters=params,
        combine=combine_str
    )
    return dataid,datain,dataout,template

def run_deeppysr(X, y, year: int = 8, type='cluster', project_path: str = None, run_type='deeppysr'):
    if project_path:
        from pathlib import Path
        from pysr import jl
        proj = str(Path(os.path.expanduser(project_path)).resolve())
        print(f"Replacing SymbolicRegression with custom package at: {proj}")
        jl.seval(f'using Pkg; Pkg.develop(path="{proj}")')
        jl.seval('using MyPySR')

    from DeepPySR.regressor import DeepPySRRegressor
    # 2. Initialize the DeepPySRRegressor
    deeppysr = DeepPySRRegressor(
        max_layers=4,           # DeepPySR specific: Depth of the symbolic hierarchy
        output_dir=f"./results_bmi/{run_type}/yr{year}_{type}_test",
        stopping_score=0.001,     # DeepPySR specific: Stop recursion if loss is below this
        **pysr_kwargs
    )

    # 3. Fit the model
    # This will recursively find relationships:
    # y = f(x_i) -> x_i = f(x_j) -> ...
    print(f"Fitting DeepPySRRegressor ({run_type}) (this may take a few minutes)...")
    if project_path:
        print(f"Using custom Julia project at: {project_path}")
    start_time = time.time()
    deeppysr.fit(X, y)
    duration = time.time() - start_time
    print(f"Fitting completed in {duration/60:.2f} minutes.")

    y_pred = deeppysr.predict(X)
    r2 = r2_score(y, y_pred,force_finite=False)
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
    print(f"\n{run_type.capitalize()} run completed! Check output directory for visualizations.")

def run_kansr(X, y, year: int = 8, type='cluster', project_path: str = None):
    if project_path:
        from pathlib import Path
        from pysr import jl
        proj = str(Path(os.path.expanduser(project_path)).resolve())
        print(f"Replacing SymbolicRegression with custom package at: {proj}")
        jl.seval(f'using Pkg; Pkg.develop(path="{proj}")')
        jl.seval('using MyPySR')

    from DeepPySR.kan_regressor import KANPySRRegressor
    kansr = KANPySRRegressor(
        kan_width=[X.shape[1], 5, 1], # Simple architecture for demo
        kan_steps=100,
        output_dir=f"./results_bmi/kansr/yr{year}_kansr_{type}",
        **pysr_kwargs
    )

    print("Fitting KANPySRRegressor...")
    start_time = time.time()
    kansr.fit(X, y)
    duration = time.time() - start_time
    print(f"Fitting completed in {duration/60:.2f} minutes.")

    y_pred = kansr.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"R2 score: {r2:.2f}")

    # Visualize the hierarchy
    plot_path = "kansr_hierarchy.png"
    kansr.plot(plot_path)
    kansr.plot_circle(filename="kansr_circle.png")
    print(f"\nKANSR Demo completed! Check output directory for visualizations.")

def run_kan(X, y, year: int=8, type='single'):
    from kan import KAN
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
    # 1. Prepare data
    year = 8
    type = 'single'

    # Option 1: Use the original pysr package (set to None)
    # Option 2: Use your custom project (set to path)
    project_path = '~/Projects/mypysr.jl'

    if project_path:
        from pathlib import Path
        from pysr import jl
        proj = str(Path(os.path.expanduser(project_path)).resolve())
        print(f"Configuring custom Julia package path: {proj}")
        # Note: Actual activation/replacement happens inside run_deeppysr/run_kansr
    else:
        print("Using default Julia environment (original pysr)")

    if type == 'single':
        id, X, y, template = load_data(f"/home/00101787/Projects/pgs/analysis/analysis_prsmochl_kan/plots_paper/yr{year}_kan/data_single.csv",year=year)
    else:
        id, X, y, template = load_data(f"/home/00101787/Projects/pgs/analysis/analysis_prsmochl_kan/plots_paper/yr{year}_kan/data.csv",year=year)

    # Now you can call any of these with the configured project:
    run_deeppysr(X, y, year, type, project_path=project_path, run_type='deeppysr' if project_path is None else 'pysrvar')
    # OR
    # run_kansr(X, y, year, type, project_path=project_path)
    # OR
    # run_kan(X, y, year, type)

if __name__ == "__main__":
    main()
