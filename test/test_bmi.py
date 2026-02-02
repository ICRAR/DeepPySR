import numpy as np
from DeepPySR.regressor import DeepPySRRegressor
from DeepPySR.kan_regressor import KANPySRRegressor
from pysr import TemplateExpressionSpec
import os
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score

def load_data(path: str, year: int = 8):
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

def run_deeppysr(X, y, year: int = 8, type='cluster'):
    # 2. Initialize the DeepPySRRegressor
    deeppysr = DeepPySRRegressor(
        max_layers=4,           # DeepPySR specific: Depth of the symbolic hierarchy
        output_dir=f"./results_bmi/deeppysr/yr{year}_{type}_test",
        stopping_score=0.001,     # DeepPySR specific: Stop recursion if loss is below this

        # PySR parameters (inherited)
        # select_k_features = 15,
        # expression_spec=template,
        # complexity_of_variables=0.1,
        maxsize=50,
        binary_operators= ["+", "*","-","/"],
        unary_operators= ["sin","cos","exp", "log","tanh","sqrt","abs"],
        parsimony=-1,
        niterations=1000,
        populations = 30,
        model_selection="accuracy", # score, best, accuracy
        # early_stop_condition="f(loss, complexity) = (loss < 0.00001) && (complexity < 20)",
        verbosity = 0,
        denoise=True,
        loss_function='''
            function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
            prediction, flag = eval_tree_array(tree, dataset.X, options)
            if !flag
            return L(Inf)
            end
            # log(cosh(x)) acts like x^2/2 for small x and |x| for large x
            return sum(log.(cosh.(prediction .- dataset.y))) / dataset.n
            end
            ''',
    )

    # 3. Fit the model
    # This will recursively find relationships:
    # y = f(x_i) -> x_i = f(x_j) -> ...
    print("Fitting DeepPySRRegressor (this may take a few minutes)...")
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
    print(f"\nDemo completed! Check '{os.path.join('demo_outputs', plot_path)}' for the visualization.")

def run_kansr(X, y, year: int = 8,type='cluster'):
    # Initialize KANPySRRegressor
    kansr = KANPySRRegressor(
        kan_width=[X.shape[1], 5, 1], # Simple architecture for demo
        kan_steps=100,
        output_dir=f"./results_bmi/kansr/yr{year}_kansr_{type}",
        pysr_kwargs={
            "model_selection": "score",
            "niterations": 100,
            "population_size": 30,
            "parsimony":0,
            "complexity_of_variables":0.1,
            "binary_operators": ["+", "*","-","/"],
            "unary_operators": ["sin","cos","exp", "log","tanh","sqrt","abs"],
            "verbosity": 0,
            "loss_function":'''
                function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
                prediction, flag = eval_tree_array(tree, dataset.X, options)
                if !flag
                return L(Inf)
                end
                # log(cosh(x)) acts like x^2/2 for small x and |x| for large x
                return sum(log.(cosh.(prediction .- dataset.y))) / dataset.n
                end
                '''
        }
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


def main():
    # 1. Prepare data
    year = 8
    type = 'single'
    if type == 'single':
        id, X, y, template = load_data(f"/home/00101787/Projects/pgs/analysis/analysis_prsmochl_kan/plots_paper/yr{year}_kan/data_single.csv",year=year)
    else:
        id, X, y, template = load_data(f"/home/00101787/Projects/pgs/analysis/analysis_prsmochl_kan/plots_paper/yr{year}_kan/data.csv",year=year)

    run_deeppysr(X, y, year,type)
    # run_kansr(X, y, year,type)



if __name__ == "__main__":
    main()
