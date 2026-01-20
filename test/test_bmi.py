import numpy as np
from DeepPySR.DeepPySR.regressor import DeepPySRRegressor
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

def main():
    # 1. Prepare data
    print("Generating synthetic data...")
    year = 8
    id, X, y, template = load_data(f"/home/00101787/Projects/pgs/analysis/analysis_prsmochl_kan/plots_paper/yr{year}_kan/data_single.csv",year=year)
    # id, X, y, template = load_data(f"/home/00101787/Projects/pgs/analysis/analysis_prsmochl_kan/plots_paper/yr{year}_kan/data.csv",year=year)
    # 2. Initialize the DeepPySRRegressor
    regressor = DeepPySRRegressor(
        max_layers=4,           # DeepPySR specific: Depth of the symbolic hierarchy
        output_dir=f"./results/yr{year}_test",
        stopping_score=0.001,     # DeepPySR specific: Stop recursion if loss is below this
        
        # PySR parameters (inherited)
        # select_k_features = 15,
        # expression_spec=template,
        complexity_of_variables=0.1,
        maxsize=50,
        # parsimony=1,
        niterations=1000,
        population_size=50,
        model_selection="score",
        # early_stop_condition="f(loss, complexity) = (loss < 0.00001) && (complexity < 20)",
        verbosity = 0,
        denoise=True,
        loss_function='''julia
            function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
            prediction, flag = eval_tree_array(tree, dataset.X, options)
            if !flag
            return L(Inf)
            end
            # log(cosh(x)) acts like x^2/2 for small x and |x| for large x
            return sum(log.(cosh.(prediction .- dataset.y))) / dataset.n
            end
            '''
    )

    # 3. Fit the model
    # This will recursively find relationships: 
    # y = f(x_i) -> x_i = f(x_j) -> ...
    print("Fitting DeepPySRRegressor (this may take a few minutes)...")
    start_time = time.time()
    regressor.fit(X, y)
    duration = time.time() - start_time
    print(f"Fitting completed in {duration/60:.2f} minutes.")

    y_pred = regressor.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"R2 score: {r2:.2f}")

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
