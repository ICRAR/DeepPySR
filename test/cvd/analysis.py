import os
import pandas as pd
import re
import sympy as sp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_complexity(formula_str):
    if not formula_str or formula_str == "Failed to extract formula" or formula_str == "None":
        return 0
    try:
        from sympy.parsing.sympy_parser import parse_expr
        # CVD variables are usually x0, x1, ... in DeepPySR but our script uses indices.
        # DeepPySRRegressor might use x0, x1...
        # Let's handle 'cond'
        sympy_cond = lambda x, y: sp.Piecewise((y, x > 0), (0, True))
        
        # Replace 'cond(a, b)' with something sympy can parse if it's not already handled
        # Actually parse_expr might create Function('cond')(a, b)
        expr = parse_expr(str(formula_str))
        
        if 'cond' in str(formula_str):
            f_cond = sp.Function('cond')
            expr = expr.replace(f_cond, sympy_cond)

        def count_nodes(e):
            if not hasattr(e, 'args') or not e.args:
                return 1
            return 1 + sum(count_nodes(arg) for arg in e.args)
            
        return count_nodes(expr)
    except Exception as e:
        # print(f"Warning: Could not calculate complexity for formula '{formula_str}': {e}")
        return len(re.findall(r'\w+|[+\-*/()^]', str(formula_str)))

def parse_params(model_name):
    params = {}
    
    pop_match = re.search(r'pop(\d+)', model_name)
    prnst_match = re.search(r'prnst(\d+)', model_name)
    ramp_match = re.search(r'ramp(\d+)', model_name)
    max_match = re.search(r'max([\d.]+)', model_name)
    r2_weight_match = re.search(r'r2w([\d.]+)', model_name)
    lambda_match = re.search(r'l([\d.]+)', model_name)
    
    # KAN params
    kan_lamb_match = re.search(r'lamb([\d.]+)', model_name)
    kan_l1_match = re.search(r'l1([\d.]+)', model_name)
    kan_ent_match = re.search(r'ent([\d.]+)', model_name)
    kan_hidden_match = re.search(r'hidden(\d+)', model_name)
    
    if pop_match: params['pop'] = int(pop_match.group(1))
    if prnst_match: params['prnst'] = int(prnst_match.group(1))
    if ramp_match: params['ramp'] = int(ramp_match.group(1))
    if max_match: params['pmax'] = float(max_match.group(1))
    if r2_weight_match: params['r2w'] = float(r2_weight_match.group(1))
    if lambda_match: params['l'] = float(lambda_match.group(1))
    
    # Standard SR params
    par_match = re.search(r'par([\d.]+)', model_name)
    popsz_match = re.search(r'popsz(\d+)', model_name)
    scl_match = re.search(r'scl([\d.]+)', model_name)
    lambda_long_match = re.search(r'lambda([\d.]+)', model_name)

    if par_match: params['parsimony'] = float(par_match.group(1))
    if popsz_match: params['pop_size'] = int(popsz_match.group(1))
    if scl_match: params['parsimony_scaling'] = float(scl_match.group(1))
    if lambda_long_match: params['l'] = float(lambda_long_match.group(1))
    
    # Extract cfg from folder name
    cfg_match = re.search(r'cfg(\w+)', model_name)
    if cfg_match: params['cfg'] = cfg_match.group(1)

    if kan_lamb_match: params['kan_lamb'] = float(kan_lamb_match.group(1))
    if kan_l1_match: params['kan_l1'] = float(kan_l1_match.group(1))
    if kan_ent_match: params['kan_ent'] = float(kan_ent_match.group(1))
    if kan_hidden_match: params['kan_hidden'] = int(kan_hidden_match.group(1))
    
    return params

def run_cvd_analysis():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(base_dir, 'results_cvd')
    
    all_metrics = []
    
    # Walk through results_cvd to find all metrics.csv and metrics_sym.csv files
    for root, dirs, files in os.walk(out_root):
        for f in files:
            if f == 'metrics.csv' or f == 'metrics_sym.csv':
                metrics_path = os.path.join(root, f)
                try:
                    df = pd.read_csv(metrics_path)
                    # Add path info to help distinguish models/params
                    rel_path = os.path.relpath(root, out_root)
                    df['path'] = rel_path
                    
                    # Try to extract parameters from path or model name
                    params = parse_params(rel_path)
                    for col, val in params.items():
                        df[col] = val
                    
                    # Mark if it's No-CV
                    # For KAN models in the new structure, both CV and KANSYM metrics come from the same folder
                    # but the requirement says best KANSYM from No-CV settings.
                    # In test_all_models.py, y_pred_kansym is always the No-CV formula prediction.
                    # So we should mark KANSYM models as is_nocv = True even if the folder is not 'nocv'
                    is_nocv_folder = 'nocv' in rel_path
                    df['is_nocv'] = is_nocv_folder

                    # Special handling for KAN in the new structure (merged)
                    if 'kan' in rel_path.lower() and f == 'metrics.csv':
                        # Check if this folder also has y_pred_kansym in predictions.csv
                        preds_path = os.path.join(root, 'predictions.csv')
                        if os.path.exists(preds_path):
                            preds_df = pd.read_csv(preds_path)
                            if 'y_pred_kansym' in preds_df.columns:
                                # We have both KAN and KANSYM results in the same predictions.csv
                                # The metrics.csv only has KAN metrics.
                                # Let's create a separate KANSYM entry
                                sym_df = df.copy()
                                sym_df['model'] = sym_df['model'].apply(lambda x: f"KANSYM_{x}" if 'KANSYM' not in x else x)
                                # KANSYM in this script is always considered No-CV as it uses the No-CV formula
                                sym_df['is_nocv'] = True
                                try:
                                    y_true = preds_df['y_true']
                                    y_pred = preds_df['y_pred_kansym']
                                    # Use y_prob_kansym if available
                                    
                                    sym_df['accuracy'] = accuracy_score(y_true, y_pred)
                                    sym_df['precision'] = precision_score(y_true, y_pred, zero_division=0)
                                    sym_df['recall'] = recall_score(y_true, y_pred, zero_division=0)
                                    sym_df['f1'] = f1_score(y_true, y_pred, zero_division=0)
                                    
                                    if 'y_prob_kansym' in preds_df.columns:
                                        sym_df['auc'] = roc_auc_score(y_true, preds_df['y_prob_kansym'])
                                    else:
                                        sym_df['auc'] = sym_df['accuracy'] 
                                    
                                    all_metrics.append(sym_df)
                                except Exception as e:
                                    print(f"Error computing KANSYM metrics for {rel_path}: {e}")
                                    all_metrics.append(sym_df) # fallback to copied metrics
                                
                                # Don't return, we still need to add the original KAN df
                    
                    all_metrics.append(df)
                except Exception as e:
                    print(f"Error reading {metrics_path}: {e}")
    
    if not all_metrics:
        print("No results found to aggregate.")
        return

    full_results_df = pd.concat(all_metrics, ignore_index=True)
    
    # 3. Calculate complexity for symbolic models
    # We'll check for relationships.csv in nocv folders for each grid point
    complexities = []
    for idx, row in full_results_df.iterrows():
        m_name = row['model']
        comp = 0
        
        # Check if it's a symbolic model and No-CV
        is_symbolic_nocv = any(x in m_name.lower() for x in ['deeppysr', 'pysr', 'kansym']) and row['is_nocv']
        if is_symbolic_nocv:
            # Try to find relationships.csv or nocv_formula.csv
            # For DeepPySR/PySR grid: out_root/model_type/params/nocv/relationships.csv
            # For KANSYM: out_root/kansym/params/nocv/nocv_formula.csv or relationships.csv
            rel_path = None
            if 'deeppysr' in m_name.lower() or 'pysr' in m_name.lower():
                # Extract parts
                model_type = 'deeppysr' if 'deeppysr' in m_name.lower() else 'pysr'
                params_str = ""
                if model_type == 'deeppysr':
                    m = re.search(r'pop.*', m_name)
                    if m: params_str = m.group(0).replace('_nocv', '')
                elif model_type == 'pysr':
                    m = re.search(r'r2w.*', m_name)
                    if m: params_str = m.group(0).replace('_nocv', '')
                
                if params_str:
                    rel_path = os.path.join(out_root, model_type, params_str, 'nocv', 'relationships.csv')
            elif 'kansym' in m_name.lower():
                # KANSYM results are now in results_cvd/kan/params/
                m = re.search(r'lamb.*', m_name)
                if m:
                    params_str = m.group(0).replace('_nocv', '')
                    # Check new path: results_cvd/kan/params/nocv/nocv_formula.csv
                    rel_path = os.path.join(out_root, 'kan', params_str, 'nocv', 'nocv_formula.csv')
                    if not os.path.exists(rel_path):
                        # Try old path: results_cvd/kansym/params/nocv/nocv_formula.csv
                        rel_path = os.path.join(out_root, 'kansym', params_str, 'nocv', 'nocv_formula.csv')
                    
                    if not os.path.exists(rel_path):
                        rel_path = os.path.join(out_root, 'kan', params_str, 'nocv', 'relationships.csv')
                    if not os.path.exists(rel_path):
                        rel_path = os.path.join(out_root, 'kansym', params_str, 'nocv', 'relationships.csv')
                else:
                    rel_path = os.path.join(out_root, 'kan', 'nocv', 'nocv_formula.csv')
                    if not os.path.exists(rel_path):
                        rel_path = os.path.join(out_root, 'kansym', 'nocv', 'nocv_formula.csv')
            
            if rel_path and os.path.exists(rel_path):
                try:
                    rel_df = pd.read_csv(rel_path)
                    if 'formula' in rel_df.columns:
                        formula = rel_df.iloc[0]['formula']
                        comp = calculate_complexity(formula)
                except:
                    pass
        complexities.append(comp)
    
    full_results_df['complexity'] = complexities
    
    # Save aggregated results
    aggregated_path = os.path.join(out_root, 'aggregated_results.csv')
    full_results_df.to_csv(aggregated_path, index=False)
    print(f"Aggregated results saved to {aggregated_path}")
    
    # 4. Select best F1 for DeepPySR, PySR, KAN, and KANSYM
    # Criteria: 
    # - KAN (CV)
    # - KANSYM (No-CV)
    # - PySR (CV and No-CV)
    # - DeepPySR (CV and No-CV)
    
    cv_results = full_results_df[full_results_df['is_nocv'] == False]
    nocv_results = full_results_df[full_results_df['is_nocv'] == True]
    
    def get_best_model(df, pattern, exclude_pattern=None):
        filtered = df[df['model'].str.contains(pattern, case=False, na=False)].copy()
        if exclude_pattern:
            filtered = filtered[~filtered['model'].str.contains(exclude_pattern, case=False, na=False)]
        if not filtered.empty:
            return filtered.loc[[filtered['f1'].idxmax()]]
        return pd.DataFrame()

    best_pypysr_cv = get_best_model(cv_results, 'DeepPySR')
    best_pypysr_nocv = get_best_model(nocv_results, 'DeepPySR')
    
    best_pysr_cv = get_best_model(cv_results, 'PySR', exclude_pattern='DeepPySR')
    best_pysr_nocv = get_best_model(nocv_results, 'PySR', exclude_pattern='DeepPySR')
    
    best_kan_cv = get_best_model(cv_results, r'^KAN_')
    best_kansym_nocv = get_best_model(nocv_results, 'KANSYM')
    
    # Models to exclude from "other_models" because they were grid searched
    grid_searched_patterns = 'DeepPySR|PySR|^KAN_|KANSYM'
    other_models = cv_results[~cv_results['model'].str.contains(grid_searched_patterns, case=False, na=False)]
    
    best_combined_df = pd.concat([
        other_models, 
        best_pypysr_cv, best_pypysr_nocv,
        best_pysr_cv, best_pysr_nocv,
        best_kan_cv, best_kansym_nocv
    ], ignore_index=True)
    
    # 5. Fix complexity for best_models_combined
    # No-CV models should have their complexity copied from the No-CV runs
    # But CV models might not have complexity in aggregated_results.csv if we only calculated it for No-CV.
    # Actually, let's ensure all models in best_combined have complexity if they are symbolic.
    for idx, row in best_combined_df.iterrows():
        if row['complexity'] == 0:
            m_name = row['model']
            is_symbolic = any(x in m_name.lower() for x in ['deeppysr', 'pysr', 'kansym', 'kan'])
            if is_symbolic:
                # Try to find the corresponding No-CV complexity
                # Find matching params in full_results_df where is_nocv is True
                params = parse_params(row['path'] if 'path' in row else m_name)
                # match by params
                match = full_results_df[full_results_df['is_nocv'] == True]
                for p_key, p_val in params.items():
                    if p_key in match.columns:
                        match = match[match[p_key] == p_val]
                
                if not match.empty:
                    # Prefer KANSYM complexity if the model is KAN or KANSYM
                    if 'kan' in m_name.lower():
                        kan_match = match[match['model'].str.contains('KANSYM', case=False, na=False)]
                        if not kan_match.empty:
                            best_combined_df.at[idx, 'complexity'] = kan_match.iloc[0]['complexity']
                        else:
                            best_combined_df.at[idx, 'complexity'] = match.iloc[0]['complexity']
                    else:
                        best_combined_df.at[idx, 'complexity'] = match.iloc[0]['complexity']

    best_combined_path = os.path.join(out_root, 'best_models_combined.csv')
    best_combined_df.to_csv(best_combined_path, index=False)
    print(f"Best models combined saved to {best_combined_path}")
    
    # 6. Save aggregated feature importance as CSV
    # Gather feature importance from nocv folders
    all_fi = []
    for idx, row in best_combined_df.iterrows():
        m_name = row['model']
        # Skip DeepPySR/PySR as requested
        if any(x in m_name.lower() for x in ['deeppysr', 'pysr']):
            continue
            
        # Try to find feature_importance.csv in nocv folder or main folder
        # For KAN/KANSYM, look in 'kan' folder under its specific params
        model_dir_name = m_name.lower()
        rel_path = row['path'] if 'path' in row else model_dir_name
        
        # If it's a KAN/KANSYM model, we need to handle the nested structure: kan/lamb.../nocv/feature_importance.csv
        if 'kan' in model_dir_name:
            # Extract the param part of the path
            # rel_path for KAN usually looks like 'kan/lamb0.0_l10.0_ent0.0_hidden2'
            # or 'kan/lamb0.0_l10.0_ent0.0_hidden2/nocv'
            kan_base = rel_path.split('/nocv')[0]
            fi_path = os.path.join(out_root, kan_base, 'nocv', 'feature_importance.csv')
        else:
            # Traditional models: results_cvd/logisticregression/feature_importance.csv
            fi_path = os.path.join(out_root, model_dir_name, 'feature_importance.csv')
            
        if os.path.exists(fi_path):
            fi_df = pd.read_csv(fi_path)
            fi_df['model'] = m_name
            all_fi.append(fi_df)
        elif 'kan' not in model_dir_name:
            # Try without lower() just in case
            fi_path = os.path.join(out_root, m_name, 'feature_importance.csv')
            if os.path.exists(fi_path):
                fi_df = pd.read_csv(fi_path)
                fi_df['model'] = m_name
                all_fi.append(fi_df)
            
    if all_fi:
        full_fi_df = pd.concat(all_fi, ignore_index=True)
        fi_out_path = os.path.join(out_root, 'feature_importance_aggregated.csv')
        full_fi_df.to_csv(fi_out_path, index=False)
        print(f"Aggregated feature importance saved to {fi_out_path}")

def compare_arg_configs_best(base_dir):
    """
    Compare the four SR argument configurations for CVD.
    """
    import matplotlib.pyplot as plt
    import os
    import pandas as pd

    csv_path = os.path.join(base_dir, 'aggregated_results.csv')
            
    if not os.path.exists(csv_path):
        print(f"Error: aggregated results CSV not found in {base_dir}. Skipping arg-config comparison.")
        return

    df = pd.read_csv(csv_path)

    if 'cfg' not in df.columns:
        print("Warning: 'cfg' column not found in metrics. Skipping plot.")
        return
    
    # Filter for DeepPySR models only (they have 'cfg' populated)
    df = df[df['model'].str.contains('DeepPySR', case=False, na=False)].copy()
    
    # Filter for consistent r2w of 1 and lambda of 0.001
    df = df[
        (df['r2w'] == 1) & 
        (df['l'] == 0.001)
    ]
    
    # Define the config requirements (must match folder naming/config in test_all_models.py)
    arg_configs = {
        "stdsr": {
            "parsimony_scaling": 0.0,
            "pmax": 0.0,
            "prnst": 0,
            "ramp": 0,
        },
        "srprn": {
            "parsimony_scaling": 0.0,
            "prnst": 50,
            "ramp": 150,
            "pmax": 0.7,
        },
        "srpsm": {
            "parsimony_scaling": 1040.0,
            "pmax": 0.0,
            "prnst": 0,
            "ramp": 0,
        },
        "fullsr": {
            "parsimony_scaling": 1040.0,
            "prnst": 50,
            "ramp": 150,
            "pmax": 0.7,
        },
    }

    # Filter df to only include rows that match the arg_configs requirements for each cfg
    filtered_rows = []
    for cfg_name, requirements in arg_configs.items():
        cfg_df = df[df['cfg'] == cfg_name]
        for col, val in requirements.items():
            if col in cfg_df.columns:
                cfg_df = cfg_df[cfg_df[col] == val]
        filtered_rows.append(cfg_df)
    
    best_df = pd.concat(filtered_rows) if filtered_rows else pd.DataFrame(columns=df.columns)
    
    if best_df.empty:
        print("No data found for comparison.")
        return

    # In CVD, we don't have 'age', so we plot as a bar chart across configurations
    metrics = ['accuracy', 'f1', 'complexity', 'auc']
    titles = ['Accuracy', 'F1 Score', 'Complexity', 'AUC']
    ylabel = ['Accuracy', 'F1 Score', 'Complexity', 'AUC']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    fig.suptitle(f"Comparison of Configurations: CVD DeepPySR (r2w=1, lambda=0.001)", fontsize=16)
    
    configs = ["stdsr", "srprn", "srpsm", "fullsr"]
    colors = {
        'stdsr': '#1f77b4',
        'srprn': '#ff7f0e',
        'srpsm': '#2ca02c',
        'fullsr': '#d62728',
    }
    
    # Label mapping for the plot
    labels = {
        "stdsr": "stdsr\n(scl=0, prn_max=0)",
        "srprn": "srprn\n(scl=0, prn=50/150/0.7)",
        "srpsm": "srpsm\n(scl=1040, prn_max=0)",
        "fullsr": "fullsr\n(scl=1040, prn=50/150/0.7)",
    }

    plot_data = best_df.copy()
    # Sort or reindex by configs
    plot_data['cfg'] = pd.Categorical(plot_data['cfg'], categories=configs, ordered=True)
    plot_data = plot_data.sort_values('cfg')

    for i, metric in enumerate(metrics):
        bars = axes[i].bar(plot_data['cfg'].map(labels), plot_data[metric], color=[colors[c] for c in plot_data['cfg']])
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(ylabel[i])
        axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for bar in bars:
            yval = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}' if metric != 'complexity' else f'{int(yval)}', 
                         va='bottom', ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_png = os.path.join(base_dir, 'argconfigs_best_comparison.png')
    plt.savefig(output_png)
    print(f"Arg-config best performance comparison plot saved to {output_png}")
    plt.close()

if __name__ == "__main__":
    run_cvd_analysis()
    # Add the comparison plot
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(base_dir, 'results_cvd')
    compare_arg_configs_best(out_root)
