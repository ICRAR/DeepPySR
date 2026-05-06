import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import glob
import sys
import re
import sympy as sp

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Add test/ and test/bmi to path to import load_bmi_agg_data
current_dir = os.path.dirname(os.path.abspath(__file__))
if not current_dir:
    current_dir = "."
sys.path.append(os.path.join(current_dir, ".."))
sys.path.append(current_dir)

from bmi_utils import load_bmi_agg_data
from analysis_utils import calculate_metrics, evaluate_formula, get_best_formula_from_raw

def process_results():
    base_dir = os.path.join(current_dir, "results_bmi_all")
    all_data = []

    # Ages to look for
    ages = [8, 10, 14, 17, 20, 23, 27]

    # 1. Process age-specific
    age_spec_dir = os.path.join(base_dir, "age_specific")
    if os.path.exists(age_spec_dir):
        for age_folder in os.listdir(age_spec_dir):
            if not age_folder.startswith("age_"):
                continue
            try:
                age = int(age_folder.split("_")[1])
            except:
                continue
            age_path = os.path.join(age_spec_dir, age_folder)

            # Load data for this age
            _, X_age, y_age = load_bmi_agg_data(age=age)

            # Baselines (including KAN/KANSym)
            baselines_dir = os.path.join(age_path, "baselines")
            if os.path.exists(baselines_dir):
                for model_name in os.listdir(baselines_dir):
                    model_path = os.path.join(baselines_dir, model_name)
                    if not os.path.isdir(model_path):
                        continue

                    # Use overall_metrics.csv/overall_metrics_sym.csv if they exist
                    overall_metrics_file = os.path.join(model_path, "overall_metrics.csv")
                    overall_metrics_sym_file = os.path.join(model_path, "overall_metrics_sym.csv")

                    if model_name.lower() == 'kan':
                        # KAN
                        if os.path.exists(overall_metrics_file):
                            df_metrics = pd.read_csv(overall_metrics_file)
                            r2, rmse, mae = df_metrics['r2'].iloc[0], df_metrics['rmse'].iloc[0], df_metrics['mae'].iloc[0]
                        else:
                            pred_file = os.path.join(model_path, "predictions.csv")
                            if os.path.exists(pred_file):
                                df_pred = pd.read_csv(pred_file)
                                r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                            else:
                                r2, rmse, mae = [np.nan]*3
                        all_data.append([age, 'KAN', 'age-specific', r2, rmse, mae, np.nan, ""])

                        # KANSym
                        if os.path.exists(overall_metrics_sym_file):
                            df_metrics_sym = pd.read_csv(overall_metrics_sym_file)
                            r2, rmse, mae = df_metrics_sym['r2'].iloc[0], df_metrics_sym['rmse'].iloc[0], df_metrics_sym['mae'].iloc[0]
                        else:
                            _, _, metrics = get_best_formula_from_raw(model_path, X_age, y_age, prefix='formulas_fold', model_type='kan')
                            r2, rmse, mae = metrics

                        formula, complexity, _ = get_best_formula_from_raw(model_path, X_age, y_age, prefix='formulas_fold', model_type='kan')
                        all_data.append([age, 'KANSym', 'age-specific', r2, rmse, mae, complexity, formula])
                    else:
                        # Other baselines
                        if os.path.exists(overall_metrics_file):
                            df_metrics = pd.read_csv(overall_metrics_file)
                            r2, rmse, mae = df_metrics['r2'].iloc[0], df_metrics['rmse'].iloc[0], df_metrics['mae'].iloc[0]
                        else:
                            pred_file = os.path.join(model_path, "predictions.csv")
                            if os.path.exists(pred_file):
                                df_pred = pd.read_csv(pred_file)
                                r2, rmse, mae = calculate_metrics(df_pred['y_true'], df_pred['y_pred'])
                            else:
                                r2, rmse, mae = [np.nan]*3
                        all_data.append([age, model_name, 'age-specific', r2, rmse, mae, np.nan, ""])

            # DeepPySR
            deeppysr_dir = os.path.join(age_path, "deeppysr")
            if os.path.exists(deeppysr_dir):
                for variant in os.listdir(deeppysr_dir):
                    v_path = os.path.join(deeppysr_dir, variant)
                    if not os.path.isdir(v_path): continue

                    overall_metrics_file = os.path.join(v_path, "overall_metrics.csv")
                    if os.path.exists(overall_metrics_file):
                        df_metrics = pd.read_csv(overall_metrics_file)
                        r2, rmse, mae = df_metrics['r2'].iloc[0], df_metrics['rmse'].iloc[0], df_metrics['mae'].iloc[0]
                        use_overall = True
                    else:
                        use_overall = False

                    res = get_best_formula_from_raw(v_path, X_age, y_age, model_type='deeppysr')

                    if isinstance(res, dict):
                        for (r2w, lamb), (formula, complexity, metrics) in res.items():
                            if not use_overall:
                                r2, rmse, mae = metrics
                            model_name = f"{variant}_r2w{r2w}_L{lamb}"
                            all_data.append([age, model_name, 'age-specific', r2, rmse, mae, complexity, formula])
                    else:
                        formula, complexity, metrics = res
                        if not use_overall:
                            r2, rmse, mae = metrics
                        all_data.append([age, variant, 'age-specific', r2, rmse, mae, complexity, formula])

            # PySR
            pysr_dir = os.path.join(age_path, "pysr")
            if os.path.exists(pysr_dir):
                for variant in os.listdir(pysr_dir):
                    v_path = os.path.join(pysr_dir, variant)
                    if not os.path.isdir(v_path): continue

                    overall_metrics_file = os.path.join(v_path, "overall_metrics.csv")
                    if os.path.exists(overall_metrics_file):
                        df_metrics = pd.read_csv(overall_metrics_file)
                        r2, rmse, mae = df_metrics['r2'].iloc[0], df_metrics['rmse'].iloc[0], df_metrics['mae'].iloc[0]
                        use_overall = True
                    else:
                        use_overall = False

                    res = get_best_formula_from_raw(v_path, X_age, y_age, model_type='pysr')

                    if isinstance(res, dict):
                        for (r2w, lamb), (formula, complexity, metrics) in res.items():
                            if not use_overall:
                                r2, rmse, mae = metrics
                            model_name = f"{variant}_r2w{r2w}_L{lamb}"
                            all_data.append([age, model_name, 'age-specific', r2, rmse, mae, complexity, formula])
                    else:
                        formula, complexity, metrics = res
                        if not use_overall:
                            r2, rmse, mae = metrics
                        all_data.append([age, variant, 'age-specific', r2, rmse, mae, complexity, formula])


    # 2. Process longitudinal
    long_dir = os.path.join(base_dir, "longitudinal")
    if os.path.exists(long_dir):
        # Load all longitudinal data
        _, X_long, y_long = load_bmi_agg_data()

        sub_dirs = ['baselines', 'deeppysr', 'pysr']
        for sd in sub_dirs:
            sd_path = os.path.join(long_dir, sd)
            if not os.path.exists(sd_path): continue

            for model_folder in os.listdir(sd_path):
                m_path = os.path.join(sd_path, model_folder)
                if not os.path.isdir(m_path): continue

                pred_file = os.path.join(m_path, "predictions.csv")
                if os.path.exists(pred_file):
                    df_pred = pd.read_csv(pred_file)
                    
                    # For DeepPySR and PySR in longitudinal, we might have multiple formulas (r2w, lamb)
                    if sd in ['deeppysr', 'pysr']:
                        res = get_best_formula_from_raw(m_path, X_long, y_long, model_type=sd)
                        
                        if isinstance(res, dict):
                            for (r2w, lamb), (formula, complexity, _) in res.items():
                                model_name = f"{model_folder}_r2w{r2w}_L{lamb}"
                                # Per age metrics
                                for age in ages:
                                    age_mask = (df_pred['age'] == age)
                                    if not df_pred[age_mask].empty:
                                        age_df = df_pred[age_mask]
                                        r2, rmse, mae = calculate_metrics(age_df['y_true'], age_df['y_pred'])
                                        all_data.append([age, model_name, 'longitudinal', r2, rmse, mae, complexity, formula])
                            continue
                        else:
                            formula, complexity, _ = res
                            model_name = model_folder
                            for age in ages:
                                age_mask = (df_pred['age'] == age)
                                if not df_pred[age_mask].empty:
                                    age_df = df_pred[age_mask]
                                    r2, rmse, mae = calculate_metrics(age_df['y_true'], age_df['y_pred'])
                                    all_data.append([age, model_name, 'longitudinal', r2, rmse, mae, complexity, formula])
                            continue

                    # Baselines
                    for age in ages:
                        age_df = df_pred[df_pred['age'] == age]
                        if age_df.empty: continue

                        if sd == 'baselines' and model_folder.lower() == 'kan':
                            # KAN
                            r2, rmse, mae = calculate_metrics(age_df['y_true'], age_df['y_pred'])
                            all_data.append([age, 'KAN', 'longitudinal', r2, rmse, mae, np.nan, ""])

                            # KANSym
                            formula, complexity, _ = get_best_formula_from_raw(m_path, X_long, y_long, prefix='formulas_fold', model_type='kan')
                            r2, rmse, mae = calculate_metrics(age_df['y_true'], age_df['y_pred_kansym'])
                            all_data.append([age, 'KANSym', 'longitudinal', r2, rmse, mae, complexity, formula])
                        else:
                            r2, rmse, mae = calculate_metrics(age_df['y_true'], age_df['y_pred'])
                            all_data.append([age, model_folder, 'longitudinal', r2, rmse, mae, np.nan, ""])

    # Create DataFrame and save
    result_df = pd.DataFrame(all_data, columns=['age', 'model', 'type', 'r2', 'rmse', 'mae', 'complexity', 'formula'])
    result_df['r2'] = result_df['r2'].clip(lower=0)
    
    # Save to bmi_aggregated_results1.csv to avoid overwriting the main one if desired, 
    # but the prompt says "do the similar thing to bmi test", and analysis1.py in bodyfat saves to bodyfat_best_models_metrics.csv etc.
    # Actually analysis1.py in bodyfat had its own save_results function.
    
    output_csv = os.path.join(base_dir, "bmi_aggregated_results.csv")
    result_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return result_df

def save_results(df):
    base_dir = os.path.join(current_dir, "results_bmi_all")
    # For BMI we keep the age-specific/longitudinal distinction
    
    # Metrics vs Age plot (like in analysis_all.py)
    # But let's follow analysis1.py style if it had something different.
    # analysis1.py had plot_best_models.
    pass

# Import the plotting functions from analysis_all since they are quite complex and already handle BMI structure
from analysis_all import plot_results, plot_settings_comparison, aggregate_feature_importance

if __name__ == "__main__":
    df = process_results()
    
    # Use the same plotting functions but with our new aggregated results
    plot_results(df)
    plot_settings_comparison(df)
    aggregate_feature_importance()
