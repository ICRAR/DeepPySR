import os
import pandas as pd
import re
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utils import load_agg_data

sympy_cond = lambda x, y: sp.Piecewise((y, x > 0), (0, True))

def parse_folder_name(folder_name):
    # Example: yr8_single_pypysr_r2w1.5_lambda0.001
    year_match = re.search(r'yr(\d+)', folder_name)
    model_provider_match = re.search(r'single_([^_]+)', folder_name)
    r2_weight_match = re.search(r'r2w([\d.]+)', folder_name)
    lambda_match = re.search(r'lambda([\d.]+)', folder_name)
    
    year = int(year_match.group(1)) if year_match else None
    model_provider = model_provider_match.group(1) if model_provider_match else None
    r2_weight = float(r2_weight_match.group(1)) if r2_weight_match else None
    complexity_lambda = float(lambda_match.group(1)) if lambda_match else None
    
    return year, model_provider, r2_weight, complexity_lambda

def parse_agg_folder_name(folder_name):
    # Example: par0.001_pop15_popsz100_scl50.0_prnst50_ramp150_max0.7_r2w1_lambda0.005
    patterns = {
        'parsimony': r'par([\d.]+)',
        'population': r'pop(\d+)',
        'pop_size': r'popsz(\d+)',
        'parsimony_scaling': r'scl([\d.]+)',
        'prune_start': r'prnst(\d+)',
        'prune_ramp': r'ramp(\d+)',
        'prune_max': r'max([\d.]+)',
        'r2_weight': r'r2w([\d.]+)',
        'complexity_lambda': r'lambda([\d.]+)'
    }
    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, folder_name)
        if match:
            # Try to convert to int if possible, otherwise float
            val = match.group(1)
            try:
                if '.' in val:
                    results[key] = float(val)
                else:
                    results[key] = int(val)
            except ValueError:
                results[key] = val
        else:
            results[key] = None
    return results

def aggregate_agg_results(base_dir):
    all_data = []
    
    for root, dirs, files in os.walk(base_dir):
        if 'relationships.csv' in files:
            folder_name = os.path.basename(root)
            params = parse_agg_folder_name(folder_name)
            
            file_path = os.path.join(root, 'relationships.csv')
            df = pd.read_csv(file_path)

            # Filter for target 'y'
            df_y = df[df['target'] == 'y'].copy()

            if not df_y.empty:
                # Add metadata columns
                for i, (key, value) in enumerate(params.items()):
                    df_y.insert(i, key, value)
                
                all_data.append(df_y)
    
    if not all_data:
        print(f"No data found in {base_dir}.")
        return
    
    aggregated_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by r2, complexity order
    # Note: 'complexity' is a column in the original CSV
    sort_cols = ['r2_weight', 'complexity_lambda', 'r2', 'complexity']
    # Filter only columns that exist in the dataframe
    sort_cols = [c for c in sort_cols if c in aggregated_df.columns]
    
    ascending = [True] * len(sort_cols)
    if 'r2' in sort_cols:
        r2_idx = sort_cols.index('r2')
        ascending[r2_idx] = False
    
    aggregated_df = aggregated_df.sort_values(by=sort_cols, ascending=ascending)
    
    # Rename columns to match requested names if necessary
    aggregated_df = aggregated_df.rename(columns={
        'r2_weight': 'pareto r2 weight',
        'complexity_lambda': 'lambda for complexity'
    })
    
    output_path = os.path.join(base_dir, 'aggregated_agg_results_bmi.csv')
    aggregated_df.to_csv(output_path, index=False)
    print(f"Aggregated results saved to {output_path}")

def aggregate_results_bmi(base_dir):
    all_data = []
    
    for root, dirs, files in os.walk(base_dir):
        if 'relationships.csv' in files:
            folder_name = os.path.basename(root)
            year, provider, r2w, compl_lambda = parse_folder_name(folder_name)
            
            file_path = os.path.join(root, 'relationships.csv')
            df = pd.read_csv(file_path)

            # Filter for target 'y'
            df_y = df[df['target'] == 'y'].copy()

            if not df_y.empty:
                # Add metadata columns
                df_y.insert(0, 'year', year)
                df_y.insert(1, 'model_provider', provider)
                df_y.insert(2, 'r2_weight', r2w)
                df_y.insert(3, 'complexity_lambda', compl_lambda)
                
                all_data.append(df_y)
    
    if not all_data:
        print("No data found.")
        return
    
    aggregated_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by year, r2, complexity order
    # Note: 'complexity' is a column in the original CSV
    # The requirement says: "sort them in year, r2, complexity order"
    aggregated_df = aggregated_df.sort_values(by=['year', 'r2', 'complexity'], ascending=[True, False, True])
    
    # Rename columns to match requested names if necessary
    # yr8_single_pypysr_r2w1.5_lambda0.001 indicates:
    # year is 8, model provider is pypysr, pareto r2 weight is 1.5, lambda for complexity is 0.001
    aggregated_df = aggregated_df.rename(columns={
        'model_provider': 'model provider',
        'r2_weight': 'pareto r2 weight',
        'complexity_lambda': 'lambda for complexity'
    })
    
    output_path = os.path.join(base_dir, 'results_bmi/deeppysr/aggregated_results_bmi.csv')
    aggregated_df.to_csv(output_path, index=False)
    print(f"Aggregated results saved to {output_path}")

def calculate_performance_metrics(base_dir):
    _, X, y_target = load_agg_data()
    y_target = y_target.values.flatten()
    
    # We need 'age' to calculate metrics per age
    if 'age' not in X.columns:
        print("Error: 'age' column not found in data.")
        return

    ages = X['age'].unique()
    all_metrics = []

    for root, dirs, files in os.walk(base_dir):
        if 'relationships.csv' in files:
            folder_name = os.path.basename(root)
            params = parse_agg_folder_name(folder_name)
            
            file_path = os.path.join(root, 'relationships.csv')
            rel_df = pd.read_csv(file_path)
            
            # Find formula for 'y'
            y_rel = rel_df[rel_df['target'] == 'y']
            if y_rel.empty:
                continue
            
            formula_str = y_rel.iloc[0]['formula']
            involved = [s.strip() for s in y_rel.iloc[0]['involved'].split(',')]
            
            try:
                # Use sympy to evaluate the formula
                # Custom mapping for 'cond'
                extra_mappings = {'cond': sympy_cond}
                modules = [extra_mappings, 'numpy']
                
                # Parse formula to sympy
                # Note: sympy.sympify might not handle custom 'cond' directly if it's not defined
                # But DeepPySRRegressor uses sp.lambdify with custom mappings.
                # We need to ensure all variables are symbols
                symbols = {s: sp.Symbol(s) for s in involved}
                # Replace 'cond' in string to something sympy understands or use parse_expr
                from sympy.parsing.sympy_parser import parse_expr
                expr = parse_expr(formula_str, local_dict=symbols)
                
                # Handle Piecewise/cond if necessary. 
                # If formula_str has 'cond(a, b)', parse_expr might create a Function('cond')(a, b)
                # We need to map it to our sympy_cond
                if 'cond' in formula_str:
                    f_cond = sp.Function('cond')
                    expr = expr.replace(f_cond, sympy_cond)

                func = sp.lambdify([symbols[s] for s in involved], expr, modules=modules)
                
                # Prepare arguments from X
                args = [X[s].values for s in involved]
                y_pred = func(*args)
                
                # Handle NaNs and Infs
                y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
                
                # Calculate overall R2
                overall_r2 = r2_score(y_target, y_pred)
                
                # Calculate metrics per age
                for age in sorted(ages):
                    mask = (X['age'] == age)
                    if not mask.any():
                        continue
                        
                    y_true_age = y_target[mask]
                    y_pred_age = y_pred[mask]
                    
                    r2 = r2_score(y_true_age, y_pred_age)
                    mae = mean_absolute_error(y_true_age, y_pred_age)
                    rmse = np.sqrt(mean_squared_error(y_true_age, y_pred_age))
                    
                    metric_row = params.copy()
                    metric_row.update({
                        'age': age,
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'overall_r2': overall_r2,
                        'formula': formula_str
                    })
                    all_metrics.append(metric_row)
                    
            except Exception as e:
                print(f"Error evaluating formula in {folder_name}: {e}")

    if not all_metrics:
        print("No metrics calculated.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    
    # Rename columns for consistency
    metrics_df = metrics_df.rename(columns={
        'r2_weight': 'pareto r2 weight',
        'complexity_lambda': 'lambda for complexity'
    })
    
    # Sort by metrics
    sort_cols = ['pareto r2 weight', 'lambda for complexity', 'age']
    sort_cols = [c for c in sort_cols if c in metrics_df.columns]
    metrics_df = metrics_df.sort_values(by=sort_cols)
    
    output_path = os.path.join(base_dir, 'performance_metrics_bmi.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"Performance metrics saved to {output_path}")

def plot_performance_metrics_interactive(base_dir):
    csv_path = os.path.join(base_dir, 'performance_metrics_bmi.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Columns that define a combination
    param_cols = [
        'parsimony', 'population', 'pop_size', 'parsimony_scaling', 
        'prune_start', 'prune_ramp', 'prune_max', 
        'pareto r2 weight', 'lambda for complexity'
    ]
    
    # Filter only columns that exist
    param_cols = [c for c in param_cols if c in df.columns]
    
    # Group by parameter combinations
    grouped = df.groupby(param_cols)
    
    # Sort by parameter combinations
    # Sorting group members by age is already done later
    
    metrics = ['r2', 'mae', 'rmse']
    titles = ['R2 Score', 'MAE', 'RMSE']
    
    # R2 (top left, 1,1), MAE (top right, 1,2), RMSE (bottom right, 2,2)
    # Legend area (bottom left, 2,1)
    fig = make_subplots(rows=2, cols=2, 
                        shared_xaxes=False, 
                        subplot_titles=['R2 Score', 'MAE', '', 'RMSE'],
                        vertical_spacing=0.15,
                        horizontal_spacing=0.1)
    
    # Mapping metrics to grid positions
    # R2: (1,1), MAE: (1,2), RMSE: (2,2)
    metric_pos = {
        'r2': (1, 1),
        'mae': (1, 2),
        'rmse': (2, 2)
    }
    
    # Color palette
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Track traces for spikelines and unified hover
    for color_idx, (params, group) in enumerate(grouped):
        # Create label from params
        label_parts = []
        if isinstance(params, (list, tuple)):
            for col, val in zip(param_cols, params):
                label_parts.append(f"{col}={val}")
        else:
            label_parts.append(f"{param_cols[0]}={params}")
        
        label = ", ".join(label_parts)
        group = group.sort_values('age')
        
        color = colors[color_idx % len(colors)]
        
        for i, metric in enumerate(metrics):
            show_legend = (i == 0) # Only show legend once per group
            row, col = metric_pos[metric]
            fig.add_trace(
                go.Scatter(
                    x=group['age'], 
                    y=group[metric], 
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=2.5),
                    legendgroup=label,
                    showlegend=show_legend,
                    # Ensure hover is shared across the group
                    hoverinfo='all',
                    hovertemplate=f"Age: %{{x}}<br>{metric.upper()}: %{{y}}<br>Formula: {group.iloc[0]['formula']}<extra></extra>"
                ),
                row=row, col=col
            )
            
    fig.update_layout(
        height=900,
        width=1800,
        title_text="Performance Metrics by Age (Interactive)",
        showlegend=True,
        # Sync hover across subplots
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=-0.05,
            xanchor="left",
            x=-0.03,
            # Limit the width of the legend to prevent it from squashing the plot
            entrywidth=350,
            # Sync entire group toggle
            groupclick="togglegroup",
            # Make the legend scrollable
            traceorder="normal",
            itemsizing="constant",
            # Add a background to the legend for better readability
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        # Adjust margins
        margin=dict(r=80, l=80, t=100, b=50),
        # Reduce font size to fit more in the legend
        legend_font_size=10
    )
    
    # Constrain legend height to make it scrollable
    fig.update_layout(legend=dict(
        maxheight=400
    ))
    
    for r, c in [(1,1), (1,2), (2,1), (2,2)]:
        if r == 2 and c == 1:
            continue # Legend area
        # Sync x-axes and add spikelines for highlighting across plots
        fig.update_xaxes(
            title_text="Age", 
            row=r, col=c,
            matches='x', # Link all x-axes
            showspikes=True, 
            spikemode='across', 
            spikesnap='cursor',
            spikedash='dot',
            spikethickness=1
        )
    
    fig.update_yaxes(title_text="R2 Score", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    fig.update_yaxes(title_text="RMSE", row=2, col=2)

    output_html = os.path.join(base_dir, 'performance_metrics_plot.html')
    fig.write_html(output_html)
    print(f"Interactive plot saved to {output_html}")

if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Process results_bmi/deeppysr
    # deeppysr_dir = os.path.join(base_directory, 'results_bmi', 'deeppysr')
    # if os.path.exists(deeppysr_dir):
    #     aggregate_results_bmi(deeppysr_dir)
        
    # Process results_bmi/agg_deeppysr
    agg_deeppysr_dir = os.path.join(base_directory, 'results_bmi', 'agg_deeppysr')
    if os.path.exists(agg_deeppysr_dir):
        aggregate_agg_results(agg_deeppysr_dir)
        calculate_performance_metrics(agg_deeppysr_dir)
        plot_performance_metrics_interactive(agg_deeppysr_dir)
