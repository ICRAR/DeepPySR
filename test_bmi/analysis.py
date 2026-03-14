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

def calculate_complexity(formula_str):
    if not formula_str or formula_str == "Failed to extract formula":
        return 0
    try:
        # Standardize the formula for sympy parsing
        # KAN formulas might have some peculiarities, but sympy.parse_expr is generally robust
        # We need to handle potential custom functions if they appear in KAN
        from sympy.parsing.sympy_parser import parse_expr
        # Involved variables are usually x_0, x_1, ... in KAN symbolic formulas
        expr = parse_expr(formula_str)
        
        def count_nodes(e):
            return 1 + sum(count_nodes(arg) for arg in e.args)
            
        return count_nodes(expr)
    except Exception as e:
        print(f"Warning: Could not calculate complexity for formula '{formula_str}': {e}")
        # Fallback: count words/operators as a very rough estimate
        return len(re.findall(r'\w+|[+\-*/()^]', formula_str))

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
    
    output_path = os.path.join(base_dir, 'results_bmi/deeppysr_old/aggregated_results_bmi.csv')
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
                overall_r2 = max(0, min(1, r2_score(y_target, y_pred)))
                
                # Calculate metrics per age
                for age in sorted(ages):
                    mask = (X['age'] == age)
                    if not mask.any():
                        continue
                        
                    y_true_age = y_target[mask]
                    y_pred_age = y_pred[mask]
                    
                    r2 = max(0, min(1, r2_score(y_true_age, y_pred_age)))
                    mae = mean_absolute_error(y_true_age, y_pred_age)
                    rmse = np.sqrt(mean_squared_error(y_true_age, y_pred_age))
                    
                    metric_row = params.copy()
                    metric_row.update({
                        'age': age,
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'overall_r2': overall_r2,
                        'complexity': y_rel.iloc[0]['complexity'],
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
    
    metrics = ['r2', 'mae', 'rmse', 'complexity']
    titles = ['R2 Score', 'MAE', 'Formula Complexity', 'RMSE']
    
    # R2 (top left, 1,1), MAE (top right, 1,2), Complexity (bottom left, 2,1), RMSE (bottom right, 2,2)
    fig = make_subplots(rows=2, cols=2, 
                        shared_xaxes=False, 
                        subplot_titles=titles,
                        vertical_spacing=0.15,
                        horizontal_spacing=0.1)
    
    # Mapping metrics to grid positions
    # R2: (1,1), MAE: (1,2), Complexity: (2,1), RMSE: (2,2)
    metric_pos = {
        'r2': (1, 1),
        'mae': (1, 2),
        'complexity': (2, 1),
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
                    hovertemplate=f"Age: %{{x}}<br>{metric.upper()}: %{{y:.4f}}<br>Model: {label}<br>Formula: {group.iloc[0]['formula']}<extra></extra>"
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
    fig.update_yaxes(title_text="Complexity", row=2, col=1)
    fig.update_yaxes(title_text="RMSE", row=2, col=2)

    output_html = os.path.join(base_dir, 'performance_metrics_plot.html')
    fig.write_html(output_html)
    print(f"Interactive plot saved to {output_html}")

def parse_age_folder_name(folder_name):
    # Example: age10_par0.01_pop20_popsz100_scl100.0_prnst50_ramp150_max0.6_r2w1.5_lambda0.001
    patterns = {
        'age': r'age(\d+)',
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

def aggregate_age_results(base_dir):
    all_data = []
    
    for root, dirs, files in os.walk(base_dir):
        if 'relationships.csv' in files:
            folder_name = os.path.basename(root)
            params = parse_age_folder_name(folder_name)
            
            file_path = os.path.join(root, 'relationships.csv')
            df = pd.read_csv(file_path)

            # Filter for target 'y' or 'target_bmi'
            df_y = df[df['target'].isin(['y', 'target_bmi'])].copy()

            if not df_y.empty:
                # Add metadata columns
                for i, (key, value) in enumerate(params.items()):
                    df_y.insert(i, key, value)
                
                all_data.append(df_y)
    
    if not all_data:
        print(f"No data found in {base_dir}.")
        return
    
    aggregated_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by age, r2, complexity
    sort_cols = ['age', 'r2_weight', 'complexity_lambda', 'r2', 'complexity']
    sort_cols = [c for c in sort_cols if c in aggregated_df.columns]
    
    ascending = [True] * len(sort_cols)
    if 'r2' in sort_cols:
        r2_idx = sort_cols.index('r2')
        ascending[r2_idx] = False
    
    aggregated_df = aggregated_df.sort_values(by=sort_cols, ascending=ascending)
    
    # Rename columns
    aggregated_df = aggregated_df.rename(columns={
        'r2_weight': 'pareto r2 weight',
        'complexity_lambda': 'lambda for complexity'
    })
    
    output_path = os.path.join(base_dir, 'aggregated_age_results_bmi.csv')
    aggregated_df.to_csv(output_path, index=False)
    print(f"Aggregated results saved to {output_path}")

def calculate_age_performance_metrics(base_dir):
    all_metrics = []

    for root, dirs, files in os.walk(base_dir):
        if 'relationships.csv' in files:
            folder_name = os.path.basename(root)
            params = parse_age_folder_name(folder_name)
            age = params.get('age')
            if age is None:
                continue
                
            # Load data for this specific age
            _, X, y_target = load_agg_data(age=age)
            y_target = y_target.values.flatten()
            
            file_path = os.path.join(root, 'relationships.csv')
            rel_df = pd.read_csv(file_path)
            
            # Find formula for 'y' or 'target_bmi'
            y_rel = rel_df[rel_df['target'].isin(['y', 'target_bmi'])]
            if y_rel.empty:
                continue
            
            formula_str = y_rel.iloc[0]['formula']
            involved = [s.strip() for s in y_rel.iloc[0]['involved'].split(',')]
            
            try:
                extra_mappings = {'cond': sympy_cond}
                modules = [extra_mappings, 'numpy']
                
                symbols = {s: sp.Symbol(s) for s in involved}
                from sympy.parsing.sympy_parser import parse_expr
                expr = parse_expr(formula_str, local_dict=symbols)
                
                if 'cond' in formula_str:
                    f_cond = sp.Function('cond')
                    expr = expr.replace(f_cond, sympy_cond)

                func = sp.lambdify([symbols[s] for s in involved], expr, modules=modules)
                
                args = [X[s].values for s in involved]
                y_pred = func(*args)
                
                y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
                
                r2 = max(0, min(1, r2_score(y_target, y_pred)))
                mae = mean_absolute_error(y_target, y_pred)
                rmse = np.sqrt(mean_squared_error(y_target, y_pred))
                
                metric_row = params.copy()
                metric_row.update({
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'complexity': y_rel.iloc[0]['complexity'],
                    'formula': formula_str
                })
                all_metrics.append(metric_row)
                    
            except Exception as e:
                print(f"Error evaluating formula in {folder_name}: {e}")

    if not all_metrics:
        print("No metrics calculated.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    
    metrics_df = metrics_df.rename(columns={
        'r2_weight': 'pareto r2 weight',
        'complexity_lambda': 'lambda for complexity'
    })
    
    sort_cols = ['pareto r2 weight', 'lambda for complexity', 'age']
    sort_cols = [c for c in sort_cols if c in metrics_df.columns]
    metrics_df = metrics_df.sort_values(by=sort_cols)
    
    output_path = os.path.join(base_dir, 'performance_metrics_age_bmi.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"Performance metrics saved to {output_path}")

def plot_age_performance_metrics_interactive(base_dir):
    csv_path = os.path.join(base_dir, 'performance_metrics_age_bmi.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    param_cols = [
        'parsimony', 'population', 'pop_size', 'parsimony_scaling', 
        'prune_start', 'prune_ramp', 'prune_max', 
        'pareto r2 weight', 'lambda for complexity'
    ]
    
    param_cols = [c for c in param_cols if c in df.columns]
    grouped = df.groupby(param_cols)
    
    metrics = ['r2', 'mae', 'rmse', 'complexity']
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=['R2 Score', 'MAE', 'Formula Complexity', 'RMSE'],
                        vertical_spacing=0.15,
                        horizontal_spacing=0.1)
    
    metric_pos = {
        'r2': (1, 1),
        'mae': (1, 2),
        'complexity': (2, 1),
        'rmse': (2, 2)
    }
    
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    for color_idx, (params, group) in enumerate(grouped):
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
            show_legend = (i == 0)
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
                    hovertemplate=f"Age: %{{x}}<br>{metric.upper()}: %{{y:.4f}}<br>Model: {label}<br>Formula: {group.iloc[0]['formula']}<extra></extra>"
                ),
                row=row, col=col
            )
            
    for r, c in [(1,1), (1,2), (2,1), (2,2)]:
        # Sync x-axes
        fig.update_xaxes(
            title_text="Age", 
            row=r, col=c,
            matches='x'
        )
    
    fig.update_yaxes(title_text="R2 Score", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    fig.update_yaxes(title_text="Complexity", row=2, col=1)
    fig.update_yaxes(title_text="RMSE", row=2, col=2)

    fig.update_layout(
        height=900,
        width=1800,
        title_text="Age-Specific Models: Performance Metrics by Age",
        showlegend=True,
    )
    
    output_html = os.path.join(base_dir, 'performance_metrics_age_plot.html')
    fig.write_html(output_html)
    print(f"Interactive plot saved to {output_html}")

def calculate_baseline_longitudinal_metrics(base_dir):
    all_metrics = []
    for root, dirs, files in os.walk(base_dir):
        if 'predictions.csv' in files:
            model_name = os.path.basename(root)
            file_path = os.path.join(root, 'predictions.csv')
            df = pd.read_csv(file_path)
            
            # Load formula if it exists (for KAN)
            kan_formula = ""
            if model_name.lower() == 'kan':
                formula_files = [f for f in files if f.startswith('formula_fold')]
                if formula_files:
                    # Just use the first one or combine them if needed. For now, take first.
                    with open(os.path.join(root, sorted(formula_files)[0]), 'r') as f:
                        kan_formula = f.read().strip()

            # Calculate metrics per age
            ages = df['age'].unique()
            for age in sorted(ages):
                mask = (df['age'] == age)
                df_age = df[mask]
                
                y_true = df_age['target_bmi']
                
                # Standard prediction
                y_pred = df_age['pred_bmi']
                r2 = max(0, min(1, r2_score(y_true, y_pred)))
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                
                all_metrics.append({
                    'model': model_name,
                    'age': age,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'formula': kan_formula if model_name.lower() == 'kan' else "",
                    'complexity': calculate_complexity(kan_formula) if model_name.lower() == 'kan' else 0
                })

                # KAN Symbolic prediction
                if model_name.lower() == 'kan' and 'pred_bmi_sym' in df_age.columns:
                    y_pred_sym = df_age['pred_bmi_sym']
                    r2_sym = max(0, min(1, r2_score(y_true, y_pred_sym)))
                    mae_sym = mean_absolute_error(y_true, y_pred_sym)
                    rmse_sym = np.sqrt(mean_squared_error(y_true, y_pred_sym))
                    all_metrics.append({
                        'model': 'KANsym',
                        'age': age,
                        'r2': r2_sym,
                        'mae': mae_sym,
                        'rmse': rmse_sym,
                        'formula': kan_formula,
                        'complexity': calculate_complexity(kan_formula)
                    })
    
    if not all_metrics:
        print(f"No baseline longitudinal metrics found in {base_dir}")
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    output_path = os.path.join(base_dir, 'baseline_longitudinal_metrics.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"Baseline longitudinal metrics saved to {output_path}")

def calculate_baseline_age_metrics(base_dir):
    all_metrics = []
    for root, dirs, files in os.walk(base_dir):
        if 'predictions.csv' in files:
            folder_name = os.path.basename(root)
            # Example folder name: yr10_elasticnet
            match = re.search(r'yr(\d+)_([^_]+)', folder_name)
            if match:
                age = int(match.group(1))
                model_name = match.group(2)
            else:
                continue
                
            file_path = os.path.join(root, 'predictions.csv')
            df = pd.read_csv(file_path)
            
            # Load formula if it exists (for KAN)
            kan_formula = ""
            if model_name.lower() == 'kan':
                formula_files = [f for f in files if f.startswith('formula_fold')]
                if formula_files:
                    with open(os.path.join(root, sorted(formula_files)[0]), 'r') as f:
                        kan_formula = f.read().strip()

            y_true = df['target_bmi']
            
            # Standard prediction
            y_pred = df['pred_bmi']
            r2 = max(0, min(1, r2_score(y_true, y_pred)))
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            all_metrics.append({
                'model': model_name,
                'age': age,
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'formula': kan_formula if model_name.lower() == 'kan' else "",
                'complexity': calculate_complexity(kan_formula) if model_name.lower() == 'kan' else 0
            })

            # KAN Symbolic prediction
            if model_name.lower() == 'kan' and 'pred_bmi_sym' in df.columns:
                y_pred_sym = df['pred_bmi_sym']
                r2_sym = max(0, min(1, r2_score(y_true, y_pred_sym)))
                mae_sym = mean_absolute_error(y_true, y_pred_sym)
                rmse_sym = np.sqrt(mean_squared_error(y_true, y_pred_sym))
                all_metrics.append({
                    'model': 'KANsym',
                    'age': age,
                    'r2': r2_sym,
                    'mae': mae_sym,
                    'rmse': rmse_sym,
                    'formula': kan_formula,
                    'complexity': calculate_complexity(kan_formula)
                })
            
    if not all_metrics:
        print(f"No baseline age metrics found in {base_dir}")
        return
    
    metrics_df = pd.DataFrame(all_metrics)
    output_path = os.path.join(base_dir, 'baseline_age_metrics.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"Baseline age metrics saved to {output_path}")

def compare_longitudinal_vs_age(long_dir, age_dir, baseline_long_dir=None, baseline_age_dir=None):
    long_csv = os.path.join(long_dir, 'performance_metrics_bmi.csv')
    age_csv = os.path.join(age_dir, 'performance_metrics_age_bmi.csv')
    
    if not os.path.exists(long_csv) or not os.path.exists(age_csv):
        print(f"Required CSVs for comparison not found in {long_dir} or {age_dir}.")
        return

    df_long = pd.read_csv(long_csv)
    df_age = pd.read_csv(age_csv)
    
    # 1. Best longitudinal model (highest overall_r2)
    best_long_model_idx = df_long['overall_r2'].idxmax()
    # Identify the best model parameters
    param_cols = [
        'parsimony', 'population', 'pop_size', 'parsimony_scaling', 
        'prune_start', 'prune_ramp', 'prune_max', 'pareto r2 weight', 'lambda for complexity'
    ]
    best_params_series = df_long.loc[best_long_model_idx, [c for c in param_cols if c in df_long.columns]]
    
    # Filter df_long to get all age metrics for this specific best model
    query_parts = []
    for col, val in best_params_series.items():
        if isinstance(val, str):
            query_parts.append(f"`{col}` == '{val}'")
        else:
            query_parts.append(f"`{col}` == {val}")
    
    best_long_metrics = df_long.query(" & ".join(query_parts)).sort_values('age')

    # 2. Best age model for EACH age (highest r2 for that age)
    best_age_metrics = df_age.loc[df_age.groupby('age')['r2'].idxmax()].sort_values('age')

    # Plot comparison
    titles = ['R2 Score', 'Formula Complexity', 'MAE', 'RMSE', 'Feature Importance (Baseline Models)']
    fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=titles,
        specs=[[{}, {}], [{}, {}], [{"colspan": 2}, None]],
        vertical_spacing=0.08
    )
    
    # Grid positions: R2 (1,1), Complexity (1,2), MAE (2,1), RMSE (2,2)
    metric_pos = {
        'r2': (1, 1),
        'complexity': (1, 2),
        'mae': (2, 1),
        'rmse': (2, 2)
    }
    
    metrics = ['r2', 'complexity', 'mae', 'rmse']
    
    # Define colors for baselines
    baseline_colors = {
        'kan': 'green',
        'kansym': 'lightgreen',
        'elasticnet': 'orange',
        'erf': 'purple',
        'xgboost': 'brown',
        'mlp': 'pink'
    }

    for i, metric in enumerate(metrics):
        row, col = metric_pos[metric]
        # Longitudinal line
        fig.add_trace(go.Scatter(
            x=best_long_metrics['age'], 
            y=best_long_metrics[metric],
            mode='lines+markers',
            name='Best DeepPySR Longitudinal',
            line=dict(color='blue', width=3),
            showlegend=(i == 0),
            hovertemplate="Age: %{x}<br>" + metric.upper() + ": %{y:.4f}<br>Model: Best DeepPySR Longitudinal<br>Formula: %{text}<extra></extra>",
            text=best_long_metrics['formula']
        ), row=row, col=col)
        
        # Age-specific line
        fig.add_trace(go.Scatter(
            x=best_age_metrics['age'], 
            y=best_age_metrics[metric],
            mode='lines+markers',
            name='Best DeepPySR Age-Specific',
            line=dict(color='red', width=3, dash='dash'),
            showlegend=(i == 0),
            hovertemplate="Age: %{x}<br>" + metric.upper() + ": %{y:.4f}<br>Model: Best DeepPySR Age-Specific<br>Formula: %{text}<extra></extra>",
            text=best_age_metrics['formula']
        ), row=row, col=col)

        # Add Baselines
        if baseline_long_dir:
            bl_long_csv = os.path.join(baseline_long_dir, 'baseline_longitudinal_metrics.csv')
            if os.path.exists(bl_long_csv):
                df_bl_long = pd.read_csv(bl_long_csv)
                for model in df_bl_long['model'].unique():
                    # For complexity plot, only show models that have complexity (KAN, KANsym)
                    if metric == 'complexity' and model.lower() not in ['kan', 'kansym']:
                        continue
                        
                    model_df = df_bl_long[df_bl_long['model'] == model].sort_values('age')
                    fig.add_trace(go.Scatter(
                        x=model_df['age'],
                        y=model_df[metric],
                        mode='lines+markers',
                        name=f'Longitudinal {model.upper()}',
                        line=dict(color=baseline_colors.get(model.lower(), 'grey'), width=1.5),
                        showlegend=(i == 0),
                        legendgroup=f'bl_long_{model}',
                        opacity=0.7,
                        hovertemplate="Age: %{x}<br>" + metric.upper() + ": %{y:.4f}<br>Model: Longitudinal " + model.upper() + "<br>Formula: %{text}<extra></extra>",
                        text=model_df['formula'] if 'formula' in model_df.columns else [""] * len(model_df)
                    ), row=row, col=col)

        if baseline_age_dir:
            bl_age_csv = os.path.join(baseline_age_dir, 'baseline_age_metrics.csv')
            if os.path.exists(bl_age_csv):
                df_bl_age = pd.read_csv(bl_age_csv)
                for model in df_bl_age['model'].unique():
                    # For complexity plot, only show models that have complexity (KAN, KANsym)
                    if metric == 'complexity' and model.lower() not in ['kan', 'kansym']:
                        continue
                        
                    model_df = df_bl_age[df_bl_age['model'] == model].sort_values('age')
                    fig.add_trace(go.Scatter(
                        x=model_df['age'],
                        y=model_df[metric],
                        mode='lines+markers',
                        name=f'Age-Specific {model.upper()}',
                        line=dict(color=baseline_colors.get(model.lower(), 'grey'), width=1.5, dash='dash'),
                        showlegend=(i == 0),
                        legendgroup=f'bl_age_{model}',
                        opacity=0.7,
                        hovertemplate="Age: %{x}<br>" + metric.upper() + ": %{y:.4f}<br>Model: Age-Specific " + model.upper() + "<br>Formula: %{text}<extra></extra>",
                        text=model_df['formula'] if 'formula' in model_df.columns else [""] * len(model_df)
                    ), row=row, col=col)
        
        fig.update_xaxes(title_text="Age", row=row, col=col)
        y_title = metric.upper() if metric != 'complexity' else 'Complexity'
        
        # For complexity, make y-axis log scale
        if metric == 'complexity':
            fig.update_yaxes(title_text=y_title, row=row, col=col, type='log')
        else:
            fig.update_yaxes(title_text=y_title, row=row, col=col)

    # Add Feature Importance Plot at the bottom
    all_importances = []
    
    # 1. Load Longitudinal Baseline Importances
    if baseline_long_dir and os.path.exists(baseline_long_dir):
        for model_folder in os.listdir(baseline_long_dir):
            imp_path = os.path.join(baseline_long_dir, model_folder, 'feature_importances.csv')
            if os.path.exists(imp_path):
                imp_df = pd.read_csv(imp_path)
                # Take absolute value for feature importance
                imp_df['importance'] = imp_df['importance'].abs()
                # Normalize to percentage
                total_imp = imp_df['importance'].sum()
                if total_imp > 0:
                    imp_df['importance'] = (imp_df['importance'] / total_imp) * 100
                imp_df['model'] = f"Longitudinal {model_folder.upper()}"
                all_importances.append(imp_df)
    
    # 2. Load Age-Specific Baseline Importances
    if baseline_age_dir and os.path.exists(baseline_age_dir):
        age_imps = []
        for folder in os.listdir(baseline_age_dir):
            if folder.startswith('yr') and '_' in folder:
                imp_path = os.path.join(baseline_age_dir, folder, 'feature_importances.csv')
                if os.path.exists(imp_path):
                    parts = folder.split('_')
                    model_name = parts[1].upper()
                    imp_df = pd.read_csv(imp_path)
                    # Take absolute value for feature importance
                    imp_df['importance'] = imp_df['importance'].abs()
                    # Normalize to percentage for this specific age model
                    total_imp = imp_df['importance'].sum()
                    if total_imp > 0:
                        imp_df['importance'] = (imp_df['importance'] / total_imp) * 100
                    imp_df['model_type'] = model_name
                    age_imps.append(imp_df)
        
        if age_imps:
            df_age_imps = pd.concat(age_imps)
            # Average across ages for each model type
            avg_age_imps = df_age_imps.groupby(['model_type', 'feature'])['importance'].mean().reset_index()
            # Normalize the average again just in case, though it should already be close to 100
            for model_type in avg_age_imps['model_type'].unique():
                model_df = avg_age_imps[avg_age_imps['model_type'] == model_type]
                total_imp = model_df['importance'].sum()
                if total_imp > 0:
                    avg_age_imps.loc[avg_age_imps['model_type'] == model_type, 'importance'] = (model_df['importance'] / total_imp) * 100
                
            avg_age_imps['model'] = "Age-Specific " + avg_age_imps['model_type']
            all_importances.append(avg_age_imps[['feature', 'importance', 'model']])
    
    if all_importances:
        combined_imp = pd.concat(all_importances)
        # Find the most important features based on the average across models
        avg_imp = combined_imp.groupby('feature')['importance'].mean().sort_values(ascending=False).index.tolist()
        
        for model in combined_imp['model'].unique():
            model_imp = combined_imp[combined_imp['model'] == model].set_index('feature').reindex(avg_imp).reset_index()
            
            # Use same colors but different patterns or dash for age-specific if needed, 
            # but for now let's just use the colors.
            base_model_name = model.replace("Longitudinal ", "").replace("Age-Specific ", "").lower()
            color = baseline_colors.get(base_model_name, 'grey')
            
            # Distinguish between Longitudinal and Age-Specific using opacity or pattern
            is_age_specific = "Age-Specific" in model
            
            fig.add_trace(go.Bar(
                x=model_imp['feature'],
                y=model_imp['importance'],
                name=model,
                marker_color=color,
                marker_pattern_shape="x" if is_age_specific else "",
                showlegend=True,
                legendgroup=f'imp_{model}',
                hovertemplate="Feature: %{x}<br>Importance: %{y:.2f}%<br>Model: " + model + "<extra></extra>"
            ), row=3, col=1)
        
        fig.update_xaxes(
            title_text="Features", 
            row=3, col=1,
            range=[-0.5, 14.5]
        )
        fig.update_yaxes(title_text="Importance (%)", row=3, col=1)

    fig.update_layout(
        height=1400,
        width=2000,
        title_text="Comparison: Best Longitudinal vs Best Age-Specific Models",
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02
        ),
        xaxis5=dict(
            rangeslider=dict(visible=True),
            rangeslider_thickness=0.05
        )
    )
    
    output_path = os.path.join(os.path.dirname(long_dir), 'longitudinal_vs_age_comparison.html')
    fig.write_html(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Process results_bmi/deeppysr_old
    # deeppysr_old_dir = os.path.join(base_directory, 'results_bmi', 'deeppysr_old')
    # if os.path.exists(deeppysr_old_dir):
    #     aggregate_results_bmi(deeppysr_old_dir)
        
    # Process results_bmi/deeppysr_longitudinal
    deeppysr_longitudinal_dir = os.path.join(base_directory, 'results_bmi', 'deeppysr_longitudinal')
    if os.path.exists(deeppysr_longitudinal_dir):
        aggregate_agg_results(deeppysr_longitudinal_dir)
        calculate_performance_metrics(deeppysr_longitudinal_dir)
        plot_performance_metrics_interactive(deeppysr_longitudinal_dir)

    # Process results_bmi/deeppysr_age
    deeppysr_age_dir = os.path.join(base_directory, 'results_bmi', 'deeppysr_age')
    if os.path.exists(deeppysr_age_dir):
        aggregate_age_results(deeppysr_age_dir)
        calculate_age_performance_metrics(deeppysr_age_dir)
        plot_age_performance_metrics_interactive(deeppysr_age_dir)

    # Process baseline models
    baseline_longitudinal_dir = os.path.join(base_directory, 'results_bmi', 'baseline_longitudinal')
    if os.path.exists(baseline_longitudinal_dir):
        calculate_baseline_longitudinal_metrics(baseline_longitudinal_dir)
    
    baseline_age_dir = os.path.join(base_directory, 'results_bmi', 'baseline_age')
    if os.path.exists(baseline_age_dir):
        calculate_baseline_age_metrics(baseline_age_dir)

    # Comparison
    if os.path.exists(deeppysr_longitudinal_dir) and os.path.exists(deeppysr_age_dir):
        compare_longitudinal_vs_age(deeppysr_longitudinal_dir, deeppysr_age_dir, 
                                    baseline_longitudinal_dir, baseline_age_dir)


