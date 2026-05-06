import os
import sys
import pandas as pd
import numpy as np

def run_deep_analysis(X, y, model_params_dict, output_root, name='Analysis', n_iterations=500, n_layers=3):
    """
    Run deep analysis using DeepPySRRegressor with provided parameters.
    Trains on the entire dataset provided.
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    from deeppysr import DeepPySR

    results = []

    for model_display_name, params in model_params_dict.items():
        print(f"  Training {model_display_name} for {name}...")
        
        # Merge or set requested parameters
        model_params = params.copy() if params else {}
        model_params['max_layers'] = n_layers
        model_params['stopping_score'] = 0.01
        if 'niterations' in model_params:
            # Prefer n_iterations if it exists in the provider
            model_params['niterations'] = n_iterations

        else:
            # Default to n_iterations for DeepPySRRegressor
            model_params['niterations'] = n_iterations

        model_params['output_dir'] = output_root
        model_params['warm_start'] = True
        
        # Initialize and fit
        regressor = DeepPySR(**model_params)
        regressor.fit(X, y)
        
        # Plot relationships using deeppysr's built-in plot functions
        try:
            print(f"    Generating plots for {model_display_name}...")
            regressor.plot(filename=os.path.join(output_root, "hierarchy.png"))
            regressor.plot_circle(filename=os.path.join(output_root, "circle.png"))
            regressor.save_relationships(filename=os.path.join(output_root, "relationships.csv"))
        except Exception as e:
            print(f"    Error generating plots: {e}")

    return results

def get_best_interpretable_params(metrics_df, model_type='Interpretable DeepPySR'):
    """
    Extract best parameters for the specified model type from metrics dataframe.
    """
    from convergence_utils import parse_model_string
    
    filtered_df = metrics_df[metrics_df['display_model'] == model_type]
    models = {}
    for _, row in filtered_df.iterrows():
        if row['display_model'] not in models:
            models[row['display_model']] = parse_model_string(row['model'])
    return models
