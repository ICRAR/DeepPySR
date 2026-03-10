import os
import pandas as pd
import re

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

def aggregate_results(base_dir):
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
    
    output_path = os.path.join(base_dir, 'aggregated_results.csv')
    aggregated_df.to_csv(output_path, index=False)
    print(f"Aggregated results saved to {output_path}")

    # Filtered results: for each year, the row with highest r2
    # We use original column names 'r2' which is still in aggregated_df
    # but we will perform grouping.
    
    # Highest r2 per year
    filtered_df = aggregated_df.loc[aggregated_df.groupby('year')['r2'].idxmax()].sort_values(by=['year'])
    
    filtered_output_path = os.path.join(base_dir, 'filtered_results.csv')
    filtered_df.to_csv(filtered_output_path, index=False)
    print(f"Filtered results saved to {filtered_output_path}")

if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.abspath(__file__))
    aggregate_results(base_directory)
