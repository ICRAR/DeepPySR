import os
import sys
import pandas as pd

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(current_dir, ""))

# Import shared convergence utilities
sys.path.append(os.path.join(project_root, "test"))
from convergence_utils import (
    parse_model_string,
    run_convergence_comparison
)

from student_utils import load_student_data

def main():
    # Load model metrics to get best models and their parameters
    metrics_file = os.path.join(current_dir, 'student_best_models_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"ERROR: Metrics file not found at {metrics_file}")
        print("Please run analysis.py first to generate the metrics file.")
        return
        
    metrics_df = pd.read_csv(metrics_file)
    output_root = os.path.join(current_dir, './convergence_results')
    
    subjects = ['mat', 'por']
    for subject in subjects:
        print("\n" + "="*70)
        print(f"{subject.upper()} STUDENT CONVERGENCE TESTS")
        print("="*70)
        
        sub_metrics = metrics_df[metrics_df['subject'] == subject]
        sub_models = {}
        for _, row in sub_metrics.iterrows():
            if row['display_model'] in ['Best DeepPySR', 'Best PySR']:
                if row['display_model'] not in sub_models:
                    sub_models[row['display_model']] = parse_model_string(row['model'])
        
        if not sub_models:
            print(f"No Best DeepPySR or Best PySR models found for {subject}")
            continue

        df = load_student_data(subject)
        X = df.drop(columns=['G3'])
        y = df['G3']
        
        sub_output = os.path.join(output_root, subject)
        run_convergence_comparison(X, y, sub_models, sub_output, name=f'{subject.upper()} Student', task='regression')

if __name__ == "__main__":
    main()
