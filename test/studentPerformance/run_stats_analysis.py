import os
import subprocess

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    print(f"\n" + "="*70)
    print(f"RUNNING {script_name}")
    print("="*70)
    script_path = os.path.join(current_dir, script_name)
    try:
        subprocess.run(['python', script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

if __name__ == "__main__":
    # 1. Run analysis to aggregate results and pick best models
    run_script('analysis.py')
    
    # 2. Run convergence tests
    run_script('student_convergence.py')
    
    # 3. Run deep analysis
    run_script('deep_analysis.py')
