import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_extreme_results(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set(style="whitegrid")

    # 1. R2 Results Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='problem', y='r2', hue='config_name')
    plt.title('R2 Results for Different Problems', fontsize=15)
    plt.ylabel('R2 Score', fontsize=12)
    plt.xlabel('Problem', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)  # Clip negative R2 for better visualization of good results
    plt.legend(title='Config Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extreme_r2_comparison.png'))
    plt.close()

    # 2. Time Consuming Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x='problem', y='time', hue='config_name')
    plt.title('Time Consumption for Different Problems', fontsize=15)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.xlabel('Problem', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Config Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'extreme_time_comparison.png'))
    plt.close()

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    csv_file = "test/nsga/results/all_extreme_results.csv"
    output_folder = "test/nsga/results"
    plot_extreme_results(csv_file, output_folder)
