import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_performance_evolution(metric='f1_macro'):
    log_path = "experiments/results.csv"
    
    if not os.path.exists(log_path):
        print("No results found yet.")
        return

    df = pd.read_csv(log_path)
    
    if df.empty:
        print("Log file is empty.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    
    chart = sns.barplot(
        data=df, 
        x="experiment_name", 
        y=metric, 
        hue="target",
        palette="viridis"
    )

    plt.title(f"Model Performance Evolution ({metric})", fontsize=16)
    plt.xlabel("Preprocessing Stage / Model", fontsize=12)
    plt.ylabel(f"Score ({metric})", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Personality Axis")
    plt.ylim(0, 1.0) # F1 is between 0 and 1

    # Save the plot
    output_path = f"reports/figures/performance_evolution_{metric}.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_performance_evolution()