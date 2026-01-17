import pandas as pd

def get_best_models(log_path="experiments/results.csv"):
    df = pd.read_csv(log_path)
    
    # We want to maximize F1-Macro
    targets = ['IE', 'NS', 'FT', 'JP']
    best_config = {}

    print(f"{'Target':<6} | {'Best Model':<30} | {'Score':<6}")
    print("-" * 50)

    for target in targets:
        # Filter for this target
        subset = df[df['target'] == target]
        
        # Find row with max f1_macro
        best_row = subset.loc[subset['f1_macro'].idxmax()]
        
        best_config[target] = {
            'experiment': best_row['experiment_name'],
            'model_type': best_row['model_type'],
            'score': best_row['f1_macro']
        }
        
        print(f"{target:<6} | {best_row['experiment_name']:<30} | {best_row['f1_macro']:.4f}")

    return best_config

if __name__ == "__main__":
    get_best_models()