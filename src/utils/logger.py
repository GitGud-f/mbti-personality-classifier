import pandas as pd
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, log_path="experiments/results.csv"):
        self.log_path = log_path
        self._initialize_log()

    def _initialize_log(self):
        """Creates the CSV with headers if it doesn't exist."""
        if not os.path.exists(self.log_path):
            df = pd.DataFrame(columns=[
                "timestamp", 
                "experiment_name", # e.g., 'Baseline_Raw_Tokens', 'Exp2_Lemmatization'
                "model_type",      # e.g., 'LogisticRegression', 'BERT'
                "target",          # 'IE', 'NS', 'FT', 'JP'
                "accuracy",
                "f1_macro",
                "precision_macro",
                "recall_macro"
            ])
            df.to_csv(self.log_path, index=False)

    def log(self, experiment_name, model_type, target, metrics):
        """
        Logs a single run to the CSV.
        metrics: dict containing 'accuracy', 'f1', 'precision', 'recall'
        """
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": experiment_name,
            "model_type": model_type,
            "target": target,
            "accuracy": round(metrics.get('accuracy', 0), 4),
            "f1_macro": round(metrics.get('f1', 0), 4),
            "precision_macro": round(metrics.get('precision', 0), 4),
            "recall_macro": round(metrics.get('recall', 0), 4)
        }
        
        # Append to CSV
        df = pd.DataFrame([entry])
        df.to_csv(self.log_path, mode='a', header=False, index=False)
        print(f"Logged result for {target} -> {self.log_path}")