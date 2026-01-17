import pandas as pd
from tqdm import tqdm
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.utils.logger import ExperimentLogger

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.pipeline import Pipeline

class BinaryBaselineRunner:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.vectorizer = TfidfVectorizer(max_features=5000) 
        self.models = {}
        self.logger = ExperimentLogger()

    def load_interim_data(self):
        return pd.read_csv("data/interim/mbti_cleaned_basic.csv")

    def train(self, data_path, experiment_name="Baseline_Raw"):
        print(f"--- Starting Experiment: {experiment_name} ---")
         
        df = pd.read_csv(data_path)
        
        df = df.dropna(subset=['clean_posts'])
        df = df[df['clean_posts'] != ""]
        X = df['clean_posts']
        
        targets = ['IE', 'NS', 'FT', 'JP']
        
        results = {}

        progress_bar = tqdm(targets, desc="Training Classifiers")
         
        for target in progress_bar:
            print(f"\nTraining Model for Target: {target}...")
            y = df[target]
            

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['data']['test_size'], 
                random_state=self.config['data']['random_state'],
                stratify=y 
            )
            
            clf = LogisticRegression(
                class_weight='balanced', 
                max_iter=1000, 
                verbose=1,  
                n_jobs=-1   
            )
            
            pipeline = Pipeline([
                ('vect', self.vectorizer),
                ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
            ], verbose=True)
            
            pipeline.fit(X_train, y_train)
            

            y_pred = pipeline.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='macro'),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'recall': recall_score(y_test, y_pred, average='macro')
            }


            score = f1_score(y_test, y_pred, average='macro')
            results[target] = score
            
            print(f"Classification Report for {target}:")
            print(classification_report(y_test, y_pred))
            
            joblib.dump(pipeline, f"models/baseline_{target}.pkl")
            self.models[target] = pipeline
            
            self.logger.log(
                experiment_name=experiment_name,
                model_type="LogisticRegression",
                target=target,
                metrics=metrics
            )
        print("\n=== FINAL BASELINE RESULTS (F1-Macro) ===")
        for t, s in results.items():
            print(f"{t}: {s:.4f}")

if __name__ == "__main__":
    runner = BinaryBaselineRunner()
    runner.train()