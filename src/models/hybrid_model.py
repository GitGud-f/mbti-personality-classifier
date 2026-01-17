import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from src.utils.logger import ExperimentLogger

class HybridModelRunner:
    def __init__(self):
        self.logger = ExperimentLogger()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.scaler = StandardScaler()

    def train(self, data_path, experiment_name="Hybrid_Syntax"):
        print(f"--- Starting Hybrid Experiment: {experiment_name} ---")
        df = pd.read_csv(data_path)
        df = df.dropna()

        # 1. Prepare Text Features (Sparse)
        print("Vectorizing Text...")
        X_text = self.vectorizer.fit_transform(df['clean_posts'])

        # 2. Prepare Syntax Features (Dense)
        # Select only the numeric columns we created in Step 1
        syntax_cols = [
            'noun_ratio', 'verb_ratio', 'adj_ratio', 'pron_ratio', 
            'ner_person_count', 'ner_org_count', 'avg_sentence_length'
        ]
        print(f"Scaling Syntax Features: {syntax_cols}")
        X_syntax = self.scaler.fit_transform(df[syntax_cols])

        # 3. Combine them horizontally
        # Result is a sparse matrix: [ TFIDF_Vectors | Syntax_Numbers ]
        X_combined = sp.hstack([X_text, X_syntax])

        targets = ['IE', 'NS', 'FT', 'JP']
        
        for target in targets:
            y = df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train
            model = LogisticRegression(class_weight='balanced', max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Log
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='macro'),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'recall': recall_score(y_test, y_pred, average='macro')
            }
            
            self.logger.log(experiment_name, "LogReg+Syntax", target, metrics)

if __name__ == "__main__":
    runner = HybridModelRunner()
    runner.train("data/processed/mbti_with_syntax.csv")