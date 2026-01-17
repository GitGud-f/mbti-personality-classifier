import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from src.utils.logger import ExperimentLogger

class TopicModelRunner:
    def __init__(self):
        self.logger = ExperimentLogger()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        # Topics are probabilities (0 to 1), so scaling is less critical but still good practice
        self.scaler = StandardScaler() 

    def train(self, data_path, experiment_name="Hybrid_Topics"):
        print(f"--- Starting Topic Experiment: {experiment_name} ---")
        df = pd.read_csv(data_path).dropna()

        # 1. Text Features
        print("Vectorizing Text...")
        X_text = self.vectorizer.fit_transform(df['clean_posts'])

        # 2. Topic Features
        # Identify columns starting with 'topic_'
        topic_cols = [c for c in df.columns if c.startswith('topic_')]
        print(f"Using {len(topic_cols)} Topic Features...")
        
        X_topics = self.scaler.fit_transform(df[topic_cols])

        # 3. Stack
        X_combined = sp.hstack([X_text, X_topics])

        targets = ['IE', 'NS', 'FT', 'JP']
        
        for target in targets:
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, y, test_size=0.2, random_state=42, stratify=y
            )
            
            model = LogisticRegression(class_weight='balanced', max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='macro'),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'recall': recall_score(y_test, y_pred, average='macro')
            }
            
            self.logger.log(experiment_name, "LogReg+LDA", target, metrics)

if __name__ == "__main__":
    runner = TopicModelRunner()
    runner.train("data/processed/mbti_with_topics.csv")