import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from src.utils.logger import ExperimentLogger

class Word2VecRunner:
    def __init__(self):
        self.logger = ExperimentLogger()
        self.scaler = StandardScaler()

    def train(self, data_path, experiment_name="Word2Vec_SelfTrained"):
        print(f"--- Starting Word2Vec Experiment: {experiment_name} ---")
        df = pd.read_csv(data_path).dropna()

        # Select Feature Columns (w2v_0 to w2v_99)
        feature_cols = [c for c in df.columns if c.startswith('w2v_')]
        X = df[feature_cols]
        
        # Scale
        X = self.scaler.fit_transform(X)

        targets = ['IE', 'NS', 'FT', 'JP']
        
        for target in targets:
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            model = LogisticRegression(class_weight='balanced', max_iter=2000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='macro'),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'recall': recall_score(y_test, y_pred, average='macro')
            }
            
            self.logger.log(experiment_name, "LogReg+W2V_Self", target, metrics)

if __name__ == "__main__":
    runner = Word2VecRunner()
    runner.train("data/processed/mbti_word2vec_own.csv")