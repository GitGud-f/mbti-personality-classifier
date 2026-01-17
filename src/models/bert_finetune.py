import pandas as pd
import numpy as np
import torch
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.utils.logger import ExperimentLogger

# Disable heavy W&B logging for this mini-project to keep things clean
os.environ["WANDB_DISABLED"] = "true"

class BertFineTuner:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger = ExperimentLogger()
        
        # Check for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- Running BERT on {self.device.upper()} ---")

    def compute_metrics(self, eval_pred):
        """Helper to calculate metrics during training"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='macro'),
            'precision': precision_score(labels, predictions, average='macro'),
            'recall': recall_score(labels, predictions, average='macro')
        }

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=256 # Limit to 256 to save memory/time
        )

    def train_target(self, df, target_col, experiment_name):
        print(f"\nFine-Tuning BERT for Target: {target_col}")
        
        # 1. Prepare Data for Hugging Face
        # We need a 'text' column and a 'label' column
        hf_df = df[['clean_posts', target_col]].copy()
        hf_df.columns = ['text', 'label']
        
        # Convert to Hugging Face Dataset Object
        dataset = Dataset.from_pandas(hf_df)
        
        # Split
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        # Tokenize
        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)

        # 2. Load Model (Binary Classification -> num_labels=2)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        ).to(self.device)

        # 3. Define Training Arguments
        # Top Student Note: These hyperparameters are standard for Fine-Tuning
        training_args = TrainingArguments(
            output_dir=f"models/bert_checkpoints/{target_col}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,  # Low LR to prevent destroying pre-trained weights
            per_device_train_batch_size=16, # Increase to 32 if you have a big GPU
            per_device_eval_batch_size=16,
            num_train_epochs=3,  # 3 Epochs is usually the sweet spot
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir='logs',
            logging_steps=50,
        )

        # 4. Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            compute_metrics=self.compute_metrics,
        )

        # 5. Train
        trainer.train()

        # 6. Final Evaluation
        eval_results = trainer.evaluate()
        
        # 7. Log Results
        metrics = {
            'accuracy': eval_results['eval_accuracy'],
            'f1': eval_results['eval_f1'],
            'precision': eval_results['eval_precision'],
            'recall': eval_results['eval_recall']
        }
        
        self.logger.log(experiment_name, "DistilBERT_FineTune", target_col, metrics)
        
        # Optional: Save the specific fine-tuned model
        # model.save_pretrained(f"models/bert_final_{target_col}")

    def run(self, input_path="data/interim/mbti_cleaned_basic.csv"):
        # We use the basic cleaning because BERT has its own tokenizer 
        # that handles sub-words. We don't want stemmed text here.
        df = pd.read_csv(input_path).dropna(subset=['clean_posts'])
        
        # Run for all 4 targets
        targets = ['IE', 'NS', 'FT', 'JP']
        for target in targets:
            self.train_target(df, target, experiment_name="08_BERT_FineTuning")

if __name__ == "__main__":
    tuner = BertFineTuner()
    tuner.run()