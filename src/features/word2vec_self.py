import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from tqdm import tqdm
import os

class Word2VecTrainer:
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def train_and_extract(self, input_path="data/interim/mbti_cleaned_basic.csv"):
        print(f"Training Word2Vec from scratch on {input_path}...")
        df = pd.read_csv(input_path).dropna(subset=['clean_posts'])

        # 1. Tokenize (List of Lists of strings)
        print("Tokenizing sentences...")
        tokenized_docs = [simple_preprocess(doc) for doc in tqdm(df['clean_posts'])]

        # 2. Train Word2Vec
        print("Training Word2Vec Model...")
        self.model = Word2Vec(
            sentences=tokenized_docs, 
            vector_size=self.vector_size, 
            window=self.window, 
            min_count=self.min_count, 
            workers=os.cpu_count(),
            sg=1  # 1 = Skip-gram (usually better for smaller datasets), 0 = CBOW
        )
        
        # Save the model for later inspection
        os.makedirs("models/word2vec", exist_ok=True)
        self.model.save("models/word2vec/mbti.w2v")
        print("Model trained and saved.")

        # 3. Generate Document Vectors (Mean Pooling)
        print("Generating Document Vectors...")
        vector_list = []
        for tokens in tqdm(tokenized_docs):
            vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
            if vectors:
                vector_list.append(np.mean(vectors, axis=0))
            else:
                vector_list.append(np.zeros(self.vector_size))

        # 4. Create DataFrame
        vector_df = pd.DataFrame(vector_list)
        vector_df.columns = [f"w2v_{i}" for i in range(self.vector_size)]

        # Combine
        result_df = pd.concat([df[['type', 'IE', 'NS', 'FT', 'JP']], vector_df], axis=1)
        
        output_path = "data/processed/mbti_word2vec_own.csv"
        result_df.to_csv(output_path, index=False)
        print(f"Saved Self-Trained Word2Vec data to {output_path}")

if __name__ == "__main__":
    trainer = Word2VecTrainer()
    trainer.train_and_extract()