import pandas as pd
import numpy as np
import gensim.downloader as api
from tqdm import tqdm
from gensim.utils import simple_preprocess

class EmbeddingExtractor:
    def __init__(self, model_name="glove-wiki-gigaword-100"):
        print(f"Loading Pre-trained Vectors: {model_name}...")
        print("Note: If this is the first run, it will download ~128MB.")
        self.vectors = api.load(model_name)
        self.vector_size = self.vectors.vector_size

    def get_document_vector(self, text):
        """
        Tokenizes text, looks up vectors, and returns the Mean Vector.
        """
        tokens = simple_preprocess(text)
        
        # Collect vectors for words that exist in GloVe
        word_vectors = [self.vectors[word] for word in tokens if word in self.vectors]
        
        if not word_vectors:
            # Return a zero vector if no words found (rare)
            return np.zeros(self.vector_size)
        
        # Calculate Mean (Average) Vector
        return np.mean(word_vectors, axis=0)

    def run(self, input_path="data/interim/mbti_cleaned_basic.csv"):
        print("Starting Vector Embedding Extraction...")
        df = pd.read_csv(input_path).dropna(subset=['clean_posts'])
        
        tqdm.pandas()
        
        # Apply the vectorizer
        # Result is a Series where each row is a numpy array
        vector_series = df['clean_posts'].progress_apply(self.get_document_vector)
        
        # Convert Series of Arrays -> DataFrame with 100 columns
        print("Expanding vectors into columns...")
        vector_df = pd.DataFrame(vector_series.tolist())
        vector_df.columns = [f"dim_{i}" for i in range(self.vector_size)]
        
        # Combine with Targets
        result_df = pd.concat([df[['type', 'IE', 'NS', 'FT', 'JP']], vector_df], axis=1)
        
        output_path = "data/processed/mbti_glove_100.csv"
        result_df.to_csv(output_path, index=False)
        print(f"Saved GloVe embeddings to {output_path}")

if __name__ == "__main__":
    extractor = EmbeddingExtractor()
    extractor.run()