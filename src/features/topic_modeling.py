import pandas as pd
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaMulticore
from tqdm import tqdm
import numpy as np
import pickle
import os

class TopicModeler:
    def __init__(self, num_topics=20):
        self.num_topics = num_topics
        self.dictionary = None
        self.lda_model = None

    def prepare_corpus(self, text_series):
        """
        Converts text column into Gensim Dictionary and Corpus.
        """
        print("Tokenizing for LDA...")
        # simple_preprocess handles lowercasing and basic tokenization
        processed_docs = [simple_preprocess(doc) for doc in tqdm(text_series)]
        
        print("Building Dictionary...")
        self.dictionary = corpora.Dictionary(processed_docs)
        
        # Filter extremes: remove words that appear in < 10 docs or > 50% of docs
        # This removes unique noise and common stopwords
        self.dictionary.filter_extremes(no_below=10, no_above=0.5)
        
        print("Creating Bow Corpus...")
        corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
        
        return corpus

    def train_lda(self, corpus):
        """
        Trains the LDA model.
        """
        print(f"Training LDA Model with {self.num_topics} topics...")
        # LdaMulticore uses all CPU cores for speed
        self.lda_model = LdaMulticore(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=42,
            passes=10,  # More passes = better topics, slower training
            workers=os.cpu_count() - 1
        )
        
        # Save model artifacts
        os.makedirs("models/lda", exist_ok=True)
        self.lda_model.save("models/lda/lda.model")
        self.dictionary.save("models/lda/lda.dict")
        
        # Print the topics found
        print("\n=== Discovered Topics ===")
        for idx, topic in self.lda_model.print_topics(-1):
            print(f"Topic {idx}: {topic}")

    def get_document_topics(self, corpus):
        """
        Converts the corpus into a matrix of Topic Probabilities.
        Returns: DataFrame of shape (n_docs, n_topics)
        """
        print("Extracting Topic Features for all documents...")
        topic_features = []
        
        for doc in tqdm(corpus):
            # lda_model[doc] returns a list of (topic_id, prob) tuples
            # e.g. [(0, 0.1), (3, 0.9)]
            doc_topics = self.lda_model.get_document_topics(doc, minimum_probability=0.0)
            
            # Convert to dense vector [0.1, 0, 0, 0.9, ...]
            dense_vec = [0.0] * self.num_topics
            for topic_id, prob in doc_topics:
                dense_vec[topic_id] = prob
            
            topic_features.append(dense_vec)
            
        cols = [f"topic_{i}" for i in range(self.num_topics)]
        return pd.DataFrame(topic_features, columns=cols)

    def run(self, input_path="data/interim/mbti_lemmatized.csv"):
        df = pd.read_csv(input_path).dropna(subset=['clean_posts'])
        
        corpus = self.prepare_corpus(df['clean_posts'])
        self.train_lda(corpus)
        
        topic_df = self.get_document_topics(corpus)
        
        # Concatenate with original data
        result_df = pd.concat([df.reset_index(drop=True), topic_df], axis=1)
        
        output_path = "data/processed/mbti_with_topics.csv"
        result_df.to_csv(output_path, index=False)
        print(f"Saved topic-enriched data to {output_path}")

if __name__ == "__main__":
    modeler = TopicModeler(num_topics=15) 
    modeler.run()