import pandas as pd
import spacy
import nltk
from nltk.stem import PorterStemmer
from tqdm import tqdm # For progress bars

class MorphologyPreprocessor:
    def __init__(self):
        # Load Spacy for Lemmatization (disable NER/Parser for speed)
        print("Loading Spacy model...")
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        
        # Load NLTK for Stemming
        self.stemmer = PorterStemmer()

    def load_data(self, input_path="data/interim/mbti_cleaned_basic.csv"):
        return pd.read_csv(input_path)

    def apply_stemming(self, text):
        """
        Splits text and applies Porter Stemmer.
        Fast, but aggressive (e.g., 'university' -> 'univers').
        """
        tokens = text.split()
        stemmed_tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(stemmed_tokens)

    def apply_lemmatization(self, text):
        """
        Uses Spacy to find the lemma.
        Slower, but accurate (e.g., 'universities' -> 'university').
        """
        doc = self.nlp(text)
        # return lemma if it's not a pronoun (pronouns often confuse models in MBTI)
        lemmas = [token.lemma_ for token in doc if not token.is_stop]
        return " ".join(lemmas)

    def apply_pos_filter(self, text, allowed_pos={'ADJ', 'NOUN', 'VERB'}):
        """
        Keeps only specific parts of speech.
        Good for filtering out noise words.
        """
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if token.pos_ in allowed_pos]
        return " ".join(tokens)

    def run(self, method='lemmatization'):
        print(f"Starting Morphological Analysis: {method}...")
        df = self.load_data()
        
        # We use tqdm to show a progress bar because this can be slow
        tqdm.pandas()

        if method == 'stemming':
            df['clean_posts'] = df['clean_posts'].progress_apply(self.apply_stemming)
            output_path = "data/interim/mbti_stemmed.csv"
            
        elif method == 'lemmatization':
            # Increase batch size for Spacy speed
            df['clean_posts'] = df['clean_posts'].progress_apply(self.apply_lemmatization)
            output_path = "data/interim/mbti_lemmatized.csv"
            
        elif method == 'pos_nouns_adj':
            df['clean_posts'] = df['clean_posts'].progress_apply(
                lambda x: self.apply_pos_filter(x, allowed_pos={'NOUN', 'ADJ'})
            )
            output_path = "data/interim/mbti_pos_filtered.csv"

        df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    # You can change this to 'stemming' or 'pos_nouns_adj' to generate different datasets
    processor = MorphologyPreprocessor()
    
    # Let's generate Lemmatized data first (usually the best balance)
    processor.run(method='lemmatization')