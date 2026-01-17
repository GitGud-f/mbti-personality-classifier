import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
from collections import Counter

class SyntaxFeatureExtractor:
    def __init__(self):
        # We need the parser and ner enabled now
        print("Loading Spacy (with Parser and NER)...")
        self.nlp = spacy.load("en_core_web_sm")

    def get_syntax_features(self, text):
        """
        Parses text and returns a dictionary of syntactic features.
        """
        doc = self.nlp(text)
        
        # 1. Basic Counts
        num_tokens = len(doc)
        if num_tokens == 0: return {}
        
        # 2. POS Tagging Ratios (Normalized by length)
        # Universal POS tags: https://universaldependencies.org/u/pos/
        pos_counts = Counter([token.pos_ for token in doc])
        
        features = {
            'noun_ratio': pos_counts.get('NOUN', 0) / num_tokens,
            'verb_ratio': pos_counts.get('VERB', 0) / num_tokens,
            'adj_ratio': pos_counts.get('ADJ', 0) / num_tokens,
            'pron_ratio': pos_counts.get('PRON', 0) / num_tokens,
        }

        # 3. Named Entity Recognition (NER) counts
        # Are they talking about People (PERSON), Organizations (ORG), or Places (GPE)?
        ent_counts = Counter([ent.label_ for ent in doc.ents])
        features['ner_person_count'] = ent_counts.get('PERSON', 0)
        features['ner_org_count'] = ent_counts.get('ORG', 0)

        # 4. Syntactical Complexity (Approximation)
        # Average length of sentences in the post
        sents = list(doc.sents)
        avg_sent_len = np.mean([len(sent) for sent in sents]) if sents else 0
        features['avg_sentence_length'] = avg_sent_len

        return features

    def run(self, input_path="data/interim/mbti_lemmatized.csv"):
        print("Extracting Syntactical Features (this takes time)...")
        df = pd.read_csv(input_path)
        

        
        tqdm.pandas()
        
        # We run this on the 'clean_posts' (lemmatized or basic)
        feature_dicts = df['clean_posts'].progress_apply(self.get_syntax_features)
        
        # Convert list of dicts to DataFrame
        feat_df = pd.DataFrame(feature_dicts.tolist())
        
        # Combine with original DF (we need the Targets IE/NS...)
        result_df = pd.concat([df, feat_df], axis=1)
        
        # Fill NaNs (if any division by zero occurred)
        result_df = result_df.fillna(0)
        
        output_path = "data/processed/mbti_with_syntax.csv"
        result_df.to_csv(output_path, index=False)
        print(f"Saved syntax-enriched data to {output_path}")
        print("New Features:", list(feat_df.columns))

if __name__ == "__main__":
    extractor = SyntaxFeatureExtractor()
    extractor.run()