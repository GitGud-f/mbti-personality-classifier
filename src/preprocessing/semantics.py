
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from tqdm import tqdm

class OntologyPreprocessor:
    def __init__(self):
        # Ensure resources are downloaded
        try:
            wn.ensure_loaded()
        except:
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')

    def get_hypernym(self, word, pos_tag_val):
        """
        Returns the name of the hypernym (parent concept) of a word.
        Only applies to Nouns to generalize topics.
        """
        # Map NLTK POS tags to WordNet tags
        if pos_tag_val.startswith('NN'):
            wn_pos = wn.NOUN
        else:
            return word # Return original word if not a noun

        # Get Synsets
        synsets = wn.synsets(word, pos=wn_pos)
        
        if not synsets:
            return word

        # Heuristic: Take the first synset (most common meaning)
        # In a full thesis, you would use 'Lesk Algorithm' for disambiguation
        primary_synset = synsets[0]
        
        # Get Hypernyms
        hypernyms = primary_synset.hypernyms()
        
        if hypernyms:
            # Return the name of the first hypernym (e.g., 'fruit.n.01' -> 'fruit')
            return hypernyms[0].name().split('.')[0]
        
        return word

    def generalize_text(self, text):
        """
        Tokenizes text, finds nouns, and replaces them with hypernyms.
        """
        tokens = text.split()
        if not tokens: return ""
        
        # NLTK POS Tagging is faster than Spacy for this specific loop
        tagged_tokens = pos_tag(tokens)
        
        new_tokens = []
        for word, tag in tagged_tokens:
            new_word = self.get_hypernym(word, tag)
            new_tokens.append(new_word)
            
        return " ".join(new_tokens)

    def run(self, input_path="data/interim/mbti_lemmatized.csv"):
        print("Starting Semantic Generalization (WordNet Hypernyms)...")
        print("Note: This replaces specific nouns with general concepts.")
        
        df = pd.read_csv(input_path)
        
        # We start from Lemmatized text so words are already in dictionary form
        df = df.dropna(subset=['clean_posts'])
        
        tqdm.pandas()
        df['clean_posts'] = df['clean_posts'].progress_apply(self.generalize_text)
        
        output_path = "data/interim/mbti_hypernyms.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved generalized data to {output_path}")
        print(f"Example transformation:\n{df['clean_posts'].iloc[0][:100]}...")

if __name__ == "__main__":
    processor = OntologyPreprocessor()
    processor.run()