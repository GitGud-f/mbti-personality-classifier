import joblib
import pandas as pd
import numpy as np
from src.preprocessing.clean import DataCleaner
from src.preprocessing.morphology import MorphologyPreprocessor
from src.features.syntax import SyntaxFeatureExtractor

class FinalMBTIPredictor:
    def __init__(self):
        self.cleaner = DataCleaner()
        self.morph = MorphologyPreprocessor()
        self.syntax = SyntaxFeatureExtractor()
        
        # Load the Best Models (Hypothetical paths based on your logic)
        # You would ensure these specific models are saved during training
        print("Loading Best-in-Class Models...")
        self.model_ie = joblib.load("models/best_IE_lemmatized.pkl")
        self.model_ns = joblib.load("models/best_NS_lemmatized.pkl")
        self.model_ft = joblib.load("models/best_FT_syntax.pkl") # This one uses syntax!
        self.model_jp = joblib.load("models/best_JP_lemmatized.pkl")

    def predict(self, text):
        """
        Takes a raw string and returns the 4-letter MBTI type
        using the best model for each letter.
        """
        # 1. Base Cleaning (Common to all)
        clean_text = self.cleaner._clean_text(text)
        
        # 2. Preprocessing A: Lemmatization (For IE, NS, JP)
        lemma_text = self.morph.apply_lemmatization(clean_text)
        
        # 3. Preprocessing B: Syntax Extraction (For FT)
        # We need the syntax vector + the text vector
        # Note: In a real production system, this part is tricky to automate 
        # without a dedicated pipeline object, but here is the logic:
        syntax_feats = self.syntax.get_syntax_features(clean_text)
        # (Assuming the model_ft is a pipeline that handles the dict->vector conversion)
        
        # --- PREDICTIONS ---
        
        # Predict IE (Expects Lemmatized String)
        ie_pred = self.model_ie.predict([lemma_text])[0]
        letter_ie = 'E' if ie_pred == 1 else 'I'
        
        # Predict NS (Expects Lemmatized String)
        ns_pred = self.model_ns.predict([lemma_text])[0]
        letter_ns = 'S' if ns_pred == 1 else 'N'
        
        # Predict FT (Expects Syntax Features + Text)
        # For simplicity in this example, assuming the pipeline handles extraction
        ft_pred = self.model_ft.predict([clean_text])[0] 
        letter_ft = 'T' if ft_pred == 1 else 'F'
        
        # Predict JP (Expects Lemmatized String)
        jp_pred = self.model_jp.predict([lemma_text])[0]
        letter_jp = 'P' if jp_pred == 1 else 'J'
        
        return f"{letter_ie}{letter_ns}{letter_ft}{letter_jp}"

# Example Usage
if __name__ == "__main__":
    predictor = FinalMBTIPredictor()
    user_input = "I really love analyzing data but I feel sad when my code breaks."
    mbti = predictor.predict(user_input)
    print(f"Predicted Type: {mbti}")