import pandas as pd
import numpy as np
import emoji
import re
import yaml
import string
from typing import List, Tuple


class DataCleaner:
    """
    Handles basic text cleaning and removal of noise
    (URLs, Mentions, MBTI specific data leakage).
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def load_data(self) -> pd.DataFrame:
        """Loads raw data from the path defined in config."""
        path = self.config["data"]["raw_path"]
        print(f"Loading data from {path}...")
        df = pd.read_csv(path)

        required_cols = {"type", "posts"}

        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns in dataset: {missing}")

        return df

    def _clean_text(self, text: str) -> str:
        """
        Applies Regex cleaning to a single string.
        """
        text = text.lower()

        if self.config["preprocessing"]["remove_urls"]:
            text = re.sub(r'https?\S+|www\S+', '', text)
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(r'@\w+|#', '', text)

        mbti_types = [
            "infj",
            "entp",
            "intp",
            "intj",
            "entj",
            "enfj",
            "infp",
            "enfp",
            "isfp",
            "istp",
            "isfj",
            "istj",
            "estp",
            "esfp",
            "estj",
            "esfj",
        ]
        pattern = r"\b(" + "|".join(mbti_types) + r")\b"
        text = re.sub(pattern, "", text)
        text = re.sub(r'\bsent (from )?my \w+(\s\w+)? using tapatalk\b', '', text, flags=re.IGNORECASE)

        if self.config["preprocessing"]["remove_punctuation"]:
            text = text.translate(str.maketrans("", "", string.punctuation))

        text = text.replace("|||", " ")

        text = re.sub(r'w w w', '', text)

        text = re.sub(r"\s+", " ", text).strip()

        return text

    def encode_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Splits the 4-letter code into 4 binary columns.
        0 represents the first letter (I, N, F, J)
        1 represents the second letter (E, S, T, P)
        """
        print("Binarizing targets into 4 classifiers...")

        # Map: I->0, E->1 | N->0, S->1 | F->0, T->1 | J->0, P->1
        df["IE"] = df["type"].apply(lambda x: 1 if "E" in x else 0)
        df["NS"] = df["type"].apply(lambda x: 1 if "S" in x else 0)
        df["FT"] = df["type"].apply(lambda x: 1 if "T" in x else 0)
        df["JP"] = df["type"].apply(lambda x: 1 if "P" in x else 0)

        return df

    def run_pipeline(self):
        """Executes the full cleaning pipeline."""
        df = self.load_data()

        # Apply cleaning
        print("Cleaning text data (this may take a moment)...")
        df["clean_posts"] = df["posts"].apply(self._clean_text)

        # Encode targets
        df = self.encode_targets(df)

        # Save to interim
        save_path = "data/interim/mbti_cleaned_basic.csv"
        df.to_csv(save_path, index=False)
        print(f"Cleaned data saved to {save_path}")
        print(df[["type", "IE", "NS", "FT", "JP", "clean_posts"]].head())


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run_pipeline()
