"""
End-to-end data processing: load → clean → tokenize → extract features.
"""

import pandas as pd
from tqdm import tqdm
from typing import Tuple
from .loader import load_raw_data
from .preprocessing import preprocess_posts, tokenize_text, generate_ngrams


def encode_mbti_type(mbti: str) -> Tuple[int, int, int, int]:
    """
    Encode MBTI type as 4 binary dimensions:
    I/E → 1/0, N/S → 1/0, F/T → 1/0, J/P → 1/0
    """
    return (
        1 if mbti[0] == 'I' else 0,
        1 if mbti[1] == 'N' else 0,
        1 if mbti[2] == 'F' else 0,
        1 if mbti[3] == 'J' else 0,
    )


def run_full_preprocessing(
    filepath: str = None,
    data_dir: str = None,
    filename: str = "mbti_1.csv"
) -> pd.DataFrame:
    """
    Load raw data and apply full preprocessing pipeline.
    Returns a DataFrame with:
    - cleaned_posts
    - tokens
    - Unigrams, Bigrams, Trigrams
    - IE, NS, FT, JP (binary labels)
    """
    # Load
    df = load_raw_data(filepath=filepath, data_dir=data_dir, filename=filename)

    # Clean posts
    print("🧹 Cleaning posts...")
    df['cleaned_posts'] = df['posts'].progress_apply(preprocess_posts)

    # Encode labels
    print("🏷️  Encoding MBTI types...")
    df[['IE', 'NS', 'FT', 'JP']] = pd.DataFrame(
        df['type'].progress_apply(encode_mbti_type).tolist(),
        index=df.index
    )

    # Tokenize
    print("🔤 Tokenizing and lemmatizing...")
    tqdm.pandas()
    df['tokens'] = df['cleaned_posts'].progress_apply(tokenize_text)

    # Generate n-grams
    print("📊 Generating n-grams...")
    ngram_results = df['tokens'].progress_apply(generate_ngrams)
    df[['Unigrams', 'Bigrams', 'Trigrams']] = pd.DataFrame(
        ngram_results.tolist(), index=df.index
    )

    return df