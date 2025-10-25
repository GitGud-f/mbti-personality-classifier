"""
Visualization utilities for n-gram analysis (for EDA only).
"""

import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple
import pandas as pd


def get_top_ngrams_by_type(
    df: pd.DataFrame,
    ngram_type: str = 'unigrams',
    top_n: int = 20
):
    """
    Plot top n-grams per personality type.
    ngram_type: 'unigrams', 'bigrams', or 'trigrams'
    """

    personality_types = sorted(df['type'].unique())
    n_col = 4
    n_row = (len(personality_types) + n_col - 1) // n_col

    fig, axes = plt.subplots(n_row, n_col, figsize=(20, 5 * n_row))
    axes = axes.flatten() if len(personality_types) > 1 else [axes]

    for idx, personality in enumerate(personality_types):
        type_data = df[df['type'] == personality]

        all_ngrams = []
        for _, row in type_data.iterrows():
            if ngram_type == 'unigrams':
                ngs = row['Unigrams']
            elif ngram_type == 'bigrams':
                ngs = row['Bigrams']
            elif ngram_type == 'trigrams':
                ngs = row['Trigrams']
            else:
                raise ValueError("ngram_type must be 'unigrams', 'bigrams', or 'trigrams'")
            all_ngrams.extend(ngs)

        ngram_counts = Counter(all_ngrams)
        top_ngrams = ngram_counts.most_common(top_n)

        ngram_texts = [' '.join(gram) for gram, count in top_ngrams]
        counts = [count for gram, count in top_ngrams]

        axes[idx].barh(range(len(ngram_texts)), counts)
        axes[idx].set_yticks(range(len(ngram_texts)))
        axes[idx].set_yticklabels(ngram_texts, fontsize=8)
        axes[idx].set_title(f'{personality} - Top {ngram_type.title()}', fontsize=10)
        axes[idx].set_xlabel('Frequency')
        axes[idx].invert_yaxis()

    # Hide unused subplots
    for j in range(len(personality_types), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle(f'Top {top_n} {ngram_type.title()} by Personality Type', fontsize=16, y=1.02)
    plt.show()