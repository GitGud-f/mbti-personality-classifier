"""
Text preprocessing utilities: cleaning, tokenization, lemmatization, and n-gram generation.
"""

import re
import contractions
import emoji
from typing import List, Tuple, Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.util import ngrams
from nltk import pos_tag

# Initialize once
STOP_WORDS = set(stopwords.words('english'))
WNL = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Clean raw text by removing URLs, emojis, mentions, and MBTI codes."""
    text = text.lower()
    text = re.sub(r'https?\S+|www\S+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r"[^a-z\']", ' ', text)
    # Remove MBTI type codes (e.g., INFJ, ENTP) to avoid leaking information
    text = re.sub(r'\b(I|E)(N|S)(F|T)(J|P)\b', '', text, flags=re.IGNORECASE)
    # Remove common footer
    text = re.sub(r'\bsent (from )?my \w+(\s\w+)? using tapatalk\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'w w w', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_posts(posts_str: str) -> str:
    """Split multi-post string (separated by '|||'), clean each, and join."""
    posts = posts_str.split('|||')
    cleaned_posts = [clean_text(post) for post in posts]
    joined = ' '.join(cleaned_posts)
    return re.sub(r'\s+', ' ', joined).strip()


def get_wordnet_pos(tag: str) -> str:
    """Map POS tag to WordNet format for lemmatization."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def tokenize_text(text: str) -> List[str]:
    """Expand contractions, tokenize, remove stopwords, and lemmatize."""
    fixed = contractions.fix(text)
    tokens = word_tokenize(fixed)
    filtered_tokens = [word for word in tokens if word.lower() not in STOP_WORDS]
    pos_tags = pos_tag(filtered_tokens)
    lemmatized = [
        WNL.lemmatize(token, pos=get_wordnet_pos(pos))
        for token, pos in pos_tags
    ]
    return lemmatized


def generate_ngrams(tokens: List[str]) -> Tuple[List, List, List]:
    """Generate unigrams, bigrams, and trigrams from token list."""
    unigrams = list(ngrams(tokens, 1))
    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))
    return unigrams, bigrams, trigrams