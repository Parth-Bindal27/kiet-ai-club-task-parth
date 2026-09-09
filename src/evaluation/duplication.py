"""Evaluate duplication and overlap in datasets."""
import nltk

def calculate_exact_duplicate_rate(texts: list[str]) -> float:
    """Calculate fraction of texts that are exact duplicates."""
    if not texts:
        return 0.0
    unique_texts = set(texts)
    return (len(texts) - len(unique_texts)) / len(texts)

def calculate_normalized_duplicate_rate(texts: list[str]) -> float:
    """Calculate duplicate rate after normalizing whitespace and case."""
    if not texts:
        return 0.0
    normalized = [" ".join(str(t).lower().split()) for t in texts]
    unique_texts = set(normalized)
    return (len(texts) - len(unique_texts)) / len(texts)

import re

def _tokenize(text: str) -> list[str]:
    """Tokenize text using NLTK or regex fallback."""
    try:
        return nltk.word_tokenize(text.lower())
    except Exception:
        return re.findall(r'\b\w+\b', text.lower())

def _get_ngram_sets(texts: list[str], n: int) -> list[set]:
    """Helper to convert texts to sets of n-grams."""
    ngram_sets = []
    for text in texts:
        tokens = _tokenize(text)
        if n == 1:
            ngrams = tokens
        elif n == 2:
            ngrams = list(nltk.bigrams(tokens))
        elif n == 3:
            ngrams = list(nltk.trigrams(tokens))
        else:
            ngrams = list(nltk.ngrams(tokens, n))
        ngram_sets.append(set(ngrams))
    return ngram_sets

def calculate_ngram_overlap(texts1: list[str], texts2: list[str], n: int = 3) -> float:
    """Calculate Jaccard overlap of n-gram sets between two corpora."""
    if not texts1 or not texts2:
        return 0.0
        
    set1_ngrams = _get_ngram_sets(texts1, n)
    set2_ngrams = _get_ngram_sets(texts2, n)
    
    vocab1 = set().union(*set1_ngrams)
    vocab2 = set().union(*set2_ngrams)
    
    intersection = vocab1.intersection(vocab2)
    union = vocab1.union(vocab2)
    
    if not union:
        return 0.0
        
    return len(intersection) / len(union)

def calculate_self_repetition(texts: list[str], n: int = 3) -> float:
    """Average pairwise n-gram Jaccard similarity within a corpus."""
    if len(texts) < 2:
        return 0.0
        
    ngram_sets = _get_ngram_sets(texts, n)
    
    import random
    if len(ngram_sets) > 500:
        ngram_sets = random.sample(ngram_sets, 500)
        
    total_sim = 0.0
    pairs = 0
    
    for i in range(len(ngram_sets)):
        for j in range(i + 1, len(ngram_sets)):
            s1, s2 = ngram_sets[i], ngram_sets[j]
            if not s1 and not s2:
                continue
            intersection = len(s1.intersection(s2))
            union = len(s1.union(s2))
            if union > 0:
                total_sim += intersection / union
            pairs += 1
            
    if pairs == 0:
        return 0.0
    return total_sim / pairs
