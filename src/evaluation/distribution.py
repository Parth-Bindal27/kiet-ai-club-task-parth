"""Evaluate distribution shifts between text datasets."""
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import nltk
from collections import Counter

import re

def _tokenize(text: str) -> list[str]:
    """Tokenize text using NLTK or regex fallback."""
    try:
        return nltk.word_tokenize(text.lower())
    except Exception:
        return re.findall(r'\b\w+\b', text.lower())

def calculate_js_divergence(p_texts: list[str], q_texts: list[str], ngram: int = 2) -> float:
    """Build n-gram distributions and calculate Jensen-Shannon divergence."""
    def get_ngrams(texts):
        all_ngrams = []
        for text in texts:
            tokens = _tokenize(text)
            if ngram == 1:
                all_ngrams.extend(tokens)
            elif ngram == 2:
                all_ngrams.extend(nltk.bigrams(tokens))
            elif ngram == 3:
                all_ngrams.extend(nltk.trigrams(tokens))
        return all_ngrams

    p_ngrams = get_ngrams(p_texts)
    q_ngrams = get_ngrams(q_texts)
    
    p_counts = Counter(p_ngrams)
    q_counts = Counter(q_ngrams)
    
    all_keys = set(p_counts.keys()).union(set(q_counts.keys()))
    
    p_dist = np.array([p_counts.get(k, 0) for k in all_keys], dtype=float)
    q_dist = np.array([q_counts.get(k, 0) for k in all_keys], dtype=float)
    
    p_dist += 1e-10
    q_dist += 1e-10
    
    p_dist /= p_dist.sum()
    q_dist /= q_dist.sum()
    
    return float(jensenshannon(p_dist, q_dist))

def calculate_wasserstein_distance(p_embeddings: np.ndarray, q_embeddings: np.ndarray) -> float:
    """Calculate 1D Wasserstein distance on PCA-projected embeddings."""
    combined = np.vstack([p_embeddings, q_embeddings])
    pca = PCA(n_components=1)
    pca.fit(combined)
    
    p_proj = pca.transform(p_embeddings).flatten()
    q_proj = pca.transform(q_embeddings).flatten()
    
    return float(wasserstein_distance(p_proj, q_proj))

def calculate_embedding_distance(real_texts: list[str], synthetic_texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> dict:
    """Calculate various embedding distribution distances."""
    if not real_texts or not synthetic_texts:
        return {"js_divergence": 0.0, "wasserstein_distance": 0.0, "mean_cosine_shift": 0.0}
        
    model = SentenceTransformer(model_name)
    real_emb = model.encode(real_texts, show_progress_bar=False)
    syn_emb = model.encode(synthetic_texts, show_progress_bar=False)
    
    w_dist = calculate_wasserstein_distance(real_emb, syn_emb)
    js_div = calculate_js_divergence(real_texts, synthetic_texts, ngram=2)
    
    real_mean = real_emb.mean(axis=0)
    syn_mean = syn_emb.mean(axis=0)
    
    cos_shift = 1.0 - (np.dot(real_mean, syn_mean) / (np.linalg.norm(real_mean) * np.linalg.norm(syn_mean)))
    
    return {
        "js_divergence": js_div,
        "wasserstein_distance": w_dist,
        "mean_cosine_shift": float(cos_shift)
    }
