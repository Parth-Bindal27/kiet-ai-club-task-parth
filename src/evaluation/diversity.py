"""Evaluate lexical and semantic diversity of text."""
from collections import Counter
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

import re

def _tokenize(text: str) -> list[str]:
    """Tokenize text using NLTK or regex fallback."""
    try:
        return nltk.word_tokenize(text.lower())
    except Exception:
        return re.findall(r'\b\w+\b', text.lower())

class LexicalDiversity:
    def calculate(self, texts: list[str]) -> dict:
        """Calculate lexical diversity metrics."""
        all_tokens = []
        for text in texts:
            tokens = _tokenize(text)
            all_tokens.extend(tokens)
            
        if not all_tokens:
            return {
                "vocabulary_size": 0,
                "type_token_ratio": 0.0,
                "unique_unigrams": 0,
                "unique_bigrams": 0,
                "unique_trigrams": 0,
                "hapax_legomena_ratio": 0.0
            }
            
        token_counts = Counter(all_tokens)
        vocab = set(all_tokens)
        
        bigrams = list(nltk.bigrams(all_tokens))
        trigrams = list(nltk.trigrams(all_tokens))
        
        vocab_size = len(vocab)
        total_tokens = len(all_tokens)
        
        hapax_legomena = sum(1 for token, count in token_counts.items() if count == 1)
        
        return {
            "vocabulary_size": vocab_size,
            "type_token_ratio": vocab_size / total_tokens if total_tokens > 0 else 0,
            "unique_unigrams": vocab_size,
            "unique_bigrams": len(set(bigrams)),
            "unique_trigrams": len(set(trigrams)),
            "hapax_legomena_ratio": hapax_legomena / vocab_size if vocab_size > 0 else 0
        }

class SemanticDiversity:
    def calculate(self, texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> dict:
        """Calculate semantic diversity metrics using sentence embeddings."""
        if not texts:
            return {"pairwise_cosine_similarity": 0.0, "embedding_variance": 0.0}
            
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=False)
        
        variance = np.var(embeddings, axis=0).mean()
        
        num_texts = len(texts)
        if num_texts > 1000:
            indices = np.random.choice(num_texts, 1000, replace=False)
            sampled_embeddings = embeddings[indices]
            sim_matrix = cosine_similarity(sampled_embeddings)
        else:
            sim_matrix = cosine_similarity(embeddings)
            
        upper_tri = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)]
        mean_sim = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
        
        return {
            "pairwise_cosine_similarity": mean_sim,
            "embedding_variance": float(variance)
        }
