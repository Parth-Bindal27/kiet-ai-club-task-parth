"""Evaluate memorization of training data."""
from src.evaluation.duplication import _get_ngram_sets

def calculate_memorization(train_texts: list[str], generated_texts: list[str], n: int = 4) -> dict:
    """Calculate how much of the training data is memorized in generated texts."""
    if not train_texts or not generated_texts:
        return {"exact_overlap_rate": 0.0, "mean_ngram_overlap": 0.0, "max_ngram_overlap": 0.0}
        
    train_set = set([" ".join(str(t).lower().split()) for t in train_texts])
    gen_normalized = [" ".join(str(t).lower().split()) for t in generated_texts]
    
    exact_matches = sum(1 for t in gen_normalized if t in train_set)
    exact_overlap_rate = exact_matches / len(generated_texts)
    
    train_ngrams = _get_ngram_sets(train_texts, n)
    gen_ngrams = _get_ngram_sets(generated_texts, n)
    
    all_train_ngrams = set().union(*train_ngrams)
    
    overlaps = []
    for g_set in gen_ngrams:
        if not g_set:
            overlaps.append(0.0)
            continue
        intersection = g_set.intersection(all_train_ngrams)
        overlaps.append(len(intersection) / len(g_set))
        
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    max_overlap = max(overlaps) if overlaps else 0.0
    
    return {
        "exact_overlap_rate": exact_overlap_rate,
        "mean_ngram_overlap": mean_overlap,
        "max_ngram_overlap": max_overlap
    }
