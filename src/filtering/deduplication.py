"""Deduplication filters."""

def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())

def _get_ngrams(text: str, n: int) -> set:
    words = text.split()
    return set(tuple(words[i:i+n]) for i in range(max(1, len(words) - n + 1)))

def _jaccard(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

class ExactDeduplicator:
    def deduplicate(self, examples: list[dict], field: str = "response") -> tuple[list[dict], dict]:
        seen = set()
        kept = []
        for ex in examples:
            val = ex.get(field, "")
            if val not in seen:
                seen.add(val)
                kept.append(ex)
                
        orig = len(examples)
        removed = orig - len(kept)
        return kept, {"original_count": orig, "removed_count": removed, "duplicate_percentage": removed/orig if orig else 0.0}

class NormalizedDeduplicator:
    def deduplicate(self, examples: list[dict], field: str = "response") -> tuple[list[dict], dict]:
        seen = set()
        kept = []
        for ex in examples:
            val = _normalize(ex.get(field, ""))
            if val not in seen:
                seen.add(val)
                kept.append(ex)
                
        orig = len(examples)
        removed = orig - len(kept)
        return kept, {"original_count": orig, "removed_count": removed, "duplicate_percentage": removed/orig if orig else 0.0}

class NGramDeduplicator:
    def __init__(self, threshold: float = 0.8, n: int = 3):
        self.threshold = threshold
        self.n = n
        
    def deduplicate(self, examples: list[dict], field: str = "response") -> tuple[list[dict], dict]:
        kept = []
        seen_ngrams = []
        
        for ex in examples:
            text = _normalize(ex.get(field, ""))
            ngrams = _get_ngrams(text, self.n)
            
            is_dup = False
            for seen in seen_ngrams:
                if _jaccard(ngrams, seen) >= self.threshold:
                    is_dup = True
                    break
                    
            if not is_dup:
                seen_ngrams.append(ngrams)
                kept.append(ex)
                
        orig = len(examples)
        removed = orig - len(kept)
        return kept, {"original_count": orig, "removed_count": removed, "duplicate_percentage": removed/orig if orig else 0.0}
