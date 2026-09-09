"""Quality filtering."""

class QualityScorer:
    def score(self, example: dict) -> float:
        response = example.get("response", "").strip()
        instruction = example.get("instruction", "").strip()
        
        if not response:
            return 0.0
            
        score = 10.0
        
        # Length
        if len(response) < 10 or len(response) > 2048:
            score -= 3.0
            
        # Repetition (basic)
        words = response.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 4.0
                
        # Completeness
        if response[-1] not in ".!?\"':;":
            score -= 1.0
            
        # Relevance (basic overlap)
        inst_words = set(instruction.lower().split())
        resp_words = set(response.lower().split())
        if inst_words and resp_words:
            overlap = len(inst_words.intersection(resp_words)) / len(inst_words)
            if overlap < 0.1:
                score -= 2.0
                
        return max(0.0, min(10.0, score))

def filter_by_quality(examples: list[dict], min_score: float) -> tuple[list[dict], dict]:
    scorer = QualityScorer()
    kept = []
    
    for ex in examples:
        if scorer.score(ex) >= min_score:
            kept.append(ex)
            
    orig = len(examples)
    removed = orig - len(kept)
    return kept, {"original_count": orig, "removed_count": removed, "filtered_percentage": removed/orig if orig else 0.0}
