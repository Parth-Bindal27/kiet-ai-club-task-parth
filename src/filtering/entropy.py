import math
from collections import Counter

class EntropyFilter:
    """Filters out generated text that has unusually low or high character entropy."""
    def __init__(self, min_entropy: float = 2.0, max_entropy: float = 6.0):
        self.min_entropy = min_entropy
        self.max_entropy = max_entropy

    def _calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        counts = Counter(text)
        length = len(text)
        entropy = 0.0
        for count in counts.values():
            prob = count / length
            entropy -= prob * math.log2(prob)
        return entropy

    def filter(self, examples: list[dict]) -> tuple[list[dict], dict]:
        valid_examples = []
        stats = {"original": len(examples), "removed": 0}
        
        for ex in examples:
            response = ex.get("response", ex.get("output", ""))
            entropy = self._calculate_entropy(response)
            
            if self.min_entropy <= entropy <= self.max_entropy:
                valid_examples.append(ex)
            else:
                stats["removed"] += 1
                
        return valid_examples, stats
