"""Pipeline for filtering."""
import os
import json
from src.filtering.validation import validate_example
from src.filtering.deduplication import ExactDeduplicator, NormalizedDeduplicator, NGramDeduplicator
from src.filtering.quality import filter_by_quality

class FilteringPipeline:
    def __init__(self, config):
        self.config = config
        
    def run(self, examples: list[dict], generation: int, output_dir: str) -> tuple[list[dict], dict]:
        stats = {}
        
        if not self.config.enabled:
            return examples, stats
            
        # Validation
        valid_examples = []
        val_stats = {"original": len(examples), "removed": 0}
        for ex in examples:
            is_valid, _ = validate_example(ex)
            if is_valid:
                valid_examples.append(ex)
            else:
                val_stats["removed"] += 1
        stats["validation"] = val_stats
        current_examples = valid_examples
        
        # Deduplication
        if self.config.deduplication:
            if self.config.dedup_method == "exact":
                deduper = ExactDeduplicator()
            elif self.config.dedup_method == "ngram":
                deduper = NGramDeduplicator()
            else:
                deduper = NormalizedDeduplicator()
                
            current_examples, dedup_stats = deduper.deduplicate(current_examples)
            stats["deduplication"] = dedup_stats
            
        # Quality
        if self.config.quality_filter:
            current_examples, qual_stats = filter_by_quality(current_examples, self.config.min_quality_score)
            stats["quality"] = qual_stats
            
        stats["original_count"] = len(examples)
        stats["final_count"] = len(current_examples)
        stats["removed_total"] = len(examples) - len(current_examples)
            
        os.makedirs(output_dir, exist_ok=True)
        stats_path = os.path.join(output_dir, f"filtering_stats_gen{generation}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
            
        return current_examples, stats
