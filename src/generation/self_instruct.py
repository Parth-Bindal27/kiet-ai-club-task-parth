"""Simplified self-instruct module."""
import json
import random

class SelfInstructGenerator:
    """Selects instructions from training data."""
    def select_instructions(self, train_data_path: str, num_samples: int, seed: int, categories: list[str] = None) -> list[dict]:
        random.seed(seed)
        
        data = []
        with open(train_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
                
        if categories:
            data = [item for item in data if item.get("category") in categories]
            
        if len(data) > num_samples:
            sampled = random.sample(data, num_samples)
        else:
            sampled = data
            
        return [{"instruction": item["instruction"], "input": item.get("input", ""), "category": item.get("category", "")} for item in sampled]
