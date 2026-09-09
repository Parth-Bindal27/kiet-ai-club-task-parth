import os
import json
import random
from src.data.dataset import load_jsonl

def mix_datasets(real_data_path: str, synthetic_data_path: str, real_ratio: float, seed: int) -> str:
    """Mixes real and synthetic data based on the real_ratio."""
    random.seed(seed)
    
    real_data = load_jsonl(real_data_path)
    synthetic_data = load_jsonl(synthetic_data_path)
    
    # We maintain the total dataset size equal to the real dataset size
    total_samples = len(real_data)
    num_real = int(total_samples * real_ratio)
    num_synthetic = total_samples - num_real
    
    real_sample = random.sample(real_data, min(num_real, len(real_data))) if num_real > 0 else []
    syn_sample = random.sample(synthetic_data, min(num_synthetic, len(synthetic_data))) if num_synthetic > 0 else []
        
    mixed_data = real_sample + syn_sample
    random.shuffle(mixed_data)
    
    output_path = synthetic_data_path.replace(".jsonl", f"_mixed_real_{real_ratio:.2f}.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in mixed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    return output_path
