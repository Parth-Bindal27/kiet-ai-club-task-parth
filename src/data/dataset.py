import os
import json
from datasets import load_dataset

def prepare_dataset(config, project_root):
    data_dir = os.path.join(project_root, config.data.data_dir, "processed")
    train_path = os.path.join(data_dir, "train", "dataset.jsonl")
    val_path = os.path.join(data_dir, "validation", "dataset.jsonl")
    test_path = os.path.join(data_dir, "test", "dataset.jsonl")
    
    paths = {
        "train": train_path,
        "validation": val_path,
        "test": test_path
    }
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        return paths
        
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)
    
    print(f"Downloading {config.data.dataset_name}...")
    dataset = load_dataset(config.data.dataset_name)
    
    if "train" not in dataset:
        dataset = dataset[list(dataset.keys())[0]]
    else:
        dataset = dataset["train"]
        
    dataset = dataset.shuffle(seed=config.experiment.seed)
    
    total = len(dataset)
    train_end = int(total * config.data.train_ratio)
    val_end = train_end + int(total * config.data.val_ratio)
    
    train_ds = dataset.select(range(0, train_end))
    val_ds = dataset.select(range(train_end, val_end))
    test_ds = dataset.select(range(val_end, total))
    
    def save_jsonl(ds, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in ds:
                f.write(json.dumps(item) + '\n')
                
    print("Saving splits...")
    save_jsonl(train_ds, train_path)
    save_jsonl(val_ds, val_path)
    save_jsonl(test_ds, test_path)
    
    return paths

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

