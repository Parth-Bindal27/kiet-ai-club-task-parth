"""Experiment tracking and result logging."""
import json
import os
import pandas as pd

class ExperimentTracker:
    def __init__(self, experiment_name: str, output_dir: str):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.results = {
            "config": {},
            "generations": {}
        }
        os.makedirs(self.output_dir, exist_ok=True)
        
    def log_config(self, config: dict):
        self.results["config"] = config
        
    def log_generation(self, generation: int, metrics: dict, training_stats: dict, generation_stats: dict = None):
        self.results["generations"][str(generation)] = {
            "metrics": metrics,
            "training_stats": training_stats,
            "generation_stats": generation_stats or {}
        }
        
    def log_filtering_stats(self, generation: int, stats: dict):
        if str(generation) not in self.results["generations"]:
            self.results["generations"][str(generation)] = {}
        self.results["generations"][str(generation)]["filtering_stats"] = stats
        
    def save(self):
        """Save everything to JSON and CSV."""
        json_path = os.path.join(self.output_dir, "results.json")
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        rows = []
        for gen_str, data in self.results["generations"].items():
            row = {"generation": int(gen_str)}
            
            metrics = data.get("metrics", {})
            for k, v in metrics.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, dict):
                            for sub_sub_k, sub_sub_v in sub_v.items():
                                row[f"{k}_{sub_k}_{sub_sub_k}"] = sub_sub_v
                        else:
                            row[f"{k}_{sub_k}"] = sub_v
                else:
                    row[k] = v
                    
            rows.append(row)
            
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
            
    @classmethod
    def load(cls, experiment_dir: str):
        tracker = cls(os.path.basename(experiment_dir), experiment_dir)
        json_path = os.path.join(experiment_dir, "results.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                tracker.results = json.load(f)
        return tracker
        
    def get_metric_over_generations(self, metric_name: str) -> list[float]:
        """Extract a specific metric across all generations."""
        gens = sorted([int(k) for k in self.results["generations"].keys()])
        
        values = []
        for g in gens:
            data = self.results["generations"][str(g)]
            metrics = data.get("metrics", {})
            
            parts = metric_name.split('.')
            val = metrics
            for p in parts:
                if isinstance(val, dict) and p in val:
                    val = val[p]
                else:
                    val = None
                    break
            
            if val is not None:
                values.append(float(val))
                
        return values
