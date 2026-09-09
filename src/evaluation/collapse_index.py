"""Calculate composite Model Collapse Index."""

class ModelCollapseIndex:
    """Project-specific composite metric for quantifying model collapse."""
    def __init__(self, weights: dict = None):
        if weights is None:
            self.weights = {
                "performance": 0.25,
                "diversity": 0.25,
                "distribution": 0.2,
                "duplication": 0.15,
                "rare_features": 0.15
            }
        else:
            self.weights = weights
            
    def calculate(self, metrics: dict, baseline_metrics: dict) -> dict:
        """
        Calculate Model Collapse Index (0 = no collapse, 1 = total collapse).
        Normalizes each component relative to baseline (generation 0).
        """
        components = {}
        
        baseline_rouge = baseline_metrics.get("rouge", {}).get("rougeL", 1e-5)
        current_rouge = metrics.get("rouge", {}).get("rougeL", 0.0)
        perf_collapse = max(0.0, min(1.0, 1.0 - (current_rouge / baseline_rouge)))
        components["performance"] = perf_collapse
        
        baseline_ttr = baseline_metrics.get("lexical_diversity", {}).get("type_token_ratio", 1e-5)
        current_ttr = metrics.get("lexical_diversity", {}).get("type_token_ratio", 0.0)
        div_collapse = max(0.0, min(1.0, 1.0 - (current_ttr / baseline_ttr)))
        components["diversity"] = div_collapse
        
        # Distribution divergence: accept direct js_divergence or nested inside embedding_distance
        js_div = metrics.get("js_divergence")
        if js_div is None:
            js_div = metrics.get("embedding_distance", {}).get("js_divergence", 0.0)
        dist_collapse = min(1.0, max(0.0, js_div / 0.693))
        components["distribution"] = dist_collapse
        
        # Duplication
        baseline_dup = baseline_metrics.get("duplication", {}).get("self_repetition", 0.0)
        current_dup = metrics.get("duplication", {}).get("self_repetition", 0.0)
        dup_collapse = max(0.0, min(1.0, current_dup - baseline_dup))
        components["duplication"] = dup_collapse
        
        # Rare features: accept direct key or nested under retention
        baseline_rare = baseline_metrics.get("rare_features", {}).get("rare")
        if baseline_rare is None:
            baseline_rare = baseline_metrics.get("rare_features", {}).get("retention", {}).get("rare", 1e-5)
        current_rare = metrics.get("rare_features", {}).get("rare")
        if current_rare is None:
            current_rare = metrics.get("rare_features", {}).get("retention", {}).get("rare", 0.0)
        rare_collapse = max(0.0, min(1.0, 1.0 - (current_rare / (baseline_rare or 1e-5))))
        components["rare_features"] = rare_collapse
        
        mci = sum(components[k] * self.weights[k] for k in self.weights)
        
        return {
            "mci": float(mci),
            "components": components
        }
