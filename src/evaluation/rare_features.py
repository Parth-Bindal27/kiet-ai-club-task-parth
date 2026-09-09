"""Track retention of rare features across generations.

Identifies low-frequency topics/features in the original dataset and
tracks whether they survive through synthetic data generations.
"""
import logging
from collections import Counter
from typing import Optional

from src.data.dataset import load_jsonl

logger = logging.getLogger(__name__)


class RareFeatureTracker:
    """Tracks rare feature retention across synthetic generations."""

    def __init__(self, data_path: str):
        """Load data and identify feature frequencies.

        Args:
            data_path: Path to the JSONL data file.
        """
        self.data_path = data_path
        self.categories = {"common": set(), "medium": set(), "rare": set()}
        self._initialize_frequencies()

    def _initialize_frequencies(self):
        """Classify words into common/medium/rare by frequency."""
        try:
            data = load_jsonl(self.data_path)
        except Exception as e:
            logger.warning(f"Could not load data for rare feature tracking: {e}")
            return

        all_words = []
        for item in data:
            text = f"{item.get('instruction', '')} {item.get('input', '')} {item.get('output', '')}"
            all_words.extend(text.lower().split())

        freqs = Counter(all_words)

        # Filter out very short words and sort by frequency (most common first)
        sorted_words = [w for w, c in freqs.most_common() if len(w) > 3]

        n_words = len(sorted_words)
        if n_words == 0:
            return

        top_split = int(n_words * 0.3)
        bottom_split = int(n_words * 0.7)

        self.categories = {
            "common": set(sorted_words[:top_split]),
            "medium": set(sorted_words[top_split:bottom_split]),
            "rare": set(sorted_words[bottom_split:]),
        }
        logger.info(
            f"Rare feature tracker: {len(self.categories['common'])} common, "
            f"{len(self.categories['medium'])} medium, "
            f"{len(self.categories['rare'])} rare features"
        )

    def evaluate_by_frequency(
        self,
        predictions: list[str],
        references: list[str],
        categories: Optional[list[str]] = None,
    ) -> dict:
        """Evaluate ROUGE scores grouped by word frequency category.

        Args:
            predictions: Model predictions.
            references: Reference answers.
            categories: Optional list of pre-assigned categories per example.

        Returns:
            Dict with common/medium/rare ROUGE scores.
        """
        if not predictions or not references:
            return {}

        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        except ImportError:
            logger.warning("rouge_score not available for rare feature evaluation")
            return {}

        # Categorize each reference by its dominant word frequency
        ref_categories = []
        for ref in references:
            words = set(ref.lower().split())
            counts = {
                "common": len(words.intersection(self.categories["common"])),
                "medium": len(words.intersection(self.categories["medium"])),
                "rare": len(words.intersection(self.categories["rare"])),
            }
            cat = max(counts, key=counts.get) if any(counts.values()) else "common"
            ref_categories.append(cat)

        results = {"common": {}, "medium": {}, "rare": {}}

        for target_cat in ["common", "medium", "rare"]:
            cat_preds = [p for p, c in zip(predictions, ref_categories) if c == target_cat]
            cat_refs = [r for r, c in zip(references, ref_categories) if c == target_cat]

            if cat_preds:
                scores = [
                    scorer.score(ref, pred)["rougeL"].fmeasure
                    for pred, ref in zip(cat_preds, cat_refs)
                ]
                results[target_cat] = {
                    "count": len(cat_preds),
                    "rouge": sum(scores) / len(scores),
                }
            else:
                results[target_cat] = {"count": 0, "rouge": 0.0}

        return results

    def track_feature_retention(self, synthetic_texts: list[str], generation: int) -> dict:
        """Track which rare/medium/common features are present in synthetic data.

        Args:
            synthetic_texts: Generated text samples.
            generation: Generation number.

        Returns:
            Dict with retention rates for common/medium/rare features.
        """
        syn_words = set()
        for text in synthetic_texts:
            syn_words.update(text.lower().split())

        retention = {}
        for cat_name, cat_words in self.categories.items():
            if not cat_words:
                retention[cat_name] = 0.0
                continue
            retained = len(syn_words.intersection(cat_words))
            retention[cat_name] = retained / len(cat_words)

        return retention
