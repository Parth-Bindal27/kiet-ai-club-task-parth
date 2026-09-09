"""Standard generation metrics: ROUGE, BLEU, Exact Match.

Uses rouge-score library directly for reliability.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_rouge(predictions: list[str], references: list[str]) -> dict:
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        predictions: Model-generated responses.
        references: Reference (ground truth) responses.

    Returns:
        Dict with rouge1, rouge2, rougeL F-measure scores.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {
        key: sum(vals) / len(vals) if vals else 0.0
        for key, vals in scores.items()
    }


def calculate_exact_match(predictions: list[str], references: list[str]) -> float:
    """Calculate exact match score (fraction of exact string matches).

    Args:
        predictions: Model predictions.
        references: Reference answers.

    Returns:
        Fraction of predictions that exactly match the reference.
    """
    if not predictions:
        return 0.0
    matches = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    return matches / len(predictions)
