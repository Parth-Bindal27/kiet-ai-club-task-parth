"""Publication-quality visualization plots for model collapse analysis.

Generates matplotlib figures and saves them to results/figures/.
"""
import os
import json
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Consistent style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "legend.fontsize": 10,
})

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548", "#607D8B"]
MARKERS = ["o", "s", "^", "D", "v", "<", ">"]


def _load_experiment_results(experiment_dir: str) -> dict:
    """Load results.json from an experiment directory."""
    path = os.path.join(experiment_dir, "results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _get_metric_series(results: dict, metric_path: str) -> tuple[list[int], list[float]]:
    """Extract a metric across generations using dot-notation path."""
    generations = sorted(int(k) for k in results.get("generations", {}).keys())
    values = []
    for gen in generations:
        data = results["generations"][str(gen)].get("metrics", {})
        parts = metric_path.split(".")
        val = data
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                val = None
                break
        if val is not None and isinstance(val, (int, float)):
            values.append(float(val))
        else:
            values.append(None)
    return generations, values


def plot_metric_over_generations(
    results: dict, metric_name: str, title: str, save_path: str,
    ylabel: str = None
) -> plt.Figure:
    """Plot a single metric across generations."""
    gens, vals = _get_metric_series(results, metric_name)
    valid = [(g, v) for g, v in zip(gens, vals) if v is not None]
    if not valid:
        logger.warning(f"No data for metric: {metric_name}")
        return None

    fig, ax = plt.subplots()
    g, v = zip(*valid)
    ax.plot(g, v, marker="o", color=COLORS[0], linewidth=2, markersize=8)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel or metric_name)
    ax.set_title(title)
    ax.set_xticks(list(g))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return fig


def plot_multiple_experiments(
    experiment_results: dict[str, dict],
    metric_name: str,
    title: str,
    save_path: str,
    ylabel: str = None,
) -> plt.Figure:
    """Overlay multiple experiments on the same plot."""
    fig, ax = plt.subplots()

    for i, (exp_name, results) in enumerate(experiment_results.items()):
        gens, vals = _get_metric_series(results, metric_name)
        valid = [(g, v) for g, v in zip(gens, vals) if v is not None]
        if valid:
            g, v = zip(*valid)
            ax.plot(g, v, marker=MARKERS[i % len(MARKERS)],
                    color=COLORS[i % len(COLORS)], linewidth=2,
                    markersize=8, label=exp_name)

    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel or metric_name)
    ax.set_title(title)
    ax.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved: {save_path}")
    return fig


def generate_all_plots(experiment_dirs: list[str], output_dir: str) -> list[str]:
    """Generate all required plots from experiment results.

    Args:
        experiment_dirs: List of experiment result directories.
        output_dir: Directory to save figures.

    Returns:
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # Load all experiment results
    all_results = {}
    for d in experiment_dirs:
        name = os.path.basename(d)
        results = _load_experiment_results(d)
        if results:
            all_results[name] = results

    if not all_results:
        logger.warning("No experiment results found to plot.")
        return saved

    # Define metrics to plot
    metrics = [
        ("perplexity", "Perplexity vs Generation", "Perplexity"),
        ("rouge.rougeL", "ROUGE-L vs Generation", "ROUGE-L"),
        ("lexical_diversity.type_token_ratio", "Type-Token Ratio vs Generation", "TTR"),
        ("lexical_diversity.vocabulary_size", "Vocabulary Size vs Generation", "Vocab Size"),
        ("semantic_diversity.pairwise_cosine_similarity", "Semantic Similarity vs Generation", "Cosine Similarity"),
        ("semantic_diversity.embedding_variance", "Embedding Variance vs Generation", "Variance"),
        ("duplication.exact_duplicate_rate", "Duplicate Rate vs Generation", "Duplicate Rate"),
        ("duplication.self_repetition", "Self-Repetition vs Generation", "Self-Repetition"),
        ("js_divergence", "Distribution Divergence vs Generation", "JS Divergence"),
        ("rare_features.rare", "Rare Feature Retention vs Generation", "Retention Rate"),
        ("memorization.mean_ngram_overlap", "Memorization vs Generation", "N-gram Overlap"),
        ("model_collapse_index.mci", "Model Collapse Index vs Generation", "MCI"),
    ]

    # Per-experiment plots
    for exp_name, results in all_results.items():
        for metric_path, title, ylabel in metrics:
            path = os.path.join(output_dir, f"{exp_name}_{metric_path.replace('.', '_')}.png")
            fig = plot_metric_over_generations(
                results, metric_path, f"{title}\n({exp_name})", path, ylabel
            )
            if fig:
                saved.append(path)

    # Comparison plots (all experiments on same figure)
    if len(all_results) > 1:
        for metric_path, title, ylabel in metrics:
            path = os.path.join(output_dir, f"comparison_{metric_path.replace('.', '_')}.png")
            fig = plot_multiple_experiments(
                all_results, metric_path, title, path, ylabel
            )
            if fig:
                saved.append(path)

    logger.info(f"Generated {len(saved)} plots in {output_dir}")
    return saved
