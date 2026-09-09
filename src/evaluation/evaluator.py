"""Orchestrator for all evaluation metrics.

Evaluates every model on the SAME untouched real test set.
The test set must NEVER be used for training, generation, or filtering.
"""
import json
import os
import logging
import torch
from typing import Optional

from src.data.dataset import load_jsonl

logger = logging.getLogger(__name__)


class Evaluator:
    """Runs all enabled evaluation metrics on a model."""

    def __init__(self, config, test_data_path: str):
        self.config = config
        self.test_data_path = test_data_path

        # Load test data once
        if os.path.exists(test_data_path):
            self.test_data = load_jsonl(test_data_path)
            self.test_texts = [
                item.get("output", "") for item in self.test_data
            ]
            self.test_instructions = [
                item.get("instruction", "") for item in self.test_data
            ]
            logger.info(f"Loaded {len(self.test_data)} test examples.")
        else:
            logger.warning(f"Test data not found: {test_data_path}")
            self.test_data = []
            self.test_texts = []
            self.test_instructions = []

        self.baseline_metrics = None

    def evaluate_model(
        self,
        model_path: str,
        generation: int,
        synthetic_data_path: Optional[str] = None,
    ) -> dict:
        """Run all enabled metrics and return results.

        Args:
            model_path: Path to the model checkpoint.
            generation: Generation number.
            synthetic_data_path: Optional path to synthetic data for diversity analysis.

        Returns:
            Dict of all computed metrics.
        """
        results = {"generation": generation}
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

        enabled_metrics = self.config.evaluation.metrics

        # --- Perplexity ---
        if "perplexity" in enabled_metrics and os.path.exists(model_path):
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from src.evaluation.perplexity import calculate_perplexity

                logger.info("Calculating perplexity...")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = model.to(device)

                # Format test texts as the model expects them
                eval_texts = []
                max_eval = self.config.evaluation.max_eval_samples or len(self.test_data)
                for item in self.test_data[:max_eval]:
                    text = (
                        f"### Instruction:\n{item.get('instruction', '')}\n\n"
                        f"### Input:\n{item.get('input', '')}\n\n"
                        f"### Response:\n{item.get('output', '')}"
                    )
                    eval_texts.append(text)

                results["perplexity"] = calculate_perplexity(
                    model, tokenizer, eval_texts, device,
                    batch_size=self.config.evaluation.batch_size,
                )

                # Clean up model from memory
                del model
                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()

                logger.info(f"Perplexity: {results['perplexity']:.2f}")
            except Exception as e:
                logger.error(f"Perplexity calculation failed: {e}")
                results["perplexity"] = float("inf")

        # --- ROUGE (generation quality) ---
        if "rouge" in enabled_metrics and os.path.exists(model_path):
            try:
                from src.evaluation.metrics import calculate_rouge
                from src.generation.prompts import format_prompt, extract_response
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info("Calculating ROUGE...")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = model.to(device)
                model.eval()

                # Generate responses for a sample of test instructions
                sample_size = min(100, len(self.test_data))
                predictions = []
                references = []

                for item in self.test_data[:sample_size]:
                    prompt = format_prompt(
                        item.get("instruction", ""),
                        item.get("input", ""),
                    )
                    inputs = tokenizer(
                        prompt, return_tensors="pt", truncation=True, max_length=512
                    ).to(device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs, max_new_tokens=128,
                            temperature=0.7, top_p=0.9, do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = extract_response(decoded)
                    predictions.append(response)
                    references.append(item.get("output", ""))

                results["rouge"] = calculate_rouge(predictions, references)
                logger.info(f"ROUGE-L: {results['rouge'].get('rougeL', 'N/A')}")

                del model
                if device == "mps":
                    torch.mps.empty_cache()
            except Exception as e:
                logger.error(f"ROUGE calculation failed: {e}")
                results["rouge"] = {"rouge1": 0, "rouge2": 0, "rougeL": 0}

        # --- Diversity metrics on synthetic data ---
        if synthetic_data_path and os.path.exists(synthetic_data_path):
            syn_data = load_jsonl(synthetic_data_path)
            syn_texts = [item.get("response", item.get("output", "")) for item in syn_data]

            if "diversity" in enabled_metrics and syn_texts:
                try:
                    from src.evaluation.diversity import LexicalDiversity, SemanticDiversity

                    logger.info("Calculating diversity metrics...")
                    lex = LexicalDiversity()
                    results["lexical_diversity"] = lex.calculate(syn_texts)

                    sem = SemanticDiversity()
                    results["semantic_diversity"] = sem.calculate(syn_texts)
                except Exception as e:
                    logger.error(f"Diversity calculation failed: {e}")

            if "distribution_shift" in enabled_metrics and syn_texts and self.test_texts:
                try:
                    from src.evaluation.distribution import (
                        calculate_js_divergence, calculate_embedding_distance,
                    )

                    logger.info("Calculating distribution shift...")
                    results["js_divergence"] = calculate_js_divergence(
                        self.test_texts, syn_texts
                    )
                    results["embedding_distance"] = calculate_embedding_distance(
                        self.test_texts, syn_texts
                    )
                except Exception as e:
                    logger.error(f"Distribution shift calculation failed: {e}")

            if "duplication" in enabled_metrics and syn_texts:
                try:
                    from src.evaluation.duplication import (
                        calculate_exact_duplicate_rate,
                        calculate_normalized_duplicate_rate,
                        calculate_self_repetition,
                    )

                    logger.info("Calculating duplication metrics...")
                    results["duplication"] = {
                        "exact_duplicate_rate": calculate_exact_duplicate_rate(syn_texts),
                        "normalized_duplicate_rate": calculate_normalized_duplicate_rate(syn_texts),
                        "self_repetition": calculate_self_repetition(syn_texts),
                    }
                except Exception as e:
                    logger.error(f"Duplication calculation failed: {e}")

            if "memorization" in enabled_metrics and syn_texts and self.test_texts:
                try:
                    from src.evaluation.memorization import calculate_memorization

                    logger.info("Calculating memorization metrics...")
                    results["memorization"] = calculate_memorization(
                        self.test_texts, syn_texts
                    )
                except Exception as e:
                    logger.error(f"Memorization calculation failed: {e}")

            if "rare_features" in enabled_metrics and syn_texts:
                try:
                    from src.evaluation.rare_features import RareFeatureTracker

                    logger.info("Calculating rare feature retention...")
                    tracker = RareFeatureTracker(self.test_data_path)
                    results["rare_features"] = tracker.track_feature_retention(
                        syn_texts, generation
                    )
                except Exception as e:
                    logger.error(f"Rare feature tracking failed: {e}")

        # --- Model Collapse Index ---
        try:
            from src.evaluation.collapse_index import ModelCollapseIndex

            if generation == 0:
                self.baseline_metrics = results.copy()
            if self.baseline_metrics:
                mci = ModelCollapseIndex()
                results["model_collapse_index"] = mci.calculate(
                    results, self.baseline_metrics
                )
        except Exception as e:
            logger.error(f"MCI calculation failed: {e}")

        # Save results
        eval_dir = os.path.join(
            os.path.dirname(model_path), "..", "eval_results"
        )
        os.makedirs(eval_dir, exist_ok=True)
        results_path = os.path.join(eval_dir, f"gen_{generation}_metrics.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results
