"""Evaluate a model on the test set.

Usage:
    python scripts/evaluate.py --config configs/base.yaml --generation 0
"""
import argparse
import json
import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.data.dataset import prepare_dataset
from src.evaluation.evaluator import Evaluator
from src.utils.seeding import set_seed
from src.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on the test set.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--generation", type=int, required=True, help="Generation to evaluate")
    parser.add_argument("--model-path", type=str, default=None, help="Override model checkpoint path")
    parser.add_argument("--synthetic-data", type=str, default=None, help="Path to synthetic data for diversity analysis")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.experiment.seed)
    setup_logging(os.path.join(PROJECT_ROOT, config.experiment.log_dir), config.experiment.name)
    logger = logging.getLogger(__name__)

    # Prepare dataset
    data_paths = prepare_dataset(config, PROJECT_ROOT)

    # Model path
    model_path = args.model_path or os.path.join(
        PROJECT_ROOT, config.experiment.output_dir,
        "checkpoints", f"gen_{args.generation}"
    )

    # Evaluate
    evaluator = Evaluator(config, data_paths["test"])
    metrics = evaluator.evaluate_model(
        model_path=model_path,
        generation=args.generation,
        synthetic_data_path=args.synthetic_data,
    )

    print(f"\nEvaluation Results (Generation {args.generation}):")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
