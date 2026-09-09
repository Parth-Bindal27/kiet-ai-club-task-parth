"""Train a model for a specific generation.

Usage:
    python scripts/train.py --config configs/base.yaml --generation 0
"""
import argparse
import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.data.dataset import prepare_dataset
from src.models.trainer import ModelTrainer
from src.utils.seeding import set_seed
from src.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train a model for a specific generation.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--generation", type=int, default=0, help="Generation number")
    parser.add_argument("--train-data", type=str, default=None, help="Override training data path")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.experiment.seed)
    setup_logging(os.path.join(PROJECT_ROOT, config.experiment.log_dir), config.experiment.name)
    logger = logging.getLogger(__name__)

    # Prepare dataset if needed
    data_paths = prepare_dataset(config, PROJECT_ROOT)

    # Determine training data
    train_data = args.train_data or data_paths["train"]
    val_data = data_paths["validation"]

    # Output directory
    output_dir = os.path.join(
        PROJECT_ROOT, config.experiment.output_dir,
        "checkpoints", f"gen_{args.generation}"
    )

    logger.info(f"Training generation {args.generation}")
    logger.info(f"Training data: {train_data}")
    logger.info(f"Output: {output_dir}")

    trainer = ModelTrainer(config)
    metrics = trainer.train(train_data, val_data, output_dir, args.generation)

    print(f"\nTraining complete! Metrics: {metrics}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
