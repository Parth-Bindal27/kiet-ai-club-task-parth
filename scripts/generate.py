"""Generate synthetic data from a trained model.

Usage:
    python scripts/generate.py --config configs/base.yaml --generation 0
"""
import argparse
import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.data.dataset import prepare_dataset
from src.generation.generator import SyntheticDataGenerator
from src.generation.self_instruct import SelfInstructGenerator
from src.utils.seeding import set_seed
from src.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data from a trained model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--generation", type=int, required=True, help="Source model generation (data will be gen+1)")
    parser.add_argument("--model-path", type=str, default=None, help="Override model checkpoint path")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.generation.seed)
    setup_logging(os.path.join(PROJECT_ROOT, config.experiment.log_dir), config.experiment.name)
    logger = logging.getLogger(__name__)

    # Determine model path
    model_path = args.model_path or os.path.join(
        PROJECT_ROOT, config.experiment.output_dir,
        "checkpoints", f"gen_{args.generation}"
    )

    # Prepare dataset to get instructions
    data_paths = prepare_dataset(config, PROJECT_ROOT)

    # Select instructions
    si = SelfInstructGenerator()
    instructions = si.select_instructions(
        train_data_path=data_paths["train"],
        num_samples=config.generation.num_samples,
        seed=config.generation.seed,
    )

    # Generate
    output_gen = args.generation + 1
    output_path = os.path.join(
        PROJECT_ROOT, config.data.data_dir,
        "synthetic", f"generation_{output_gen}", "synthetic_data.jsonl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    generator = SyntheticDataGenerator(config)
    stats = generator.generate(model_path, instructions, output_path, output_gen)

    print(f"\nGeneration complete! Stats: {stats}")
    print(f"Synthetic data saved to: {output_path}")


if __name__ == "__main__":
    main()
