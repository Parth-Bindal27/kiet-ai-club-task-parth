"""Prepare the dataset (download + split).

Usage:
    python scripts/prepare_data.py --config configs/base.yaml
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.data.dataset import prepare_dataset
from src.utils.seeding import set_seed


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.experiment.seed)

    paths = prepare_dataset(config, PROJECT_ROOT)
    print(f"Dataset prepared:")
    for k, v in paths.items():
        # Count lines
        with open(v) as f:
            n = sum(1 for _ in f)
        print(f"  {k}: {v} ({n} examples)")


if __name__ == "__main__":
    main()
