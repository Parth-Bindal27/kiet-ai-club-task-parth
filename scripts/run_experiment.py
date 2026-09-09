"""Run a complete experiment across all generations.

Usage:
    python scripts/run_experiment.py --config configs/experiments/exp_b_synthetic_100.yaml
"""
import argparse
import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.runner import ExperimentRunner
from src.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run a complete multi-generation experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    setup_logging(os.path.join(PROJECT_ROOT, "logs"), "experiment")
    logger = logging.getLogger(__name__)

    logger.info(f"Starting experiment from config: {args.config}")
    runner = ExperimentRunner(args.config, project_root=PROJECT_ROOT)
    result_dir = runner.run()

    print(f"\nExperiment complete! Results saved to: {result_dir}")


if __name__ == "__main__":
    main()
