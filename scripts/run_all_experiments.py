"""Run all experiment configurations.

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --experiments exp_a exp_b
"""
import argparse
import glob
import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.experiments.runner import ExperimentRunner
from src.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run all experiment configurations.")
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="Specific experiment prefixes to run (e.g., exp_a exp_b). Default: all."
    )
    parser.add_argument(
        "--config-dir", type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "experiments"),
        help="Directory containing experiment configs."
    )
    args = parser.parse_args()

    setup_logging(os.path.join(PROJECT_ROOT, "logs"), "all_experiments")
    logger = logging.getLogger(__name__)

    # Find all experiment configs
    config_files = sorted(glob.glob(os.path.join(args.config_dir, "*.yaml")))

    if args.experiments:
        config_files = [
            f for f in config_files
            if any(exp in os.path.basename(f) for exp in args.experiments)
        ]

    if not config_files:
        print("No experiment configs found.")
        return

    print(f"Found {len(config_files)} experiment configs:")
    for f in config_files:
        print(f"  - {os.path.basename(f)}")
    print()

    results = {}
    for i, config_path in enumerate(config_files, 1):
        exp_name = os.path.basename(config_path).replace(".yaml", "")
        print(f"\n{'='*60}")
        print(f"[{i}/{len(config_files)}] Running: {exp_name}")
        print(f"{'='*60}")

        try:
            runner = ExperimentRunner(config_path, project_root=PROJECT_ROOT)
            result_dir = runner.run()
            results[exp_name] = {"status": "SUCCESS", "dir": result_dir}
            print(f"✓ {exp_name} completed successfully")
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            results[exp_name] = {"status": "FAILED", "error": str(e)}
            print(f"✗ {exp_name} failed: {e}")

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS SUMMARY")
    print(f"{'='*60}")
    for name, result in results.items():
        print(f"  {name}: {result['status']}")


if __name__ == "__main__":
    main()
