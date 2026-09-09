"""Generate the research report from experiment results.

Usage:
    python scripts/generate_report.py --output results/reports/report.md
"""
import argparse
import glob
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.visualization.report import generate_report
from src.visualization.plots import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="Generate research report.")
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "results", "reports", "report.md"))
    parser.add_argument("--figures-dir", type=str, default=os.path.join(PROJECT_ROOT, "results", "figures"))
    args = parser.parse_args()

    # Find all experiment directories
    results_dir = os.path.join(PROJECT_ROOT, "results", "experiments")
    exp_dirs = sorted(glob.glob(os.path.join(results_dir, "*")))
    exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]

    if not exp_dirs:
        print("No experiments found. Run experiments first.")
        return

    print(f"Found {len(exp_dirs)} experiments:")
    for d in exp_dirs:
        print(f"  - {os.path.basename(d)}")

    # Generate plots
    print("\nGenerating plots...")
    plots = generate_all_plots(exp_dirs, args.figures_dir)
    print(f"Generated {len(plots)} plots")

    # Generate report
    print("\nGenerating report...")
    path = generate_report(exp_dirs, args.output, args.figures_dir)
    print(f"Report saved to: {path}")


if __name__ == "__main__":
    main()
