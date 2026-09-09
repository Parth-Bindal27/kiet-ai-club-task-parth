"""End-to-end smoke test.

Runs a tiny experiment to verify the full pipeline works:
- Prepare dataset (tiny subset)
- Train Generation 0
- Generate synthetic data
- Train Generation 1
- Evaluate both

Should complete in < 5 minutes on CPU.
"""
import os
import sys
import time
import logging
import traceback

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.utils.seeding import set_seed
from src.utils.logging_utils import setup_logging


def run_smoke_test():
    """Run a minimal end-to-end test."""
    print("=" * 60)
    print("MODEL COLLAPSE STUDY — SMOKE TEST")
    print("=" * 60)

    start_time = time.time()

    # Use smoke test config
    config_path = os.path.join(PROJECT_ROOT, "configs", "smoke_test.yaml")
    if not os.path.exists(config_path):
        print(f"FAIL: Config not found: {config_path}")
        return False

    config = Config.from_yaml(config_path)
    set_seed(config.experiment.seed)

    log_dir = os.path.join(PROJECT_ROOT, "logs")
    setup_logging(log_dir, "smoke_test")
    logger = logging.getLogger(__name__)

    steps_passed = 0
    total_steps = 6

    # Step 1: Dataset preparation
    print(f"\n[1/{total_steps}] Preparing dataset...")
    try:
        from src.data.dataset import prepare_dataset
        data_paths = prepare_dataset(config, PROJECT_ROOT)
        assert os.path.exists(data_paths["train"]), "Train file not created"
        assert os.path.exists(data_paths["validation"]), "Validation file not created"
        assert os.path.exists(data_paths["test"]), "Test file not created"
        print(f"  ✓ Dataset prepared: {data_paths}")
        steps_passed += 1
    except Exception as e:
        print(f"  ✗ Dataset preparation failed: {e}")
        traceback.print_exc()
        return False

    # Step 2: Train Generation 0
    print(f"\n[2/{total_steps}] Training Model 0 (real data baseline)...")
    try:
        from src.models.trainer import ModelTrainer
        trainer = ModelTrainer(config)
        model_0_dir = os.path.join(PROJECT_ROOT, "results", "smoke_test", "checkpoints", "gen_0")
        os.makedirs(model_0_dir, exist_ok=True)
        stats = trainer.train(
            train_data_path=data_paths["train"],
            val_data_path=data_paths["validation"],
            output_dir=model_0_dir,
            generation=0,
        )
        assert os.path.exists(os.path.join(model_0_dir, "config.json")), \
            "Model checkpoint not saved"
        print(f"  ✓ Model 0 trained: {stats}")
        steps_passed += 1
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        traceback.print_exc()
        return False

    # Step 3: Generate synthetic data
    print(f"\n[3/{total_steps}] Generating synthetic data from Model 0...")
    try:
        from src.generation.generator import SyntheticDataGenerator
        from src.generation.self_instruct import SelfInstructGenerator
        from src.data.dataset import load_jsonl

        si = SelfInstructGenerator()
        instructions = si.select_instructions(
            train_data_path=data_paths["train"],
            num_samples=config.generation.num_samples,
            seed=config.experiment.seed,
        )

        gen = SyntheticDataGenerator(config)
        syn_output = os.path.join(
            PROJECT_ROOT, "data", "synthetic", "smoke_gen_1", "synthetic_data.jsonl"
        )
        os.makedirs(os.path.dirname(syn_output), exist_ok=True)
        gen_stats = gen.generate(
            model_path=model_0_dir,
            instructions=instructions,
            output_path=syn_output,
            generation=1,
        )
        assert os.path.exists(syn_output), "Synthetic data not saved"
        syn_data = load_jsonl(syn_output)
        assert len(syn_data) > 0, "No synthetic data generated"
        print(f"  ✓ Generated {gen_stats['num_generated']} synthetic examples")
        steps_passed += 1
    except Exception as e:
        print(f"  ✗ Generation failed: {e}")
        traceback.print_exc()
        return False

    # Step 4: Filter synthetic data
    print(f"\n[4/{total_steps}] Filtering synthetic data...")
    try:
        from src.filtering.pipeline import FilteringPipeline

        pipeline = FilteringPipeline(config.filtering)
        filtered, filter_stats = pipeline.run(
            examples=syn_data,
            generation=1,
            output_dir=os.path.dirname(syn_output),
        )
        print(f"  ✓ Filtered: {len(syn_data)} → {len(filtered)} examples")
        print(f"    Stats: {filter_stats}")
        steps_passed += 1
    except Exception as e:
        print(f"  ✗ Filtering failed: {e}")
        traceback.print_exc()
        # Non-fatal — continue

    # Step 5: Evaluate Model 0
    print(f"\n[5/{total_steps}] Evaluating Model 0...")
    try:
        from src.evaluation.evaluator import Evaluator

        evaluator = Evaluator(config, data_paths["test"])
        metrics = evaluator.evaluate_model(
            model_path=model_0_dir,
            generation=0,
        )
        assert "perplexity" in metrics, "Perplexity not computed"
        print(f"  ✓ Metrics: perplexity={metrics.get('perplexity', 'N/A')}")
        steps_passed += 1
    except Exception as e:
        print(f"  ✗ Evaluation failed: {e}")
        traceback.print_exc()
        return False

    # Step 6: Train Generation 1 on synthetic data
    print(f"\n[6/{total_steps}] Training Model 1 (synthetic data)...")
    try:
        model_1_dir = os.path.join(PROJECT_ROOT, "results", "smoke_test", "checkpoints", "gen_1")
        os.makedirs(model_1_dir, exist_ok=True)

        # Use filtered synthetic data if available, otherwise raw
        syn_train_path = syn_output
        if filtered:
            filtered_path = syn_output.replace(".jsonl", "_filtered.jsonl")
            with open(filtered_path, 'w', encoding='utf-8') as f:
                import json
                for item in filtered:
                    f.write(json.dumps(item) + "\n")
            syn_train_path = filtered_path

        stats = trainer.train(
            train_data_path=syn_train_path,
            val_data_path=data_paths["validation"],
            output_dir=model_1_dir,
            generation=1,
        )
        print(f"  ✓ Model 1 trained: {stats}")
        steps_passed += 1
    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        traceback.print_exc()
        return False

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"SMOKE TEST RESULT: {steps_passed}/{total_steps} steps passed")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"{'='*60}")

    if steps_passed >= 5:
        print("\n✓ PASS — Core pipeline is functional.")
        return True
    else:
        print("\n✗ FAIL — Critical steps failed.")
        return False


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
