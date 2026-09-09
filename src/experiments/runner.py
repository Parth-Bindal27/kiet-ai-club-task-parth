"""Experiment execution pipeline.

Orchestrates the full multi-generation training loop:
  Gen 0: train on real data → evaluate
  Gen N: generate synthetic from Model N-1 → filter → mix → train → evaluate

Every model is evaluated on the SAME untouched real test set.
"""
import os
import json
import time
import logging
from typing import Optional

from src.config import Config
from src.data.dataset import prepare_dataset, load_jsonl
from src.data.mixer import mix_datasets
from src.models.trainer import ModelTrainer
from src.generation.generator import SyntheticDataGenerator
from src.generation.self_instruct import SelfInstructGenerator
from src.filtering.pipeline import FilteringPipeline
from src.evaluation.evaluator import Evaluator
from src.experiments.tracker import ExperimentTracker
from src.utils.seeding import set_seed
from src.utils.compute import ComputeTracker

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs a complete multi-generation model collapse experiment."""

    def __init__(self, config_path: str, project_root: str = "."):
        self.config = Config.from_yaml(config_path)
        self.project_root = project_root
        self.experiment_name = self.config.experiment.name

        # Set up directories
        self.experiment_dir = os.path.join(
            project_root, self.config.experiment.output_dir,
            "experiments", self.experiment_name
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Initialize components
        self.tracker = ExperimentTracker(self.experiment_name, self.experiment_dir)
        self.tracker.log_config(self.config.to_dict())

        self.trainer = ModelTrainer(self.config)
        self.generator = SyntheticDataGenerator(self.config)
        self.filtering_pipeline = FilteringPipeline(self.config.filtering)
        self.self_instruct = SelfInstructGenerator()

        # Data paths (populated during run)
        self.data_paths = {}

    def run(self) -> str:
        """Run the full experiment pipeline.

        Returns:
            Path to the experiment results directory.
        """
        set_seed(self.config.experiment.seed)
        logger.info(f"Starting experiment: {self.experiment_name}")
        logger.info(f"Generations: {self.config.experiment.generations}")
        logger.info(f"Synthetic ratio: {self.config.data.synthetic_ratio}")

        # Step 1: Prepare dataset
        logger.info("Preparing dataset...")
        self.data_paths = prepare_dataset(self.config, self.project_root)
        logger.info(f"Data paths: {self.data_paths}")

        # Initialize evaluator with the SACRED test set
        evaluator = Evaluator(self.config, self.data_paths["test"])

        # Step 2: Run each generation
        num_generations = self.config.experiment.generations
        real_data_path = self.data_paths["train"]
        val_data_path = self.data_paths["validation"]

        for gen in range(num_generations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"GENERATION {gen}")
            logger.info(f"{'='*60}")

            model_dir = os.path.join(self.experiment_dir, "checkpoints", f"gen_{gen}")
            os.makedirs(model_dir, exist_ok=True)

            # Determine training data for this generation
            if gen == 0:
                # Generation 0 always trains on real data
                train_data_path = real_data_path
                logger.info("Training on REAL data (Generation 0 baseline)")
            else:
                # Generate synthetic data from previous model
                prev_model_dir = os.path.join(
                    self.experiment_dir, "checkpoints", f"gen_{gen - 1}"
                )
                synthetic_data_path = self._generate_synthetic(
                    gen, prev_model_dir, real_data_path
                )

                # Apply filtering if enabled
                if self.config.filtering.enabled:
                    synthetic_data_path = self._filter_synthetic(
                        gen, synthetic_data_path
                    )

                # Mix real and synthetic data based on config
                # Curriculum Mixing
                if getattr(self.config.experiment, "curriculum_mixing", False):
                    synthetic_ratio = min(1.0, 0.2 * gen)
                else:
                    synthetic_ratio = self.config.data.synthetic_ratio
                if synthetic_ratio >= 1.0:
                    # 100% synthetic
                    train_data_path = synthetic_data_path
                    logger.info(f"Training on 100% SYNTHETIC data")
                elif synthetic_ratio <= 0.0:
                    # 100% real (control condition)
                    train_data_path = real_data_path
                    logger.info(f"Training on 100% REAL data")
                else:
                    # Mixed
                    real_ratio = 1.0 - synthetic_ratio
                    mix_output = os.path.join(
                        self.experiment_dir, "data", f"mixed_gen_{gen}.jsonl"
                    )
                    os.makedirs(os.path.dirname(mix_output), exist_ok=True)
                    train_data_path = mix_datasets(
                        real_data_path, synthetic_data_path,
                        real_ratio=real_ratio,
                        seed=self.config.experiment.seed + gen,
                    )
                    logger.info(
                        f"Training on MIXED data "
                        f"({real_ratio:.0%} real + {synthetic_ratio:.0%} synthetic)"
                    )

            # Train model
            logger.info(f"Training Model {gen}...")
            compute = ComputeTracker()
            compute.start()
            try:
                training_stats = self.trainer.train(
                    train_data_path=train_data_path,
                    val_data_path=val_data_path,
                    output_dir=model_dir,
                    generation=gen,
                )
            except Exception as e:
                logger.error(f"Training failed for generation {gen}: {e}")
                training_stats = {"error": str(e)}
            compute.stop()
            training_stats["compute"] = compute.get_stats()

            # Evaluate model on the SACRED test set
            logger.info(f"Evaluating Model {gen} on test set...")
            synthetic_for_eval = None
            if gen > 0:
                synthetic_for_eval = synthetic_data_path

            try:
                metrics = evaluator.evaluate_model(
                    model_path=model_dir,
                    generation=gen,
                    synthetic_data_path=synthetic_for_eval,
                )
            except Exception as e:
                logger.error(f"Evaluation failed for generation {gen}: {e}")
                metrics = {"error": str(e)}

            # Log results
            gen_stats = {}
            if gen > 0 and synthetic_for_eval:
                gen_stats["synthetic_data_path"] = synthetic_for_eval
            self.tracker.log_generation(gen, metrics, training_stats, gen_stats)
            self.tracker.save()

            logger.info(f"Generation {gen} complete. Metrics: {json.dumps(metrics, indent=2, default=str)}")

        logger.info(f"\nExperiment '{self.experiment_name}' completed!")
        logger.info(f"Results saved to: {self.experiment_dir}")

        return self.experiment_dir

    def _generate_synthetic(
        self, generation: int, model_path: str, real_data_path: str
    ) -> str:
        """Generate synthetic data using the previous generation's model."""
        logger.info(f"Generating synthetic data for generation {generation}...")

        # Select instructions to use as prompts
        instructions = self.self_instruct.select_instructions(
            train_data_path=real_data_path,
            num_samples=self.config.generation.num_samples,
            seed=self.config.experiment.seed + generation,
        )

        # Generate
        output_dir = os.path.join(
            self.project_root, self.config.data.data_dir,
            "synthetic", f"generation_{generation}"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "synthetic_data.jsonl")

        gen_stats = self.generator.generate(
            model_path=model_path,
            instructions=instructions,
            output_path=output_path,
            generation=generation,
        )
        logger.info(f"Generated {gen_stats['num_generated']} synthetic examples")
        return output_path

    def _filter_synthetic(self, generation: int, synthetic_data_path: str) -> str:
        """Apply filtering pipeline to synthetic data."""
        logger.info(f"Filtering synthetic data for generation {generation}...")

        examples = load_jsonl(synthetic_data_path)
        filtered_examples, stats = self.filtering_pipeline.run(
            examples=examples,
            generation=generation,
            output_dir=os.path.dirname(synthetic_data_path),
        )

        # Save filtered data
        filtered_path = synthetic_data_path.replace(".jsonl", "_filtered.jsonl")
        with open(filtered_path, 'w', encoding='utf-8') as f:
            for item in filtered_examples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        self.tracker.log_filtering_stats(generation, stats)
        logger.info(
            f"Filtering: {stats.get('original_count', '?')} → "
            f"{stats.get('final_count', len(filtered_examples))} examples"
        )
        return filtered_path
