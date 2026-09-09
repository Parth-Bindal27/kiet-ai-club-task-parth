"""Unit tests for the model collapse study project."""
import os
import sys
import json
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestConfig(unittest.TestCase):
    """Tests for the configuration system."""

    def test_load_base_config(self):
        from src.config import Config
        config = Config.from_yaml(os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
        self.assertEqual(config.experiment.name, "base")
        self.assertEqual(config.experiment.seed, 42)
        self.assertEqual(config.model.name, "gpt2")
        self.assertEqual(config.training.learning_rate, 5e-5)
        self.assertEqual(config.data.train_ratio, 0.8)

    def test_load_smoke_config(self):
        from src.config import Config
        config = Config.from_yaml(os.path.join(PROJECT_ROOT, "configs", "smoke_test.yaml"))
        self.assertEqual(config.experiment.name, "smoke_test")
        self.assertEqual(config.experiment.generations, 1)
        self.assertEqual(config.data.max_train_samples, 50)

    def test_config_inheritance(self):
        from src.config import Config
        config = Config.from_yaml(
            os.path.join(PROJECT_ROOT, "configs", "experiments", "exp_b_synthetic_100.yaml")
        )
        # Should inherit base values
        self.assertEqual(config.model.name, "gpt2")
        self.assertEqual(config.training.learning_rate, 5e-5)
        # Should override specific values
        self.assertEqual(config.data.synthetic_ratio, 1.0)
        self.assertEqual(config.experiment.name, "exp_b_synthetic_100")

    def test_config_to_dict(self):
        from src.config import Config
        config = Config.from_yaml(os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
        d = config.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("experiment", d)
        self.assertIn("model", d)


class TestSeeding(unittest.TestCase):
    """Tests for deterministic seeding."""

    def test_set_seed_deterministic(self):
        import numpy as np
        from src.utils.seeding import set_seed

        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        self.assertTrue(all(x == y for x, y in zip(a, b)))

    def test_different_seeds_different_results(self):
        import numpy as np
        from src.utils.seeding import set_seed

        set_seed(42)
        a = np.random.rand(10)
        set_seed(123)
        b = np.random.rand(10)
        self.assertFalse(all(x == y for x, y in zip(a, b)))


class TestSplitter(unittest.TestCase):
    """Tests for deterministic dataset splitting."""

    def test_split_sizes(self):
        from src.data.splitter import deterministic_split
        train, val, test = deterministic_split(100, 0.8, 0.1, 0.1, seed=42)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(test), 10)

    def test_split_deterministic(self):
        from src.data.splitter import deterministic_split
        t1, v1, te1 = deterministic_split(100, 0.8, 0.1, 0.1, seed=42)
        t2, v2, te2 = deterministic_split(100, 0.8, 0.1, 0.1, seed=42)
        self.assertEqual(t1, t2)
        self.assertEqual(v1, v2)
        self.assertEqual(te1, te2)

    def test_no_overlap(self):
        from src.data.splitter import deterministic_split
        train, val, test = deterministic_split(100, 0.8, 0.1, 0.1, seed=42)
        all_indices = set(train + val + test)
        self.assertEqual(len(all_indices), 100)


class TestDeduplication(unittest.TestCase):
    """Tests for deduplication methods."""

    def test_exact_dedup(self):
        from src.filtering.deduplication import ExactDeduplicator
        deduper = ExactDeduplicator()
        examples = [
            {"response": "Hello world"},
            {"response": "Hello world"},
            {"response": "Different text"},
        ]
        result, stats = deduper.deduplicate(examples)
        self.assertEqual(len(result), 2)
        self.assertEqual(stats["removed_count"], 1)

    def test_normalized_dedup(self):
        from src.filtering.deduplication import NormalizedDeduplicator
        deduper = NormalizedDeduplicator()
        examples = [
            {"response": "Hello World"},
            {"response": "hello world"},
            {"response": "Different"},
        ]
        result, stats = deduper.deduplicate(examples)
        self.assertEqual(len(result), 2)

    def test_no_duplicates(self):
        from src.filtering.deduplication import ExactDeduplicator
        deduper = ExactDeduplicator()
        examples = [
            {"response": "A"},
            {"response": "B"},
            {"response": "C"},
        ]
        result, stats = deduper.deduplicate(examples)
        self.assertEqual(len(result), 3)
        self.assertEqual(stats["removed_count"], 0)


class TestDuplicationMetrics(unittest.TestCase):
    """Tests for duplication evaluation metrics."""

    def test_exact_duplicate_rate(self):
        from src.evaluation.duplication import calculate_exact_duplicate_rate
        texts = ["a", "b", "a", "c"]
        rate = calculate_exact_duplicate_rate(texts)
        self.assertAlmostEqual(rate, 0.25)

    def test_no_duplicates(self):
        from src.evaluation.duplication import calculate_exact_duplicate_rate
        texts = ["a", "b", "c", "d"]
        rate = calculate_exact_duplicate_rate(texts)
        self.assertEqual(rate, 0.0)


class TestCollapseIndex(unittest.TestCase):
    """Tests for Model Collapse Index."""

    def test_no_collapse(self):
        from src.evaluation.collapse_index import ModelCollapseIndex
        mci = ModelCollapseIndex()
        baseline = {
            "rouge": {"rougeL": 0.5},
            "lexical_diversity": {"type_token_ratio": 0.4},
            "duplication": {"self_repetition": 0.1},
            "rare_features": {"retention": {"rare": 0.3}},
        }
        # Same metrics as baseline → no collapse
        result = mci.calculate(baseline, baseline)
        self.assertAlmostEqual(result["mci"], 0.0, places=1)

    def test_mci_bounded(self):
        from src.evaluation.collapse_index import ModelCollapseIndex
        mci = ModelCollapseIndex()
        baseline = {
            "rouge": {"rougeL": 0.5},
            "lexical_diversity": {"type_token_ratio": 0.4},
            "duplication": {"self_repetition": 0.1},
            "rare_features": {"retention": {"rare": 0.3}},
        }
        collapsed = {
            "rouge": {"rougeL": 0.0},
            "lexical_diversity": {"type_token_ratio": 0.0},
            "embedding_distance": {"js_divergence": 0.7},
            "duplication": {"self_repetition": 1.0},
            "rare_features": {"retention": {"rare": 0.0}},
        }
        result = mci.calculate(collapsed, baseline)
        self.assertGreater(result["mci"], 0.0)
        self.assertLessEqual(result["mci"], 1.0)


class TestValidation(unittest.TestCase):
    """Tests for example validation."""

    def test_valid_example(self):
        from src.filtering.validation import validate_example
        ex = {"instruction": "Write a poem", "response": "Roses are red..."}
        valid, reason = validate_example(ex)
        self.assertTrue(valid)

    def test_empty_response(self):
        from src.filtering.validation import validate_example
        ex = {"instruction": "Write a poem", "response": ""}
        valid, reason = validate_example(ex)
        self.assertFalse(valid)


class TestJSONLIO(unittest.TestCase):
    """Tests for JSONL I/O."""

    def test_write_and_read(self):
        from src.data.dataset import load_jsonl

        data = [
            {"instruction": "Hello", "output": "World"},
            {"instruction": "Foo", "output": "Bar"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name

        try:
            loaded = load_jsonl(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["instruction"], "Hello")
        finally:
            os.unlink(path)


class TestComputeTracker(unittest.TestCase):
    """Tests for compute tracker."""

    def test_start_stop(self):
        from src.utils.compute import ComputeTracker
        tracker = ComputeTracker()
        tracker.start()
        import time
        time.sleep(0.1)
        tracker.stop()
        stats = tracker.get_stats()
        self.assertIn("wall_time_seconds", stats)
        self.assertGreater(stats["wall_time_seconds"], 0.05)

    def test_context_manager(self):
        from src.utils.compute import ComputeTracker
        tracker = ComputeTracker()
        with tracker.track("test_op"):
            import time
            time.sleep(0.05)
        self.assertIn("test_op", tracker.stats)


if __name__ == "__main__":
    unittest.main()
