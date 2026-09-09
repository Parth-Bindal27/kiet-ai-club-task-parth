"""Configuration system for the model collapse experiment.

Uses dataclasses with YAML loading and support for config overrides.
All experimental parameters are configurable rather than hard-coded.
"""
import os
import yaml
import collections.abc
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict


@dataclass
class ExperimentConfig:
    """Top-level experiment settings."""
    name: str = "base"
    seed: int = 42
    generations: int = 5
    output_dir: str = "results"
    log_dir: str = "logs"
    curriculum_mixing: bool = False


@dataclass
class ModelConfig:
    """Model architecture and LoRA settings."""
    name: str = "gpt2"
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    fp16: bool = False
    weight_decay: float = 0.01

    def __post_init__(self):
        self.learning_rate = float(self.learning_rate)
        self.batch_size = int(self.batch_size)
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        self.num_epochs = int(self.num_epochs)
        self.warmup_ratio = float(self.warmup_ratio)
        self.max_seq_length = int(self.max_seq_length)
        self.weight_decay = float(self.weight_decay)
        if isinstance(self.fp16, str):
            self.fp16 = self.fp16.lower() in ("true", "1", "yes")


@dataclass
class GenerationConfig:
    """Synthetic data generation settings."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    repetition_penalty: float = 1.2
    num_samples: int = 5000
    batch_size: int = 8
    seed: int = 42
    prompt_template: str = "alpaca"

    def __post_init__(self):
        self.temperature = float(self.temperature)
        self.top_p = float(self.top_p)
        self.max_new_tokens = int(self.max_new_tokens)
        self.repetition_penalty = float(self.repetition_penalty)
        self.num_samples = int(self.num_samples)
        self.batch_size = int(self.batch_size)
        self.seed = int(self.seed)


@dataclass
class DataConfig:
    """Dataset and data mixing settings."""
    dataset_name: str = "yahma/alpaca-cleaned"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    synthetic_ratio: float = 1.0
    max_train_samples: Optional[int] = None
    data_dir: str = "data"

    def __post_init__(self):
        self.train_ratio = float(self.train_ratio)
        self.val_ratio = float(self.val_ratio)
        self.test_ratio = float(self.test_ratio)
        self.synthetic_ratio = float(self.synthetic_ratio)
        if self.max_train_samples is not None:
            self.max_train_samples = int(self.max_train_samples)


@dataclass
class FilteringConfig:
    """Synthetic data filtering settings."""
    enabled: bool = True
    deduplication: bool = True
    dedup_method: str = "normalized"
    quality_filter: bool = True
    min_quality_score: float = 5.0
    min_length: int = 10
    max_length: int = 2048
    max_repetition_ratio: float = 0.5
    entropy_filter: bool = False

    def __post_init__(self):
        if isinstance(self.enabled, str):
            self.enabled = self.enabled.lower() in ("true", "1", "yes")
        if isinstance(self.deduplication, str):
            self.deduplication = self.deduplication.lower() in ("true", "1", "yes")
        if isinstance(self.quality_filter, str):
            self.quality_filter = self.quality_filter.lower() in ("true", "1", "yes")
        self.min_quality_score = float(self.min_quality_score)
        self.min_length = int(self.min_length)
        self.max_length = int(self.max_length)
        self.max_repetition_ratio = float(self.max_repetition_ratio)
        if isinstance(self.entropy_filter, str):
            self.entropy_filter = self.entropy_filter.lower() in ("true", "1", "yes")


@dataclass
class EvaluationConfig:
    """Evaluation settings."""
    metrics: List[str] = field(default_factory=lambda: [
        "perplexity", "rouge", "diversity", "distribution_shift",
        "duplication", "rare_features", "memorization"
    ])
    batch_size: int = 8
    max_eval_samples: Optional[int] = None


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        """Load configuration from a YAML file with optional overrides.
        
        Supports inheritance: if the YAML contains a 'base_config' key,
        that base config is loaded first, then overridden by the current file.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}

        # Support config inheritance via base_config key
        base_path = yaml_data.pop("base_config", None)
        if base_path:
            # Resolve relative to the config file's directory
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.path.dirname(yaml_path), base_path)
            with open(base_path, 'r') as f:
                base_data = yaml.safe_load(f) or {}
            cls._deep_update(base_data, yaml_data)
            yaml_data = base_data

        if overrides:
            cls._deep_update(yaml_data, overrides)

        return cls(
            experiment=ExperimentConfig(**yaml_data.get("experiment", {})),
            model=ModelConfig(**yaml_data.get("model", {})),
            training=TrainingConfig(**yaml_data.get("training", {})),
            generation=GenerationConfig(**yaml_data.get("generation", {})),
            data=DataConfig(**yaml_data.get("data", {})),
            filtering=FilteringConfig(**yaml_data.get("filtering", {})),
            evaluation=EvaluationConfig(**yaml_data.get("evaluation", {})),
        )

    def to_dict(self) -> dict:
        """Convert config to a plain dictionary."""
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _deep_update(base: dict, override: dict) -> dict:
        """Recursively update base dict with override dict."""
        for k, v in override.items():
            if isinstance(v, collections.abc.Mapping) and isinstance(base.get(k), collections.abc.Mapping):
                Config._deep_update(base[k], v)
            else:
                base[k] = v
        return base
