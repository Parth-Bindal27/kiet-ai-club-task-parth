# Model Collapse Under Iterative Synthetic Data Training

A complete, reproducible research system investigating whether repeatedly training a language model on model-generated synthetic data causes "model collapse" — and under what conditions it can be delayed or prevented.

## Research Question

> How does repeated training on model-generated synthetic data affect model performance, diversity, distributional coverage, memorization, and rare-feature retention across successive generations, and can careful synthetic-data curation or mixing real data prevent or delay model collapse?

## Quick Start

### 1. Environment Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 2. Prepare Dataset

```bash
python3 scripts/prepare_data.py --config configs/base.yaml
```

This downloads the [Alpaca-Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset and creates deterministic train/validation/test splits.

### 3. Run Smoke Test

```bash
python3 scripts/smoke_test.py
```

Runs a tiny end-to-end experiment (50 training samples, 1 generation, 10 synthetic samples) to verify everything works.

### 4. Run a Single Experiment

```bash
# Train Generation 0 (baseline on real data)
python3 scripts/train.py --config configs/base.yaml --generation 0

# Generate synthetic data from Model 0
python3 scripts/generate.py --config configs/base.yaml --generation 0

# Train Generation 1 on synthetic data
python3 scripts/train.py --config configs/base.yaml --generation 1

# Evaluate any model
python3 scripts/evaluate.py --config configs/base.yaml --generation 0
```

### 5. Run a Full Experiment (All Generations)

```bash
python3 scripts/run_experiment.py --config configs/experiments/exp_b_synthetic_100.yaml
```

### 6. Run All Experiments

```bash
python3 scripts/run_all_experiments.py
```

### 7. Generate Research Report

```bash
python3 scripts/generate_report.py --output results/reports/report.md
```

### 8. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Project Structure

```
model-collapse-study/
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml                    # Default configuration
│   ├── smoke_test.yaml              # Tiny config for testing
│   └── experiments/                 # Per-experiment configs
│       ├── exp_a_real_baseline.yaml
│       ├── exp_b_synthetic_100.yaml
│       ├── exp_c_mixed_*.yaml
│       ├── exp_d_filtering_*.yaml
│       ├── exp_e_temp_*.yaml
│       └── exp_f_inject_*.yaml
├── src/
│   ├── config.py                    # YAML + dataclass config system
│   ├── hardware.py                  # Hardware auto-detection
│   ├── data/                        # Data loading, splitting, mixing
│   ├── models/                      # Model loading and training
│   ├── generation/                  # Synthetic data generation
│   ├── filtering/                   # Dedup, quality, validation
│   ├── evaluation/                  # All evaluation metrics
│   ├── experiments/                 # Experiment runner and tracker
│   ├── visualization/               # Plots and report generation
│   └── utils/                       # Seeding, logging, compute tracking
├── scripts/                         # CLI entry points
├── dashboard/                       # Streamlit dashboard
├── tests/                           # Unit and integration tests
├── data/
│   ├── raw/                         # Original downloaded data
│   ├── processed/{train,validation,test}/
│   └── synthetic/generation_N/
├── results/
│   ├── experiments/                 # Per-experiment results
│   ├── figures/                     # Generated plots
│   ├── reports/                     # Generated reports
│   └── checkpoints/                 # Model checkpoints
└── logs/
```

---

## Dataset

**Alpaca-Cleaned** (`yahma/alpaca-cleaned`): ~52,000 instruction-following examples.

| Split | Size | Purpose |
|---|---|---|
| Train | 80% (~41,600) | Training models |
| Validation | 10% (~5,200) | Model selection / hyperparameter tuning |
| Test | 10% (~5,200) | **SACRED** — evaluation only |

> **⚠️ DATA INTEGRITY**: The test set is NEVER used for training, generation, prompt construction, filtering, model selection, or hyperparameter tuning. Every generation is evaluated on the exact same held-out test set.

Split is deterministic (seed=42) and documented in `src/data/splitter.py`.

---

## Experimental Pipeline

```
Generation 0:  REAL DATA → Train Model 0 → Evaluate on TEST SET
Generation 1:  Model 0 generates Synthetic Data 1 → [Filter] → Train Model 1 → Evaluate on TEST SET
Generation 2:  Model 1 generates Synthetic Data 2 → [Filter] → Train Model 2 → Evaluate on TEST SET
...
Generation N:  Model N-1 generates Synthetic Data N → [Filter] → Train Model N → Evaluate on TEST SET
```

Each generation starts from the **base GPT-2 checkpoint** (not from the previous model), isolating the effect of data quality from weight accumulation.

---

## Experiments

| ID | Name | Description |
|---|---|---|
| A | Real Baseline | Every generation trains on real data (control) |
| B | 100% Synthetic | Each generation trains on previous model's synthetic output |
| C | Mixed Ratios | 50%, 75%, 90% real data mixed with synthetic |
| D | Filtering | No filter, dedup only, quality only, dedup + quality |
| E | Temperature | Generation at temperature 0.2 vs 0.7 |
| F | Real Injection | 0%, 10%, 50% real data injected each generation |

---

## Evaluation Metrics

- **Perplexity**: Language modeling quality on real test set
- **ROUGE**: Generation quality vs reference answers
- **Lexical Diversity**: Vocabulary size, type-token ratio, unique n-grams
- **Semantic Diversity**: Embedding variance, pairwise cosine similarity
- **Distribution Shift**: Jensen-Shannon divergence, Wasserstein distance
- **Duplication**: Exact/normalized duplicate rates, n-gram overlap
- **Rare Feature Retention**: Performance on low-frequency topics
- **Memorization**: Cross-generation text overlap
- **Model Collapse Index (MCI)**: Project-specific composite metric (NOT a universal standard)

---

## Configuration

All experiments are configured via YAML. See `configs/base.yaml` for the full schema.

```yaml
experiment:
  name: "my_experiment"
  seed: 42
  generations: 5

model:
  name: "gpt2"
  use_lora: false

training:
  learning_rate: 5e-5
  batch_size: 4
  num_epochs: 3

generation:
  temperature: 0.7
  num_samples: 5000

data:
  synthetic_ratio: 1.0

filtering:
  deduplication: true
  quality_filter: true
  min_quality_score: 5
```

---

## Hardware Requirements

- **Minimum**: CPU with 8 GB RAM (smoke test only)
- **Recommended**: Apple Silicon (M1/M2/M3/M4/M5) with 16+ GB unified memory, or NVIDIA GPU with 8+ GB VRAM
- **Tested on**: Apple M5, 16 GB, MPS backend

The system auto-detects hardware and adjusts batch sizes accordingly.

---

## Research Integrity

- Results are never fabricated. Missing results are marked `[RESULT NOT YET AVAILABLE]`.
- The test set is never contaminated.
- All random seeds are deterministic and documented.
- The Model Collapse Index (MCI) is clearly labeled as a project-specific metric.
- If collapse is not observed, that is reported honestly.

---

## License

This is a research project. The Alpaca dataset is CC BY-NC 4.0.
GPT-2 is MIT licensed.
