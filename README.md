<div align="center">
  <img src="assets/banner.png" alt="Model Collapse Banner" width="100%">
  
  # 📉 Model Collapse: The Synthetic Data Decay
  
  *A complete, reproducible research system investigating the degradation of AI models trained on AI-generated data.*

  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
  ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
  ![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange?style=for-the-badge)
</div>

---

## 🔬 The Research Question
> **How does repeated training on model-generated synthetic data affect an AI's performance, diversity, and feature retention?** 

When an AI model generates text, and a *new* AI model is trained on that text, does the quality degrade? Can careful synthetic-data curation, entropy filtering, or curriculum mixing delay this "model collapse"? This project is built to answer exactly that.

---

## ⚡ Quick Start Guide

### 1️⃣ Environment Setup
Get your virtual environment ready and install the core dependencies.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `.\venv\Scripts\activate`

pip install -r requirements.txt
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 2️⃣ Run the Experiment Pipeline
We have packaged the entire generation, filtering, and training loop into a single command. 
```bash
python3 scripts/run_all_experiments.py
```
> **What this does:** It runs through 14 different experimental configurations, training multiple generations of models on the `databricks-dolly-15k` dataset using the `facebook/opt-125m` architecture, and evaluates them at every step!

### 3️⃣ View Results & Dashboard
Once the experiments conclude, you can generate a report or view the interactive dashboard.
```bash
# Generate a markdown report
python3 scripts/generate_report.py --output results/reports/report.md

# Launch the visual dashboard
streamlit run dashboard/app.py
```

---

## 🚀 Key Features & Customizations

This repository implements several unique, cutting-edge techniques to analyze and prevent model collapse:

- 🧠 **Modern Architecture**: Uses Meta's `facebook/opt-125m` for fast, state-of-the-art representational learning.
- 📚 **Dolly Dataset**: Replaces standard sets with `databricks-dolly-15k` to test collapse on high-quality instruction data.
- 🛡️ **Entropy Filtering**: A novel, mathematically-driven filter that calculates Shannon entropy to automatically detect and purge repetitive AI-hallucinations before they poison the next generation.
- 🔄 **Curriculum Mixing**: Dynamically alters the ratio of real-to-synthetic data across generations (e.g., 20% synthetic in Gen 1, 40% in Gen 2) to simulate the gradual AI-pollution of the internet.

---

## 📊 Evaluation Metrics

At every generation step, the model is evaluated on a completely untouched, pure-human test set. We track:

* **Perplexity & ROUGE**: Measures basic language modeling and response quality.
* **Lexical & Semantic Diversity**: Tracks vocabulary degradation and embedding variance.
* **Distribution Shift**: Calculates Jensen-Shannon divergence to see how far the AI drifts from human norms.
* **Model Collapse Index (MCI)**: A custom composite metric designed specifically for this project to quantify total degradation.

---

## 📁 Project Architecture

```text
model-collapse-study/
├── 🖼️ assets/                # README Graphics
├── ⚙️ configs/               # YAML Experiment Configurations
├── 💻 src/                   # Core Logic
│   ├── data/                 # Dataset splitting & curriculum mixing
│   ├── models/               # Model training (OPT-125m)
│   ├── generation/           # Synthetic Data Generator (Self-Instruct)
│   ├── filtering/            # Entropy, Deduplication, & Quality filters
│   └── evaluation/           # Deep metric calculations
├── 📜 scripts/               # CLI Entry Points
├── 📈 dashboard/             # Streamlit Visualizer
└── 📊 results/               # Experiment Outputs (Auto-generated)
```

---

<div align="center">
  <b>Built for Advanced Agentic Research & AI Safety</b> <br>
  <i>Do not pollute the training sets.</i>
</div>
