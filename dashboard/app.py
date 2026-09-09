"""Streamlit dashboard for Model Collapse Study.

Launch with: streamlit run dashboard/app.py
"""
import os
import sys
import json
import glob

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(layout="wide", page_title="Model Collapse Dashboard")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")


def load_experiment(exp_dir: str) -> dict:
    """Load experiment results from JSON."""
    path = os.path.join(exp_dir, "results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def get_metric_df(results: dict, metric_path: str) -> pd.DataFrame:
    """Extract a metric series as a DataFrame."""
    rows = []
    for gen_str, data in sorted(results.get("generations", {}).items(), key=lambda x: int(x[0])):
        metrics = data.get("metrics", {})
        val = metrics
        for part in metric_path.split("."):
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                val = None
                break
        if val is not None and isinstance(val, (int, float)):
            rows.append({"Generation": int(gen_str), "Value": float(val)})
    return pd.DataFrame(rows)


def main():
    st.title("🔬 Model Collapse Study Dashboard")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Overview", "Performance", "Diversity", "Collapse",
        "Synthetic Data", "Experiment Comparison", "Model Collapse Index"
    ])

    if not os.path.exists(RESULTS_DIR):
        st.warning("No results directory found. Please run experiments first.")
        st.code("python scripts/run_experiment.py --config configs/experiments/exp_b_synthetic_100.yaml")
        return

    exp_dirs = sorted([
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    ])

    if not exp_dirs:
        st.warning("No experiments found. Run an experiment first.")
        return

    # ─── Overview ───
    if page == "Overview":
        st.header("📊 Experiment Overview")

        selected = st.sidebar.selectbox("Select Experiment", exp_dirs)
        results = load_experiment(os.path.join(RESULTS_DIR, selected))

        if not results:
            st.info("No results for this experiment.")
            return

        config = results.get("config", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", config.get("model", {}).get("name", "N/A"))
        col2.metric("Generations", config.get("experiment", {}).get("generations", "N/A"))
        col3.metric("Synthetic Ratio", f"{config.get('data', {}).get('synthetic_ratio', 'N/A')}")
        col4.metric("Temperature", config.get("generation", {}).get("temperature", "N/A"))

        # Results table
        csv_path = os.path.join(RESULTS_DIR, selected, "results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.dataframe(df, use_container_width=True)

        # Raw config
        with st.expander("Full Configuration"):
            st.json(config)

    # ─── Performance ───
    elif page == "Performance":
        st.header("📈 Performance Metrics")
        selected = st.sidebar.selectbox("Select Experiment", exp_dirs)
        results = load_experiment(os.path.join(RESULTS_DIR, selected))

        col1, col2 = st.columns(2)

        with col1:
            df = get_metric_df(results, "perplexity")
            if not df.empty:
                fig = px.line(df, x="Generation", y="Value", title="Perplexity vs Generation",
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Perplexity data not available.")

        with col2:
            df = get_metric_df(results, "rouge.rougeL")
            if not df.empty:
                fig = px.line(df, x="Generation", y="Value", title="ROUGE-L vs Generation",
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ROUGE data not available.")

    # ─── Diversity ───
    elif page == "Diversity":
        st.header("🌈 Diversity Metrics")
        selected = st.sidebar.selectbox("Select Experiment", exp_dirs)
        results = load_experiment(os.path.join(RESULTS_DIR, selected))

        col1, col2 = st.columns(2)

        with col1:
            df = get_metric_df(results, "lexical_diversity.type_token_ratio")
            if not df.empty:
                fig = px.line(df, x="Generation", y="Value", title="Type-Token Ratio vs Generation",
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)

            df = get_metric_df(results, "lexical_diversity.vocabulary_size")
            if not df.empty:
                fig = px.line(df, x="Generation", y="Value", title="Vocabulary Size vs Generation",
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            df = get_metric_df(results, "semantic_diversity.pairwise_cosine_similarity")
            if not df.empty:
                fig = px.line(df, x="Generation", y="Value",
                              title="Pairwise Cosine Similarity vs Generation", markers=True)
                st.plotly_chart(fig, use_container_width=True)

            df = get_metric_df(results, "semantic_diversity.embedding_variance")
            if not df.empty:
                fig = px.line(df, x="Generation", y="Value",
                              title="Embedding Variance vs Generation", markers=True)
                st.plotly_chart(fig, use_container_width=True)

    # ─── Collapse ───
    elif page == "Collapse":
        st.header("💥 Collapse Indicators")
        selected = st.sidebar.selectbox("Select Experiment", exp_dirs)
        results = load_experiment(os.path.join(RESULTS_DIR, selected))

        col1, col2 = st.columns(2)

        with col1:
            df = get_metric_df(results, "duplication.exact_duplicate_rate")
            if not df.empty:
                fig = px.line(df, x="Generation", y="Value", title="Duplicate Rate vs Generation",
                              markers=True)
                st.plotly_chart(fig, use_container_width=True)

            df = get_metric_df(results, "js_divergence")
            if not df.empty:
                fig = px.line(df, x="Generation", y="Value",
                              title="JS Divergence vs Generation", markers=True)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            df = get_metric_df(results, "rare_features.rare")
            if not df.empty:
                fig = px.line(df, x="Generation", y="Value",
                              title="Rare Feature Retention vs Generation", markers=True)
                st.plotly_chart(fig, use_container_width=True)

            df = get_metric_df(results, "memorization.mean_ngram_overlap")
            if not df.empty:
                fig = px.line(df, x="Generation", y="Value",
                              title="Memorization (N-gram Overlap) vs Generation", markers=True)
                st.plotly_chart(fig, use_container_width=True)

    # ─── Synthetic Data ───
    elif page == "Synthetic Data":
        st.header("🏭 Synthetic Data Statistics")
        selected = st.sidebar.selectbox("Select Experiment", exp_dirs)
        results = load_experiment(os.path.join(RESULTS_DIR, selected))

        for gen_str, data in sorted(results.get("generations", {}).items(), key=lambda x: int(x[0])):
            gen_stats = data.get("generation_stats", {})
            filter_stats = data.get("filtering_stats", {})
            if gen_stats or filter_stats:
                with st.expander(f"Generation {gen_str}"):
                    if gen_stats:
                        st.json(gen_stats)
                    if filter_stats:
                        st.subheader("Filtering Statistics")
                        st.json(filter_stats)

    # ─── Experiment Comparison ───
    elif page == "Experiment Comparison":
        st.header("🔄 Experiment Comparison")

        selected_exps = st.sidebar.multiselect("Select Experiments", exp_dirs, default=exp_dirs[:3])

        if not selected_exps:
            st.info("Select at least one experiment from the sidebar.")
            return

        metric = st.selectbox("Metric", [
            "perplexity", "rouge.rougeL",
            "lexical_diversity.type_token_ratio",
            "semantic_diversity.pairwise_cosine_similarity",
            "duplication.exact_duplicate_rate",
            "js_divergence", "rare_features.rare",
            "model_collapse_index.mci",
        ])

        fig = go.Figure()
        for exp_name in selected_exps:
            results = load_experiment(os.path.join(RESULTS_DIR, exp_name))
            df = get_metric_df(results, metric)
            if not df.empty:
                fig.add_trace(go.Scatter(
                    x=df["Generation"], y=df["Value"],
                    mode="lines+markers", name=exp_name
                ))

        fig.update_layout(
            title=f"{metric} — Experiment Comparison",
            xaxis_title="Generation",
            yaxis_title=metric,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ─── Model Collapse Index ───
    elif page == "Model Collapse Index":
        st.header("📏 Model Collapse Index (MCI)")

        st.warning(
            "⚠️ **MCI is a project-specific composite metric.** "
            "It is NOT a universally accepted scientific measure. "
            "Always examine individual metrics alongside MCI."
        )

        selected = st.sidebar.selectbox("Select Experiment", exp_dirs)
        results = load_experiment(os.path.join(RESULTS_DIR, selected))

        df = get_metric_df(results, "model_collapse_index.mci")
        if not df.empty:
            fig = px.line(df, x="Generation", y="Value",
                          title="Model Collapse Index vs Generation", markers=True)
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # Show component breakdown
            for gen_str, data in sorted(results.get("generations", {}).items(), key=lambda x: int(x[0])):
                mci_data = data.get("metrics", {}).get("model_collapse_index", {})
                if mci_data.get("components"):
                    with st.expander(f"Generation {gen_str} — MCI Components"):
                        components = mci_data["components"]
                        comp_df = pd.DataFrame([
                            {"Component": k, "Value": v} for k, v in components.items()
                        ])
                        st.bar_chart(comp_df.set_index("Component"))
        else:
            st.info("MCI data not available. Run experiments first.")


if __name__ == "__main__":
    main()
