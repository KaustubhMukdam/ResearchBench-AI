"""ResearchBench AI — Streamlit frontend."""
import io
import os

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="ResearchBench AI", layout="wide")
st.title("ResearchBench AI")


def _call_api(endpoint: str, **kwargs):
    url = f"{API_BASE}{endpoint}"
    try:
        resp = requests.post(url, **kwargs, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"Cannot connect to backend at {API_BASE}. Is the server running?")
        return None
    except requests.Timeout:
        st.error("Request timed out after 5 minutes.")
        return None
    except requests.HTTPError as e:
        st.error(f"Backend error: {e}")
        return None


# ── Research Tab ────────────────────────────────────────────────────────

tab_research, tab_ds = st.tabs(["Research", "Data Science"])

with tab_research:
    st.header("Research Pipeline")
    topic = st.text_input("Research topic", placeholder="e.g. transformers for tabular data")
    if st.button("Run Research", type="primary") and topic.strip():
        with st.spinner("Running research pipeline..."):
            data = _call_api("/research", json={"topic": topic.strip()})
        if data is None:
            st.stop()

        gap = data.get("gap_summary", "")
        papers = data.get("papers", [])
        extractions = data.get("extractions", [])
        claims = data.get("verified_claims", [])
        benchmark = data.get("benchmark_table", [])

        if gap:
            with st.expander("Gap Analysis", expanded=True):
                st.markdown(gap)

        if papers:
            with st.expander(f"Papers ({len(papers)})", expanded=False):
                paper_df = pd.DataFrame(
                    [
                        {
                            "Title": p["title"],
                            "Year": p.get("year", ""),
                            "Citations": p.get("citation_count", 0),
                            "Source": p.get("source", ""),
                        }
                        for p in papers
                    ]
                )
                st.dataframe(paper_df, use_container_width=True)

                for p in papers:
                    ext = next((e for e in extractions if e.get("arxiv_id") == p.get("arxiv_id")), None)
                    if ext and ext.get("methods"):
                        with st.expander(f"Methods: {p['title']}"):
                            for m in ext["methods"]:
                                st.markdown(f"- **{m['name']}**: {m['description']}")
                                if m.get("dataset"):
                                    st.markdown(f"  - Dataset: {m['dataset']}")
                                if m.get("metric") and m.get("score"):
                                    st.markdown(f"  - {m['metric']}: {m['score']}")
        else:
            st.info("No papers found for this topic.")

        if benchmark:
            with st.expander("Benchmark Table", expanded=False):
                st.dataframe(pd.DataFrame(benchmark), use_container_width=True)

        if claims:
            with st.expander(f"Verified Claims ({len(claims)})", expanded=False):
                for c in claims:
                    label = "[SUPPORTED]" if c.get("supported") else "[NOT SUPPORTED]"
                    st.markdown(f"{label} {c['claim']} (confidence: {c.get('confidence', 0):.2f})")

    elif not topic.strip():
        st.warning("Enter a topic first.")

# ── DS Tab ──────────────────────────────────────────────────────────────

with tab_ds:
    st.header("Data Science Pipeline")
    uploaded = st.file_uploader("Upload CSV", type="csv")

    df_preview = None
    if uploaded is not None:
        df_preview = pd.read_csv(uploaded)
        uploaded.seek(0)
        st.subheader("Data Preview")
        st.dataframe(df_preview.head(5), use_container_width=True)
        st.caption(f"{len(df_preview)} rows, {len(df_preview.columns)} columns")

    cols = list(df_preview.columns) if df_preview is not None else []
    target = st.selectbox("Target column", cols, disabled=not cols)

    if st.button("Run Analysis", type="primary", disabled=uploaded is None or not target):
        uploaded.seek(0)
        files = {"file": (uploaded.name, uploaded, "text/csv")}
        with st.spinner("Running DS pipeline..."):
            data = _call_api("/analyze-dataset", data={"target_column": target}, files=files)
        if data is None:
            st.stop()

        eda = data.get("eda")
        suggestions = data.get("feature_suggestions", [])
        model_results = data.get("model_results")
        validation = data.get("validation")
        shap_data = data.get("shap")

        if eda:
            with st.expander("EDA Summary", expanded=True):
                c = eda
                st.markdown(
                    f"**{c['row_count']}** rows x **{c['column_count']}** columns  |  "
                    f"Target: `{c['target_column']}`"
                )
                if c.get("target_class_balance"):
                    bal = c["target_class_balance"]
                    st.markdown("**Class balance:**")
                    for k, v in bal.items():
                        st.markdown(f"- {k}: {v}")
                if c.get("has_missing_values"):
                    st.warning("Missing values detected")
                if c.get("has_high_cardinality"):
                    st.warning("High-cardinality features detected")

        if suggestions:
            with st.expander(f"Feature Suggestions ({len(suggestions)})", expanded=True):
                for s in suggestions:
                    st.markdown(f"- **{s['name']}**: {s['description']}")

        if model_results:
            with st.expander("Model Results", expanded=True):
                mr = model_results
                st.markdown(f"**Best model:** {mr['best_model_name']}")
                st.markdown(f"**CV mean F1:** {mr['cv_mean']:.4f}  |  **CV std:** {mr['cv_std']:.4f}")
                if mr.get("results"):
                    res_df = pd.DataFrame(mr["results"])
                    st.dataframe(res_df, use_container_width=True)

        if validation:
            with st.expander("Validation Report", expanded=True):
                v = validation
                if v.get("has_issues"):
                    st.error(f"{len(v['issues'])} issue(s) detected")
                    for issue in v["issues"]:
                        sev = issue.get("severity", "low")
                        st.markdown(f"**[{sev.upper()}]** {issue['description']}")
                else:
                    st.success("All clear — no issues detected")

        if shap_data and shap_data.get("top_features"):
            with st.expander("Feature Importance (SHAP)", expanded=True):
                tf = shap_data["top_features"]
                names = [f["feature"] for f in tf][:10]
                vals = [f["importance"] for f in tf][:10]
                fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
                ax.barh(range(len(names)), vals, color="steelblue")
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names)
                ax.invert_yaxis()
                ax.set_xlabel("Importance")
                st.pyplot(fig)
        elif shap_data is not None and not shap_data.get("top_features"):
            st.info("No feature importance data available.")
