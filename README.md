# ResearchBench AI

A multi-agent AI platform that:
1. **Research Module** — surveys arXiv/Semantic Scholar papers, extracts benchmark data, and verifies citations via RAG
2. **DS Module** — runs automated EDA, feature engineering, multi-model training, leakage detection, and SHAP explainability on any CSV

---

## Phase 0 Setup (Start Here)

### 1. Clone and create environment

```bash
git clone <your-repo>
cd researchbench_ai
python -m venv venv
venv\Scripts\activate   # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Open .env and fill in your keys:
```

| Key | Where to get it |
|-----|----------------|
| `LANGSMITH_API_KEY` | https://smith.langchain.com → Settings → API Keys |
| `GROQ_API_KEY` | https://console.groq.com → API Keys |
| `TAVILY_API_KEY` | https://tavily.com → Dashboard |
| `SEMANTIC_SCHOLAR_API_KEY` | https://www.semanticscholar.org/product/api (apply, free) |

### 3. Verify LangSmith tracing

```bash
python verify_langsmith.py
```

Expected:
```
LangSmith tracing is active.
   Run ID   : <uuid>
   Project  : researchbench-ai
   Dashboard: https://smith.langchain.com/projects/researchbench-ai
```

### 4. Run the baseline Ragas evaluation

```bash
# Quick test (5 queries, fast):
python eval/run_ragas.py --limit 5

# Full baseline (30 queries, ~15-20 min):
python eval/run_ragas.py --mode baseline
```

This establishes your **baseline Faithfulness score** before any Citation Verifier Agent exists.

### 5. After building the Citation Verifier Agent (Phase 1), run again:

```bash
python eval/run_ragas.py --mode with_verifier
```

### 6. Compare runs to get your headline metric:

```bash
python eval/compare_runs.py
```

Output:
```
📌 Resume metric:
   "Cross-agent citation verification improved Faithfulness score from
    0.61 → 0.84 on 30 arXiv test queries (+23.0 points), measured via Ragas."
```

---

## Project Structure

```
researchbench_ai/
├── app/utils/config.py          # All env vars loaded here
├── eval/
│   ├── test_queries.json        # Fixed 30-query benchmark set
│   ├── run_ragas.py             # Baseline + with_verifier eval
│   ├── compare_runs.py          # Compares runs, prints resume metric
│   └── results_log.jsonl        # Auto-generated eval history (gitignored)
├── verify_langsmith.py          # Smoke test for LangSmith setup
├── requirements.txt
├── .env.example
└── docs/                        # Full project documentation
```

---

## Stack

| Layer | Tool |
|-------|------|
| Orchestration | LangGraph |
| LLM | Groq (Llama 3 70B) |
| Search | Tavily + arXiv API + Semantic Scholar |
| Vector Memory | ChromaDB |
| Evaluation | Ragas |
| Observability | LangSmith |
| Backend | FastAPI |
| Frontend | Streamlit |
