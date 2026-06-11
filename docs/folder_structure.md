# Folder Structure — ResearchBench AI

```
researchbench_ai/
│
├── .env                        # API keys — NEVER commit this
├── .gitignore                  # Must include .env, __pycache__, .chromadb/
├── requirements.txt
├── README.md
│
├── app/                        # FastAPI backend
│   ├── main.py                 # FastAPI app entry point, route registration
│   ├── routers/
│   │   ├── research.py         # POST /research endpoint
│   │   └── dataset.py          # POST /analyze-dataset endpoint
│   ├── models/                 # Pydantic request/response schemas
│   │   ├── research_models.py  # PaperExtraction, ComparisonTable, GapReport
│   │   └── ds_models.py        # EDAReport, ModelResults, DSReport
│   ├── services/               # Business logic — separated from routes
│   │   ├── research_service.py # Calls research agent graph
│   │   └── ds_service.py       # Calls DS agent graph
│   └── utils/
│       ├── cache.py            # ChromaDB read/write helpers
│       └── config.py           # Load env vars, constants
│
├── agents/                     # LangGraph agent definitions
│   ├── research_graph.py       # StateGraph for research pipeline
│   ├── ds_graph.py             # StateGraph for DS pipeline
│   ├── nodes/                  # One file per agent node
│   │   ├── paper_retrieval.py  # arXiv + Semantic Scholar + Tavily
│   │   ├── method_extraction.py# Groq LLM + Pydantic output
│   │   ├── citation_verifier.py# ChromaDB RAG + claim verification
│   │   ├── comparison_gap.py   # Synthesis + benchmark table
│   │   ├── eda_agent.py        # pandas + ydata-profiling
│   │   ├── feature_agent.py    # LLM-based feature suggestions
│   │   ├── modeling_agent.py   # XGBoost, LightGBM, RF
│   │   ├── validation_agent.py # Leakage + overfit detection
│   │   └── explainability_agent.py  # SHAP values
│   └── tools/                  # LangChain Tool wrappers
│       ├── arxiv_tool.py
│       ├── semantic_scholar_tool.py
│       └── tavily_tool.py
│
├── memory/                     # ChromaDB persistent storage
│   └── .chromadb/              # Auto-created by ChromaDB (add to .gitignore)
│
├── frontend/
│   └── streamlit_app.py        # Single Streamlit app file
│
├── eval/                       # Ragas evaluation suite
│   ├── test_queries.json       # Fixed 30-query benchmark set with ground truth
│   └── run_ragas.py            # Script to run Ragas eval and log results
│
├── docs/                       # All project documentation
│   ├── project_context.md
│   ├── PRD.md
│   ├── tech_stack.md
│   ├── architecture.md
│   ├── folder_structure.md
│   ├── tasks.md
│   ├── learnings.md
│   ├── debug_log.md
│   ├── experiment_log.md       # ML experiment runs
│   └── eval.md                 # Ragas benchmark results over time
│
└── tests/
    ├── test_paper_retrieval.py
    ├── test_citation_verifier.py
    └── test_modeling_agent.py
```

## Naming conventions
- Python files: `snake_case` (`paper_retrieval.py`)
- Pydantic models: `PascalCase` classes (`PaperExtraction`, `EDAReport`)
- LangGraph nodes: `snake_case` function names (`retrieve_papers`, `verify_citations`)
- FastAPI routes: `snake_case` paths (`/analyze_dataset`)
- Constants: `UPPER_SNAKE_CASE` in `config.py`
- Test files: `test_` prefix matching the module they test

## Key files to understand first
1. `agents/research_graph.py` — the LangGraph StateGraph that wires all research agents together
2. `agents/nodes/citation_verifier.py` — the most important node; where hallucination reduction happens
3. `eval/run_ragas.py` — where your headline metric (Faithfulness score) is produced
4. `app/models/research_models.py` — Pydantic schemas; understanding these = understanding what each agent produces and expects
