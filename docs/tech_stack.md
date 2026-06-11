# Tech Stack — ResearchBench AI

## Orchestration
| Technology | Version | Why chosen |
|------------|---------|------------|
| LangGraph | ≥0.2 | Graph-based agent workflows; supports state persistence, retries, conditional edges, and human-in-the-loop — more production-grade than CrewAI |
| Python | 3.11 | LangChain/LangGraph ecosystem requirement; typing improvements over 3.10 |

## LLM Inference
| Technology | Version | Why chosen |
|------------|---------|------------|
| Groq | API (latest) | Fastest inference available on free tier (~300 tokens/sec on Llama 3); essential for agentic loops that make many LLM calls |
| Llama 3 (70B) | via Groq | Open weights model; sufficient reasoning for extraction and verification tasks; no OpenAI cost |

## Search & Retrieval
| Technology | Why chosen |
|------------|------------|
| Tavily Search API | Optimised for LLM agents; returns clean, structured web results; has a free tier |
| arXiv API | Official, free, no rate limit for bulk access; structured metadata (title, abstract, authors, categories, date) |
| Semantic Scholar API | Academic graph — returns citations, references, paper influence scores; free with API key (1 RPS) |

## Vector Memory
| Technology | Why chosen |
|------------|------------|
| ChromaDB | Local, embedded, zero-infrastructure; perfect for MVP; persists embeddings to disk so paper cache survives restarts |
| sentence-transformers | Generates embeddings for paper abstracts; `all-MiniLM-L6-v2` is fast and sufficient for semantic similarity |

## Evaluation & Observability
| Technology | Why chosen |
|------------|------------|
| Ragas | Purpose-built RAG evaluation library; measures Faithfulness, Factual Correctness, Context Recall — exactly the metrics needed to quantify hallucination improvement |
| LangSmith | Official LangChain tracing; captures every agent step, token count, latency, cost; free tier is generous for solo projects |

## Data Science Agents
| Technology | Why chosen |
|------------|------------|
| pandas | Standard DataFrame manipulation for EDA agent |
| scikit-learn | Model training (RandomForest, LogisticRegression, baseline models) |
| XGBoost | Gradient boosting — standard Kaggle baseline |
| LightGBM | Faster than XGBoost on large tabular datasets |
| SHAP | Model explainability — generates feature importance values; required for Explainability Agent |
| ydata-profiling | Automated EDA report generation — speeds up the EDA Agent significantly |

## Schema Validation
| Technology | Why chosen |
|------------|------------|
| Pydantic v2 | Enforces structured output schemas on every agent's response; prevents malformed data from propagating between agents |

## Backend
| Technology | Version | Why chosen |
|------------|---------|------------|
| FastAPI | ≥0.110 | Async support; automatic OpenAPI docs; clean integration with LangGraph async workflows |
| uvicorn | latest | ASGI server for FastAPI |

## Frontend
| Technology | Why chosen |
|------------|------------|
| Streamlit | Fastest Python-native UI; no JavaScript required; sufficient for MVP; supports file upload, tables, charts natively |

## Alternatives considered
- **CrewAI rejected:** Simpler to use but lacks graph-based state management; not well-suited for complex multi-step pipelines with retries and conditional branching
- **OpenAI GPT-4 rejected:** Cost too high for an agentic system making 10–20 LLM calls per run
- **FAISS rejected:** Doesn't persist to disk natively; requires more setup than ChromaDB for a solo project
- **AutoML libraries (AutoGluon/H2O) rejected:** They're black-box solutions; this project's value is in the agent reasoning chain, not just model selection

## Known tradeoffs
- Groq free tier has rate limits (~30 requests/min) — agentic loops must include retry logic and backoff
- Ragas evaluation metrics use LLM calls themselves — keep evaluation to sampled test sets, not every live run
- Semantic Scholar free API key = 1 RPS dedicated — build a caching layer (ChromaDB) to avoid re-hitting the API
- Streamlit is not production-grade for multi-user scenarios — acceptable for MVP and portfolio demo
