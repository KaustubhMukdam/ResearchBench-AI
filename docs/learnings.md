# Learnings — ResearchBench AI

Update at the end of every session (5 minutes max). Write in your own words — no copy-pasting from AI tools.

---

## Template

### [Date] — [Topic or session focus]

**What I learned:**
[Concept]: [Your explanation in plain English]

**Code snippet that clicked:**
```python
# paste the specific snippet that made something clear
```

**What confused me today:**
[Write the confusion precisely]

**How I solved it:**
[Tool used, what fixed it, why it works]

**What I'd do differently:**
[Hindsight — approach, structure, or tool choice]

---

## Concepts to understand before building (work through these in order)

### LangGraph fundamentals
- [ ] What is a `StateGraph`? What is "state" in an agent context?
- [ ] What is a "node" in LangGraph? (it's just a Python function)
- [ ] What is an "edge"? What is a "conditional edge"?
- [ ] How does `state` get passed between nodes?
- [ ] What is `checkpointing` and why does it matter for retries?

### RAG (Retrieval-Augmented Generation)
- [ ] What problem does RAG solve? (LLM doesn't know your documents)
- [ ] What is an "embedding"? What does it mean for two things to be "semantically similar"?
- [ ] What is a vector database? (ChromaDB stores embeddings + retrieves by similarity)
- [ ] What is "context window"? Why does it matter for RAG?
- [ ] The RAG pipeline: chunk → embed → store → retrieve → inject into prompt

### Ragas
- [ ] What is "Faithfulness"? (are all claims grounded in the retrieved context?)
- [ ] What is "Factual Correctness"? (are claims accurate vs. ground truth?)
- [ ] What is "Context Recall"? (did retrieval find the right documents?)
- [ ] How does Ragas actually measure these? (it makes LLM calls internally)

### FastAPI
- [ ] What is an async endpoint and why does it matter here? (agents run async)
- [ ] What is a Pydantic model in FastAPI? (request/response schema + validation)
- [ ] What is the OpenAPI spec? (auto-generated docs at /docs)

### SHAP (for DS Module)
- [ ] What is a SHAP value? (each feature's contribution to a specific prediction)
- [ ] What is the difference between global and local SHAP explanations?
- [ ] What is a SHAP summary plot?
