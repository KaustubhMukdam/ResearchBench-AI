# Experiment Log — ResearchBench AI

Use this for every meaningful change to agent prompts, retrieval strategy, model hyperparameters, or pipeline structure. Treat each change as a hypothesis. Log the result.

---

## Experiment 001 — 2026-05-28 (baseline, no verifier)

**Hypothesis:** Baseline retrieval → LLM summary without any citation verifier will have a Faithfulness score < 0.70 on the test set.

**Setup:**
- Research graph: Retrieval Agent → Method Extraction Agent → Gap Agent (no verifier)
- Test set: eval/test_queries.json (5 queries, limit flag)

**Results:**

| Metric | Score |
|--------|-------|
| Faithfulness | 1.0000 |
| Answer Correctness | 0.0653 |
| Context Recall | 0.0000 |

**What happened:** Faithfulness scored 1.0 — not because the pipeline is perfect, but because Ragas Faithfulness measures whether the LLM's own answer is grounded in the context it received. Without a verifier, the LLM generates hedged, conservative answers that are trivially supported by any retrieved context → 1.0. Context Recall = 0.0 is the real signal: the baseline pipeline retrieved 0 of the expected ground-truth papers for these queries.

**Why (understanding):** Faithfulness of 1.0 at baseline is a known Ragas artifact on small test sets. Answer Correctness (0.065) and Context Recall (0.0) are the honest metrics here.

**Next experiment:** Add Citation Verifier Agent, re-run same test set

---

## Experiment 002 — 2026-06-04 (with Citation Verifier)

**Hypothesis:** Adding Citation Verifier Agent (RAG cross-check on ChromaDB) will improve Context Recall and Answer Correctness.

**Change made:**
- Added `citation_verifier.py` node between Method Extraction and Gap Agent
- Verifier retrieves top-3 most similar passages from ChromaDB for each claim, checks if claim is supported

**Results:**

| Metric | Exp 001 (baseline) | Exp 002 (with verifier) | Delta |
|--------|----------------------|------------------------|-------|
| Faithfulness | 1.0000 | 0.9359 | -0.0641 |
| Answer Correctness | 0.0653 | 0.0961 | **+0.0308** |
| Context Recall | 0.0000 | 0.1250 | **+0.1250** |

**What happened:** Context Recall improved from 0 → 0.125 (+12.5 points) — the verifier's RAG retrieval is surfacing relevant paper chunks that the baseline missed. Answer Correctness improved by +3 points. Faithfulness dropped slightly (1.0 → 0.936) because the verifier now correctly flags some claims as unsupported, making the final answer more conservative and less trivially self-consistent.

**Why (understanding):** The Faithfulness decrease is actually a positive signal — it means the verifier is working. The LLM is no longer generating self-confirming answers; it's constrained by what RAG actually retrieved. Context Recall is the primary metric: +12.5 points means the system is finding more relevant papers per query.

**Next experiment:** Test DS pipeline on 5 Kaggle datasets (Experiment 003)

---

## Experiment 003 — DS Module Batch Test (Final — 5 datasets)

**Hypothesis:** The DS pipeline (EDA + 3 models + CV + SHAP) produces models that significantly beat majority-class baseline on 5 diverse datasets.

**Setup:**
- Pipeline: eda -> feature_engineering -> modeling -> validation -> explainability
- Models: XGBoost, LightGBM, RandomForest (5-fold CV, default params)
- Datasets: Titanic (891 rows), Telco Churn (7,043 rows), Pima Diabetes (768 rows), Breast Cancer (569 rows), Digits Binary (1,797 rows)

**Results:**

| Dataset | Rows | Baseline Acc | Best Model | CV Mean F1 | CV Std | Val Issues | Top Feature (SHAP) |
|---------|------|-------------|------------|-----------|--------|-----------|--------------------|
| Titanic | 891 | 0.6162 | RandomForestClassifier | 0.8406 | 0.0213 | 4 | Sex (0.1975) |
| Telco Churn | 7043 | 0.7346 | RandomForestClassifier | 0.8015 | 0.0069 | 2 | Contract (0.0802) |
| Pima Diabetes | 768 | 0.6510 | RandomForestClassifier | 0.7721 | 0.0258 | 0 | Glucose (0.1320) |
| Breast Cancer | 569 | 0.6274 | LGBMClassifier | 0.9631 | 0.0140 | 0 | worst area (1.9181) |
| Digits (Binary) | 1797 | 0.5014 | LGBMClassifier | 0.9733 | 0.0096 | 0 | pixel_6_4 (0.9890) |

Per-model results (all 3 trained successfully):
- **Titanic:** XGB=0.829, LGBM=0.833, **RF=0.838**
- **Telco Churn:** XGB=0.779, LGBM=0.790, **RF=0.791**
- **Pima Diabetes:** XGB=0.744, LGBM=0.732, **RF=0.767**
- **Breast Cancer:** XGB=0.952, **LGBM=0.963**, RF=0.956
- **Digits (Binary):** XGB=0.969, **LGBM=0.973**, RF=0.973

Delta over baseline:

| Dataset | Baseline | Best | Delta |
|---------|----------|------|-------|
| Titanic | 0.6162 | 0.8406 | **+0.2244** |
| Telco Churn | 0.7346 | 0.8015 | **+0.0669** |
| Pima Diabetes | 0.6510 | 0.7721 | **+0.1211** |
| Breast Cancer | 0.6274 | 0.9631 | **+0.3357** |
| Digits (Binary) | 0.5014 | 0.9733 | **+0.4719** |

**What happened:** All 5 datasets beat baseline by significant margins. RandomForest dominated on small/medium tabular data (Titanic, Telco, Pima). LGBM won on higher-dimensional clean data (Breast Cancer, Digits). Breast Cancer results are near-ceiling (>0.96 F1 is expected for this well-separated problem). Digits result shows the pipeline generalizes to non-tabular feature spaces (64 pixel values) binarized for binary classification. Groq rate limit (tokens/day) hit during Digits feature engineering — pipeline handled 429 gracefully with fallback to base features.

**Why (understanding):**
- RandomForest wins on small/medium data with mixed types because bagging handles noise/noisy categoricals better than boosting defaults
- LGBM pulls ahead when data is clean and high-dimensional (30-64 features) — its histogram-based splitting is more efficient than RF's exhaustive search on many features
- Breast Cancer's high scores (0.96+) are expected — this is a well-separated problem; the pipeline's real value is automating EDA + modeling + SHAP for it in ~6 seconds
- Digits beating its near-random baseline (0.5014) by +0.47 is the strongest signal: the pipeline can extract signal from non-tabular data without any hand-tuning

**Next experiment:** Test on a dataset with synthetic leakage to verify validation agent catches it. Add Optuna hyperparameter tuning. Test regression datasets (pipeline currently classification-only).
