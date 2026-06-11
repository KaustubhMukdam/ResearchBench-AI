# project_context.md

## Project
Name: ResearchBench AI
Status: in-progress
Started: May 2026

## One-liner
A multi-agent AI platform that surveys ML research papers, extracts benchmarks with citation verification, and runs automated EDA + model evaluation on structured datasets — bridging academic research and practical implementation.

## Stack
- Frontend: Streamlit (Python)
- Backend: FastAPI + Python 3.11
- Orchestration: LangGraph
- LLM Inference: Groq (Llama 3)
- Search: Tavily (web) + arXiv API + Semantic Scholar API
- Vector DB: ChromaDB
- RAG Evaluation: Ragas
- Observability: LangSmith
- Schema Validation: Pydantic v2

## Key decisions made
- Used LangGraph over CrewAI for graph-based workflows, retries, state persistence, and production suitability
- Used Groq for fast, cheap LLM inference — ideal for iterative agentic loops
- Used ChromaDB for local vector memory so the system doesn't re-fetch papers already indexed
- Used Ragas to quantify hallucination rate before/after Citation Verifier Agent — this is the headline metric
- Kept Streamlit as frontend — focus on working system over UI polish

## Current focus
Setting up LangSmith observability + Ragas evaluation harness BEFORE writing any agent code

## Known issues / blockers
- Semantic Scholar API: default unauthenticated = 1 RPS (shared). Apply for a free API key early.
- Ragas LLM-based metrics make extra LLM calls — run only on sampled test sets (20–30 queries), not every live run

## What this project is NOT doing
- Not a general-purpose chatbot
- Not a real-time paper alert system
- Not a full literature management tool (like Zotero/Mendeley)
- No user authentication or multi-user support in MVP
- No UI upgrade beyond Streamlit in MVP
