"""
Feature Engineering Agent.
Takes the EDA report and column metadata, uses Groq to suggest
feature transformations, encodings, and derived features.
"""

from __future__ import annotations
import json
import logging

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from app.models.ds_models import DSState, FeatureSuggestion, FeatureSuggestions
from app.utils.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

_llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)

_FEATURE_PROMPT = """\
You are a data science feature engineering expert. Based on the EDA report below,
suggest 3-5 new features that could improve model performance.

EDA Summary:
{eda_summary}

Columns:
{columns_text}

Target column: {target}
Problem type: {problem_type}

Return a JSON array with EXACTLY this structure:
[
  {{
    "name": "feature_name",
    "description": "one sentence explanation of why this feature helps",
    "code": "pandas code to create this feature (use df as the DataFrame variable)"
  }}
]

Rules:
- Only suggest features that use existing columns (no external data)
- Code must be valid pandas operations
- Do NOT suggest target leakage features
- Return valid JSON only. No markdown fences, no explanation.
"""


def _columns_text(eda_report) -> str:
    """Format column info for the LLM prompt."""
    lines = []
    for col in eda_report.columns:
        lines.append(
            f"  - {col.name} ({col.dtype}): {col.unique_count} unique, "
            f"{col.missing_pct}% missing"
        )
    return "\n".join(lines)


def run_feature_engineering(state: DSState) -> DSState:
    """
    LangGraph node: Feature Engineering
    Input:  state.eda_report
    Output: state.feature_suggestions
    """
    eda = state.eda_report
    if not eda:
        logger.warning("[Feature] no EDA report available")
        state.feature_suggestions = FeatureSuggestions()
        return state

    logger.info("[Feature] generating feature suggestions via Groq")

    prompt = _FEATURE_PROMPT.format(
        eda_summary=eda.summary,
        columns_text=_columns_text(eda),
        target=eda.target_column,
        problem_type=state.problem_type,
    )

    try:
        response = _llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)
        if isinstance(data, list):
            suggestions = [FeatureSuggestion(**s) for s in data]
        else:
            suggestions = [FeatureSuggestion(**s) for s in data.get("suggestions", data)]

        state.feature_suggestions = FeatureSuggestions(suggestions=suggestions)
        logger.info(f"[Feature] generated {len(suggestions)} feature suggestions")

    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"[Feature] LLM suggestion failed: {e}")
        state.feature_suggestions = FeatureSuggestions()

    return state
