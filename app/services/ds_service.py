"""DS service — wraps the LangGraph DS pipeline for FastAPI."""
import logging
import tempfile
import os

import pandas as pd

from app.ds_graph import run_ds_pipeline
from app.schemas import DSResponse, FeatureSuggestionOut
from app.cost_tracker import TokenCounter

logger = logging.getLogger(__name__)


def _eda_to_dict(eda) -> dict:
    if not eda:
        return {}
    return {
        "row_count": eda.row_count,
        "column_count": eda.column_count,
        "target_column": eda.target_column,
        "target_class_balance": eda.target_class_balance,
        "has_missing_values": eda.has_missing_values,
        "has_high_cardinality": eda.has_high_cardinality,
        "columns": [c.model_dump() for c in eda.columns],
        "summary": eda.summary,
    }


def _model_results_to_dict(mr) -> dict:
    if not mr:
        return {}
    return {
        "best_model_name": mr.best_model_name,
        "cv_mean": mr.cv_mean,
        "cv_std": mr.cv_std,
        "results": [r.model_dump() for r in mr.results],
    }


def _validation_to_dict(vr) -> dict:
    if not vr:
        return {}
    return {
        "has_issues": vr.has_issues,
        "leakage_detected": vr.leakage_detected,
        "overfitting_detected": vr.overfitting_detected,
        "issues": [i.model_dump() for i in vr.issues],
    }


def _shap_to_dict(ss) -> dict:
    if not ss or not ss.top_features:
        return {}
    return {
        "best_model_name": ss.best_model_name,
        "top_features": [f.model_dump() for f in ss.top_features],
    }


def run_ds(csv_bytes: bytes, filename: str, target_column: str) -> DSResponse:
    suffix = os.path.splitext(filename)[1] or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(csv_bytes)
        tmp_path = f.name

    try:
        counter = TokenCounter()
        state = run_ds_pipeline(
            file_path=tmp_path,
            target_column=target_column,
            callbacks=[counter],
        )

        suggestions = []
        if state.feature_suggestions and state.feature_suggestions.suggestions:
            suggestions = [
                FeatureSuggestionOut(name=s.name, description=s.description)
                for s in state.feature_suggestions.suggestions
            ]

        return DSResponse(
            eda=_eda_to_dict(state.eda_report),
            feature_suggestions=suggestions,
            model_results=_model_results_to_dict(state.model_results),
            validation=_validation_to_dict(state.validation_report),
            shap=_shap_to_dict(state.shap_summary),
            error=state.error,
            token_usage=counter.summary(),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
