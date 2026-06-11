"""
Data Science LangGraph StateGraph.
Wires all Phase 2 nodes into a linear pipeline:
  eda -> feature_engineering -> modeling -> validation -> explainability
"""

from __future__ import annotations
import logging
from typing import Optional

from langgraph.graph import StateGraph, END
from langchain_core.callbacks import BaseCallbackHandler

from app.models.ds_models import DSState
from app.nodes.eda_agent import run_eda
from app.nodes.feature_agent import run_feature_engineering
from app.nodes.modeling_agent import run_modeling
from app.nodes.validation_agent import run_validation
from app.nodes.explainability_agent import run_explainability

logger = logging.getLogger(__name__)


def build_ds_graph() -> StateGraph:
    """
    Build and compile the data science pipeline graph.
    All nodes take DSState and return DSState.
    """
    graph = StateGraph(DSState)

    graph.add_node("eda", run_eda)
    graph.add_node("feature_engineering", run_feature_engineering)
    graph.add_node("modeling", run_modeling)
    graph.add_node("validation", run_validation)
    graph.add_node("explainability", run_explainability)

    graph.set_entry_point("eda")
    graph.add_edge("eda", "feature_engineering")
    graph.add_edge("feature_engineering", "modeling")
    graph.add_edge("modeling", "validation")
    graph.add_edge("validation", "explainability")
    graph.add_edge("explainability", END)

    return graph.compile()


def run_ds_pipeline(
    file_path: str,
    target_column: str,
    callbacks: Optional[list[BaseCallbackHandler]] = None,
) -> DSState:
    """
    Run the full data science pipeline for a CSV dataset.
    Returns the final DSState with all reports populated.
    """
    logger.info(f"Starting DS pipeline: file='{file_path}', target='{target_column}'")
    graph = build_ds_graph()

    initial_state = DSState(file_path=file_path, target_column=target_column)
    config = {}
    if callbacks:
        config["callbacks"] = callbacks
    final_state = graph.invoke(initial_state, config)

    eda = final_state.get("eda_report")
    results = final_state.get("model_results")
    shap = final_state.get("shap_summary")

    logger.info(
        f"Pipeline complete - "
        f"EDA: {eda.row_count if eda else 0} rows, "
        f"Best model: {results.best_model_name if results else 'N/A'}, "
        f"Top feature: {shap.top_features[0].feature if (shap and shap.top_features) else 'N/A'}"
    )
    return DSState(**final_state)
