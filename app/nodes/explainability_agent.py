"""
Explainability Agent.
Computes SHAP values for the best-performing model to explain feature importance.
"""

from __future__ import annotations
import logging
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from app.models.ds_models import DSState, FeatureImportance, SHAPSummary

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


def _preprocess(df: pd.DataFrame, target: str):
    """Same preprocessing as modeling_agent for consistency."""
    X = df.drop(columns=[target])
    y = df[target].copy()

    if y.dtype in ("object", "category"):
        y = LabelEncoder().fit_transform(y)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    for col in X.columns:
        if X[col].isna().any():
            X[col] = SimpleImputer(strategy="most_frequent").fit_transform(X[[col]]).ravel()

    return X.values, y, X.columns.tolist()


def run_explainability(state: DSState) -> DSState:
    """
    LangGraph node: Explainability
    Input:  state.file_path, state.target_column, state.model_results.best_model_name
    Output: state.shap_summary
    """
    logger.info("[Explainability] computing SHAP values")

    try:
        df = pd.read_csv(state.file_path)
    except Exception as e:
        logger.error(f"[Explainability] failed to read CSV: {e}")
        state.shap_summary = SHAPSummary()
        return state

    try:
        X, y, feature_names = _preprocess(df, state.target_column)
    except Exception as e:
        logger.error(f"[Explainability] preprocessing failed: {e}")
        state.shap_summary = SHAPSummary()
        return state

    if not state.model_results or not state.model_results.best_model_name:
        logger.warning("[Explainability] no model results to explain")
        state.shap_summary = SHAPSummary()
        return state

    best_name = state.model_results.best_model_name
    logger.info(f"[Explainability] explaining model: {best_name}")

    # Train the best model on full data for SHAP
    try:
        import shap
    except ImportError:
        logger.error("[Explainability] 'shap' not installed, skipping")
        state.shap_summary = SHAPSummary()
        return state

    try:
        if "XGB" in best_name:
            from xgboost import XGBClassifier, XGBRegressor
            model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0)
        elif "LGBM" in best_name or "LightGBM" in best_name:
            from lightgbm import LGBMClassifier, LGBMRegressor
            model = LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
        else:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)

        if state.problem_type == "regression":
            logger.warning("[Explainability] regression SHAP not fully implemented, using TreeExplainer")

        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # binary classification, class 1

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]

        importances = []
        for rank, idx in enumerate(sorted_idx[:20]):
            importances.append(
                FeatureImportance(
                    feature=feature_names[idx],
                    importance=round(float(mean_abs_shap[idx]), 4),
                    rank=rank + 1,
                )
            )

        state.shap_summary = SHAPSummary(
            top_features=importances,
            best_model_name=best_name,
        )
        logger.info(f"[Explainability] top feature: {importances[0].feature} ({importances[0].importance:.4f})")

    except Exception as e:
        logger.error(f"[Explainability] SHAP computation failed: {e}")
        state.shap_summary = SHAPSummary()

    return state
