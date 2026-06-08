"""
Modeling Agent.
Trains XGBoost, LightGBM, and RandomForest with 5-fold cross-validation.
Handles preprocessing: missing value imputation, categorical encoding.
Returns structured ModelResults with the best model identified.
"""

from __future__ import annotations
import logging
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.models.ds_models import DSState, ModelResults, ModelResult

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


def _preprocess(df: pd.DataFrame, target: str, problem_type: str):
    """
    Preprocess DataFrame for modeling:
    - Separate X and y
    - Encode categorical columns
    - Impute missing values
    Returns (X, y, feature_names, label_encoder_for_target)
    """
    X = df.drop(columns=[target])
    y = df[target].copy()

    target_encoder = None
    if problem_type == "classification" and y.dtype in ("object", "category"):
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols_remaining = X.select_dtypes(exclude=[np.number]).columns
    X = X[num_cols.tolist() + cat_cols_remaining.tolist()]

    for col in X.columns:
        if X[col].isna().any():
            X[col] = SimpleImputer(strategy="most_frequent").fit_transform(X[[col]]).ravel()

    feature_names = X.columns.tolist()
    return X.values, y, feature_names, target_encoder


def _cv_score(model, X, y, problem_type: str, n_splits: int = 5):
    """Run cross-validation and return per-fold metrics."""
    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies, precisions, recalls, f1s, roc_aucs = [], [], [], [], []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        if problem_type == "classification" and len(np.unique(y)) == 2:
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_aucs.append(roc_auc_score(y_test, y_proba))
            except Exception:
                pass

    result = ModelResult(
        model_name=type(model).__name__,
        accuracy=round(float(np.mean(accuracies)), 4),
        precision=round(float(np.mean(precisions)), 4),
        recall=round(float(np.mean(recalls)), 4),
        f1_score=round(float(np.mean(f1s)), 4),
        roc_auc=round(float(np.mean(roc_aucs)), 4) if roc_aucs else None,
    )
    return result, accuracies


def run_modeling(state: DSState) -> DSState:
    """
    LangGraph node: Modeling
    Input:  state.file_path, state.target_column, state.problem_type
    Output: state.model_results
    """
    logger.info(f"[Modeling] starting with file='{state.file_path}', target='{state.target_column}'")

    try:
        df = pd.read_csv(state.file_path)
    except Exception as e:
        logger.error(f"[Modeling] failed to read CSV: {e}")
        state.error = f"Modeling failed: {e}"
        return state

    try:
        X, y, feature_names, target_encoder = _preprocess(df, state.target_column, state.problem_type)
    except Exception as e:
        logger.error(f"[Modeling] preprocessing failed: {e}")
        state.error = f"Preprocessing failed: {e}"
        return state

    logger.info(f"[Modeling] X shape: {X.shape}, features: {len(feature_names)}")

    models = []
    try:
        from xgboost import XGBClassifier, XGBRegressor
        if state.problem_type == "classification":
            models.append(XGBClassifier(n_estimators=100, max_depth=6, random_state=42, verbosity=0))
        else:
            models.append(XGBRegressor(n_estimators=100, max_depth=6, random_state=42, verbosity=0))
    except ImportError:
        logger.warning("[Modeling] XGBoost not available, skipping")

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        if state.problem_type == "classification":
            models.append(LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1))
        else:
            models.append(LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1))
    except ImportError:
        logger.warning("[Modeling] LightGBM not available, skipping")

    if state.problem_type == "classification":
        models.append(RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
    else:
        models.append(RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42))

    results_list = []
    all_cv_scores = []
    best_model_name = ""
    best_f1 = -1.0

    for model in models:
        try:
            result, cv_scores = _cv_score(model, X, y, state.problem_type)
            results_list.append(result)
            mean_f1 = result.f1_score
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_model_name = result.model_name
                all_cv_scores = cv_scores
            logger.info(
                f"[Modeling] {result.model_name}: "
                f"acc={result.accuracy:.3f}, f1={result.f1_score:.3f}"
            )
        except Exception as e:
            logger.warning(f"[Modeling] {type(model).__name__} failed: {e}")

    if not results_list:
        logger.error("[Modeling] no models trained successfully")
        state.error = "No models trained successfully"
        return state

    cv_mean = round(float(np.mean(all_cv_scores)), 4) if all_cv_scores else 0.0
    cv_std = round(float(np.std(all_cv_scores)), 4) if all_cv_scores else 0.0

    state.model_results = ModelResults(
        best_model_name=best_model_name,
        cv_scores=[round(float(s), 4) for s in all_cv_scores],
        cv_mean=cv_mean,
        cv_std=cv_std,
        results=results_list,
    )

    logger.info(
        f"[Modeling] best: {best_model_name} "
        f"(f1={best_f1:.3f}, cv_mean={cv_mean:.3f})"
    )
    return state
