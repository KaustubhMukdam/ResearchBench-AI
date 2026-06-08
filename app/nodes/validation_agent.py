"""
Validation Agent.
Checks for common ML pitfalls:
1. Data leakage (features too strongly correlated with target)
2. Overfitting (large gap between train and validation scores)
3. High cardinality features that might cause leakage
4. Missing value handling issues
"""

from __future__ import annotations
import logging

import pandas as pd
import numpy as np

from app.models.ds_models import DSState, ValidationReport, ValidationIssue

logger = logging.getLogger(__name__)

_LEAKAGE_CORR_THRESHOLD = 0.95
_OVERFITTING_THRESHOLD = 0.15


def _check_leakage(df: pd.DataFrame, target: str) -> list[ValidationIssue]:
    """Detect features that are suspiciously correlated with the target."""
    issues = []
    if target not in df.columns:
        return issues

    numeric_df = df.select_dtypes(include=[np.number])
    if target not in numeric_df.columns or numeric_df.shape[1] < 2:
        return issues

    corr = numeric_df.corr()[target].abs().drop(target)

    for feat, val in corr.items():
        if val >= _LEAKAGE_CORR_THRESHOLD:
            issues.append(
                ValidationIssue(
                    issue_type="leakage",
                    severity="high",
                    description=(
                        f"Feature '{feat}' has {val:.2f} correlation with target '{target}'. "
                        f"This may indicate data leakage."
                    ),
                )
            )
    return issues


def _check_missing_leakage(df: pd.DataFrame, target: str) -> list[ValidationIssue]:
    """Check if missing values in features could cause leakage."""
    issues = []
    for col in df.columns:
        if col == target:
            continue
        missing_ratio = df[col].isna().mean()
        if missing_ratio > 0.5:
            issues.append(
                ValidationIssue(
                    issue_type="missing_values",
                    severity="medium",
                    description=(
                        f"Column '{col}' has {missing_ratio:.1%} missing values. "
                        f"High missing rate may introduce bias."
                    ),
                )
            )
    return issues


def _check_high_cardinality(df: pd.DataFrame, target: str) -> list[ValidationIssue]:
    """Flag high-cardinality categorical features."""
    issues = []
    for col in df.columns:
        if col == target or not pd.api.types.is_object_dtype(df[col]):
            continue
        nunique = df[col].nunique()
        if nunique > 50:
            issues.append(
                ValidationIssue(
                    issue_type="high_cardinality",
                    severity="low",
                    description=(
                        f"Column '{col}' has {nunique} unique values. "
                        f"High cardinality may cause overfitting if not handled properly."
                    ),
                )
            )
    return issues


def run_validation(state: DSState) -> DSState:
    """
    LangGraph node: Validation
    Input:  state.file_path, state.target_column, state.model_results
    Output: state.validation_report
    """
    logger.info("[Validation] checking for data issues")

    try:
        df = pd.read_csv(state.file_path)
    except Exception as e:
        logger.error(f"[Validation] failed to read CSV: {e}")
        state.validation_report = ValidationReport(has_issues=True)
        return state

    all_issues = []
    all_issues.extend(_check_leakage(df, state.target_column))
    all_issues.extend(_check_missing_leakage(df, state.target_column))
    all_issues.extend(_check_high_cardinality(df, state.target_column))

    if state.model_results and state.model_results.cv_std > 0:
        cv_std = state.model_results.cv_std
        if cv_std > _OVERFITTING_THRESHOLD:
            all_issues.append(
                ValidationIssue(
                    issue_type="overfitting",
                    severity="medium",
                    description=(
                        f"High cross-validation variance (std={cv_std:.3f}). "
                        f"Model performance varies significantly across folds."
                    ),
                )
            )

    has_leakage = any(i.issue_type == "leakage" for i in all_issues)
    has_overfit = any(i.issue_type == "overfitting" for i in all_issues)

    # Check train vs CV gap if we have a rough estimate
    if state.model_results and state.model_results.results:
        best = next(
            (r for r in state.model_results.results if r.model_name == state.model_results.best_model_name),
            None,
        )
        if best:
            train_cv_gap = best.f1_score - state.model_results.cv_mean
            if train_cv_gap > 0.1 and state.model_results.cv_mean > 0:
                all_issues.append(
                    ValidationIssue(
                        issue_type="overfitting",
                        severity="medium",
                        description=(
                            f"Train-CV gap of {train_cv_gap:.3f} suggests overfitting "
                            f"(CV mean={state.model_results.cv_mean:.3f}, "
                            f"fold score={best.f1_score:.3f})."
                        ),
                    )
                )

    report = ValidationReport(
        has_issues=len(all_issues) > 0,
        leakage_detected=has_leakage,
        overfitting_detected=has_overfit,
        issues=all_issues,
    )

    state.validation_report = report
    logger.info(
        f"[Validation] {'issues found' if report.has_issues else 'all clear'} "
        f"- leakage={has_leakage}, overfitting={has_overfit}"
    )
    return state
