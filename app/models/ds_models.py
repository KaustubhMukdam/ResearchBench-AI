"""
Pydantic schemas for the Data Science Module (Phase 2).
Single source of truth for all DS pipeline data shapes.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0


class EDAReport(BaseModel):
    row_count: int = 0
    column_count: int = 0
    columns: list[ColumnInfo] = Field(default_factory=list)
    target_column: str = ""
    target_class_balance: dict[str, int] = Field(default_factory=dict)
    high_correlation_pairs: list[tuple[str, str, float]] = Field(default_factory=list)
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    has_missing_values: bool = False
    has_high_cardinality: bool = False
    summary: str = ""


class FeatureSuggestion(BaseModel):
    name: str
    description: str
    code: str = ""


class FeatureSuggestions(BaseModel):
    suggestions: list[FeatureSuggestion] = Field(default_factory=list)


class ModelResult(BaseModel):
    model_name: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: Optional[float] = None


class ModelResults(BaseModel):
    best_model_name: str = ""
    cv_scores: list[float] = Field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    results: list[ModelResult] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    issue_type: str = ""
    severity: str = "low"
    description: str = ""


class ValidationReport(BaseModel):
    has_issues: bool = False
    leakage_detected: bool = False
    overfitting_detected: bool = False
    issues: list[ValidationIssue] = Field(default_factory=list)


class FeatureImportance(BaseModel):
    feature: str = ""
    importance: float = 0.0
    rank: int = 0


class SHAPSummary(BaseModel):
    top_features: list[FeatureImportance] = Field(default_factory=list)
    best_model_name: str = ""


class DSState(BaseModel):
    file_path: str = ""
    target_column: str = ""
    problem_type: str = "classification"
    eda_report: Optional[EDAReport] = None
    feature_suggestions: Optional[FeatureSuggestions] = None
    model_results: Optional[ModelResults] = None
    validation_report: Optional[ValidationReport] = None
    shap_summary: Optional[SHAPSummary] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
