"""
EDA Agent.
Reads a CSV file, runs exploratory data analysis, and returns a structured EDAReport.
Detects missing values, data types, correlations, class balance, and high-cardinality features.
"""

from __future__ import annotations
import logging
import pandas as pd
import numpy as np

from app.models.ds_models import DSState, EDAReport, ColumnInfo

logger = logging.getLogger(__name__)


def _detect_problem_type(df: pd.DataFrame, target: str) -> str:
    """Detect if problem is classification or regression based on target dtype and unique values."""
    if target not in df.columns:
        return "classification"
    s = df[target]
    if s.dtype in ("object", "category", "bool"):
        return "classification"
    if pd.api.types.is_integer_dtype(s):
        unique_ratio = s.nunique() / len(s)
        return "classification" if unique_ratio < 0.05 else "regression"
    return "regression"


def _find_high_correlations(
    df: pd.DataFrame, threshold: float = 0.8
) -> list[tuple[str, str, float]]:
    """Find pairs of numeric columns with correlation above threshold."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return []
    corr_matrix = numeric_df.corr().abs()
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if val >= threshold and not np.isnan(val):
                pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], round(val, 3)))
    return pairs


def _build_summary(report: EDAReport, df: pd.DataFrame, target: str) -> str:
    """Generate a plain-text summary of the EDA findings."""
    lines = [
        f"Dataset: {report.row_count} rows x {report.column_count} columns",
        f"Target column: {target}",
        f"Problem type: classification",
        f"Numeric columns: {len(report.numeric_columns)}",
        f"Categorical columns: {len(report.categorical_columns)}",
    ]
    if report.has_missing_values:
        missing_cols = [c for c in report.columns if c.missing_count > 0]
        lines.append(f"Missing values found in {len(missing_cols)} columns")
    if report.high_correlation_pairs:
        lines.append(
            f"High correlation pairs (>0.8): {len(report.high_correlation_pairs)}"
        )
    if report.target_class_balance:
        total = sum(report.target_class_balance.values())
        if total > 0:
            max_class = max(report.target_class_balance.values())
            ratio = max_class / total
            lines.append(f"Target class balance: largest class {ratio:.1%} of data")
            if ratio > 0.8:
                lines.append("WARNING: Significant class imbalance detected")
    return "\n".join(lines)


def run_eda(state: DSState) -> DSState:
    """
    LangGraph node: EDA
    Input:  state.file_path, state.target_column
    Output: state.eda_report
    """
    file_path = state.file_path
    target = state.target_column
    logger.info(f"[EDA] reading '{file_path}', target='{target}'")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"[EDA] failed to read CSV: {e}")
        state.error = f"Failed to read CSV: {e}"
        return state

    logger.info(f"[EDA] loaded {len(df)} rows, {len(df.columns)} columns")

    columns = []
    numeric_cols = []
    categorical_cols = []
    for col in df.columns:
        info = ColumnInfo(
            name=col,
            dtype=str(df[col].dtype),
            missing_count=int(df[col].isna().sum()),
            missing_pct=round(float(df[col].isna().mean() * 100), 2),
            unique_count=int(df[col].nunique()),
        )
        columns.append(info)
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    problem_type = _detect_problem_type(df, target)

    class_balance: dict[str, int] = {}
    if target in df.columns and problem_type == "classification":
        raw = df[target].value_counts().to_dict()
        class_balance = {str(k): int(v) for k, v in raw.items()}

    corr_pairs = _find_high_correlations(df)
    has_missing = any(c.missing_count > 0 for c in columns)
    has_high_card = any(c.unique_count > 100 for c in columns if c.name != target)

    eda = EDAReport(
        row_count=len(df),
        column_count=len(df.columns),
        columns=columns,
        target_column=target,
        target_class_balance=class_balance,
        high_correlation_pairs=corr_pairs,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        has_missing_values=has_missing,
        has_high_cardinality=has_high_card,
        summary="",
    )
    eda.summary = _build_summary(eda, df, target)

    state.eda_report = eda
    state.problem_type = problem_type
    logger.info(f"[EDA] complete - {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    return state
