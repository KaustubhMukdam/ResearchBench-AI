"""Phase 2 smoke test — run from project root."""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from app.ds_graph import run_ds_pipeline

# ── Prepare test dataset ────────────────────────────────────────────────
data = load_breast_cancer(as_frame=True)
df = data.frame
df["target"] = data.target  # already present in as_frame, but ensure name
csv_path = "/tmp/breast_cancer.csv"
df.to_csv(csv_path, index=False)
print(f"Test dataset: {csv_path} ({len(df)} rows, {len(df.columns)} columns)")
print(f"Target distribution:\n{df['target'].value_counts().to_dict()}")

# ── Run pipeline ────────────────────────────────────────────────────────
result = run_ds_pipeline(file_path=csv_path, target_column="target")

# ── Results ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"EDA         : {result.eda_report.row_count} rows, {result.eda_report.column_count} cols")
print(f"Problem type: {result.problem_type}")
if result.eda_report.target_class_balance:
    print(f"Class dist  : {result.eda_report.target_class_balance}")
print(f"High corr   : {len(result.eda_report.high_correlation_pairs)} pairs")

if result.feature_suggestions and result.feature_suggestions.suggestions:
    for s in result.feature_suggestions.suggestions:
        print(f"Feature idea: {s.name} — {s.description}")
else:
    print("Feature ideas: (none generated)")

if result.model_results:
    print(f"\nBest model   : {result.model_results.best_model_name}")
    print(f"CV scores    : {[round(s, 3) for s in result.model_results.cv_scores]}")
    print(f"CV mean      : {result.model_results.cv_mean:.4f}")
    print(f"CV std       : {result.model_results.cv_std:.4f}")
    for m in result.model_results.results:
        print(f"  {m.model_name}: acc={m.accuracy:.3f}, f1={m.f1_score:.3f}, roc_auc={m.roc_auc}")

if result.validation_report:
    print(f"\nValidation issues: {len(result.validation_report.issues)}")
    for issue in result.validation_report.issues:
        print(f"  [{issue.severity}] {issue.issue_type}: {issue.description}")

if result.shap_summary and result.shap_summary.top_features:
    print(f"\nTop 5 features ({result.shap_summary.best_model_name}):")
    for fi in result.shap_summary.top_features[:5]:
        print(f"  #{fi.rank}: {fi.feature} ({fi.importance:.4f})")

print(f"\nError: {result.error}")
print('='*60)
