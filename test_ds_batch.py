"""Batch test DS pipeline on 3 Kaggle datasets + log to experiment_log.md."""
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

import pandas as pd
import numpy as np

from app.ds_graph import run_ds_pipeline

DATASETS = [
    {"name": "Titanic", "path": "data/titanic.csv", "target": "Survived"},
    {"name": "Telco Churn", "path": "data/telco_churn.csv", "target": "Churn"},
    {"name": "Pima Diabetes", "path": "data/pima_diabetes.csv", "target": "Outcome"},
]


def majority_baseline_accuracy(df: pd.DataFrame, target: str) -> float:
    """Most common class proportion = baseline accuracy."""
    counts = df[target].value_counts()
    return round(float(counts.max() / counts.sum()), 4)


def run_all():
    results = []
    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds['name']}")
        print(f"{'='*60}")

        df = pd.read_csv(ds["path"])
        baseline = majority_baseline_accuracy(df, ds["target"])
        print(f"Baseline (majority class): {baseline:.4f}")

        result = run_ds_pipeline(file_path=ds["path"], target_column=ds["target"])

        best_name = result.model_results.best_model_name if result.model_results else "NONE"
        cv_mean = result.model_results.cv_mean if result.model_results else 0.0
        n_issues = len(result.validation_report.issues) if result.validation_report else 0
        top_feat = ""
        if result.shap_summary and result.shap_summary.top_features:
            tf = result.shap_summary.top_features[0]
            top_feat = f"{tf.feature} ({tf.importance:.4f})"

        print(f"Best model  : {best_name}")
        print(f"CV mean F1  : {cv_mean:.4f}")
        print(f"CV std      : {result.model_results.cv_std if result.model_results else 0:.4f}")
        print(f"Validation  : {n_issues} issues")
        print(f"Top feature : {top_feat}")
        print(f"Error       : {result.error}")

        results.append({
            "name": ds["name"],
            "rows": len(df),
            "cols": len(df.columns),
            "baseline_acc": baseline,
            "best_model": best_name,
            "cv_mean_f1": round(cv_mean, 4),
            "cv_std": round(result.model_results.cv_std, 4) if result.model_results else 0.0,
            "validation_issues": n_issues,
            "top_feature": top_feat,
            "error": result.error,
        })

    return results


def append_to_log(results):
    """Append Phase 2 results to docs/experiment_log.md."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"\n---\n",
        f"\n## Experiment 003 — DS Module Batch Test ({now})\n",
        f"\n**Hypothesis:** The DS pipeline (EDA + 3 models + SHAP) produces models that significantly outperform\n",
        f"majority-class baseline on 3 diverse Kaggle datasets.\n",
        f"\n**Setup:**\n",
        f"- Pipeline: eda -> feature_engineering -> modeling -> validation -> explainability\n",
        f"- Models: XGBoost, LightGBM, RandomForest (5-fold CV, default params)\n",
        f"- Datasets: Titanic (891 rows), Telco Churn (7,043 rows), Pima Diabetes (768 rows)\n",
        f"\n**Results:**\n",
        f"\n| Dataset | Rows | Baseline Acc | Best Model | CV Mean F1 | CV Std | Validation Issues | Top Feature |\n",
        f"|---------|------|-------------|------------|-----------|--------|-----------------|-------------|\n",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['rows']} | {r['baseline_acc']:.4f} | "
            f"{r['best_model']} | {r['cv_mean_f1']:.4f} | {r['cv_std']:.4f} | "
            f"{r['validation_issues']} | {r['top_feature']} |\n"
        )

    lines.append("\n**What happened:** [fill in after reviewing]\n")
    lines.append("**Why (understanding):** [fill in]\n")
    lines.append("**Next experiment:** [fill in]\n")

    with open("docs/experiment_log.md", "a") as f:
        f.writelines(lines)

    print(f"\n{'='*60}")
    print("Results appended to docs/experiment_log.md")
    print(f"{'='*60}")


if __name__ == "__main__":
    results = run_all()
    append_to_log(results)
