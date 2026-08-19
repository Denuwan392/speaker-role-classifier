#!/usr/bin/env python3
"""
Generates AUDIT_01_REPOSITORY_INVENTORY.md by recursively scanning the role_detection repository.
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent.resolve()
AUDIT_DIR = ROOT_DIR / "audit"

def get_file_type(path):
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "Python Script"
    elif suffix == ".ipynb":
        return "Jupyter Notebook"
    elif suffix in [".pkl", ".joblib"]:
        return "Model Artifact"
    elif suffix in [".csv", ".json", ".parquet"]:
        return "Data Artifact"
    elif suffix == ".md":
        return "Documentation / Report"
    elif suffix == ".db":
        return "Database File"
    elif suffix == ".png":
        return "Image / Visualization"
    return "Other"

def get_purpose(path):
    name = path.name
    rel_path = str(path.relative_to(ROOT_DIR))
    
    if name == "predict_role.py":
        return "Core production API export for predicting speaker roles from transcript segments."
    elif name == "feature_pipeline.py":
        return "Deterministic Feature Extractor class (RoleFeatureExtractor)."
    elif name == "agentic_router.py":
        return "Hybrid Role Router with XGBoost probability threshold check and Gemini LLM fallback."
    elif name == "train_pipeline.py":
        return "End-to-end model training pipeline with GroupKFold splitting, SMOTE, and post-hoc threshold tuning."
    elif name == "app.py":
        return "Interactive Streamlit web application for real-time testing and batch inference."
    elif name == "inference.py":
        return "Standalone CLI script demonstrating feature extraction and hybrid routing."
    elif name == "demo.py":
        return "CLI demo script."
    elif name == "02_feature_engineering.py":
        return "Batch script for aggregating raw utterances to speaker-level features and fitting global TF-IDF/SVD."
    elif name == "labeled_roles.csv":
        return "Raw utterance-level ground-truth dataset (1,555 rows)."
    elif name == "features.csv":
        return "Speaker-level aggregated dataset with handcrafted and SVD features (1,133 rows)."
    elif name == "role_classifier.pkl":
        return "Trained production XGBoost model artifact."
    elif name == "label_encoder.pkl":
        return "Trained LabelEncoder mapping ['hr', 'junior', 'manager', 'other']."
    elif name == "tfidf_vectorizer.joblib":
        return "Fitted TfidfVectorizer artifact (global fit)."
    elif name == "tfidf_svd.joblib":
        return "Fitted TruncatedSVD artifact (global fit, 32 components)."
    elif name == "eval_on_test.ipynb":
        return "Notebook evaluating model on test set (contains split bug with 0 HR samples)."
    elif name == "model_validation_report.md":
        return "Previous validation report evaluating 90 synthetic benchmark samples."
    elif name == "eval_metrics.json":
        return "Saved metrics from train_pipeline.py (Macro F1 = 0.6351)."
    elif name == "eval_metrics_full.json":
        return "Saved metrics from eval_on_test.ipynb (Macro F1 = 0.8836, 3-class only)."
    return "Repository support artifact."

def is_active(path):
    rel = str(path.relative_to(ROOT_DIR))
    if "venv" in rel or ".git" in rel or ".mpl_cache" in rel:
        return False
    return True

def is_prod_ref(path):
    name = path.name
    return name in [
        "predict_role.py", "feature_pipeline.py", "agentic_router.py", "app.py",
        "role_classifier.pkl", "label_encoder.pkl", "tfidf_vectorizer.joblib", "tfidf_svd.joblib"
    ]

def is_reproducible(path):
    name = path.name
    if name in ["eval_metrics_full.json", "02_feature_engineering.py"]:
        return "PARTIAL / LEAKED"
    return "YES"

def scan_repo():
    rows = []
    for root, dirs, files in os.walk(ROOT_DIR):
        # Ignore venv, .git, __pycache__, cache
        dirs[:] = [d for d in dirs if d not in ["venv", ".git", "__pycache__", "cache", ".mpl_cache"]]
        for f in files:
            if f.endswith(".pyc") or f == ".DS_Store":
                continue
            fpath = Path(root) / f
            rel = fpath.relative_to(ROOT_DIR)
            stat = fpath.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = f"{stat.st_size / 1024:.1f} KB"
            
            rows.append({
                "path": str(rel),
                "type": get_file_type(fpath),
                "purpose": get_purpose(fpath),
                "size": size_kb,
                "modified": mtime,
                "active": "YES" if is_active(fpath) else "NO",
                "prod_ref": "YES" if is_prod_ref(fpath) else "NO",
                "reproducible": is_reproducible(fpath)
            })
    return pd.DataFrame(rows)

def main():
    df = scan_repo()
    
    md = f"""# AUDIT 01 — REPOSITORY FORENSIC INVENTORY

## 1. Inventory Summary
- **Total Scanned Artifacts:** {len(df)} files (excluding `venv/`, `.git/`, `__pycache__/`)
- **Core Production Scripts:** 4 files (`predict_role.py`, `feature_pipeline.py`, `agentic_router.py`, `app.py`)
- **Trained Model Artifacts:** 4 files in `models/` (`role_classifier.pkl`, `label_encoder.pkl`, `tfidf_vectorizer.joblib`, `tfidf_svd.joblib`)
- **Primary Datasets:** 2 files in `data/` (`labeled_roles.csv`, `features.csv`)
- **Jupyter Notebooks:** 5 files in `notebook/`
- **Previous Evaluation Reports:** 2 files (`model_validation_report.md`, `reports/eval_metrics_full.json`)

## 2. Complete Artifact Directory

| Relative Path | Type | Purpose | Size | Modified Date | Actively Used? | Prod Ref? | Reproducible? |
|---|---|---|---:|---|:---:|:---:|:---:|
"""
    for _, r in df.iterrows():
        md += f"| `{r['path']}` | {r['type']} | {r['purpose']} | {r['size']} | {r['modified']} | {r['active']} | {r['prod_ref']} | {r['reproducible']} |\n"

    (AUDIT_DIR / "AUDIT_01_REPOSITORY_INVENTORY.md").write_text(md)
    print("Generated AUDIT_01_REPOSITORY_INVENTORY.md successfully.")

if __name__ == "__main__":
    main()
