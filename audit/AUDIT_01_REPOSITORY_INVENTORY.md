# AUDIT 01 — REPOSITORY FORENSIC INVENTORY

## 1. Inventory Summary
- **Total Scanned Artifacts:** 315 files (excluding `venv/`, `.git/`, `__pycache__/`)
- **Core Production Scripts:** 4 files (`predict_role.py`, `feature_pipeline.py`, `agentic_router.py`, `app.py`)
- **Trained Model Artifacts:** 4 files in `models/` (`role_classifier.pkl`, `label_encoder.pkl`, `tfidf_vectorizer.joblib`, `tfidf_svd.joblib`)
- **Primary Datasets:** 2 files in `data/` (`labeled_roles.csv`, `features.csv`)
- **Jupyter Notebooks:** 5 files in `notebook/`
- **Previous Evaluation Reports:** 2 files (`model_validation_report.md`, `reports/eval_metrics_full.json`)

## 2. Complete Artifact Directory

| Relative Path | Type | Purpose | Size | Modified Date | Actively Used? | Prod Ref? | Reproducible? |
|---|---|---|---:|---|:---:|:---:|:---:|
| `train_pipeline.py` | Python Script | End-to-end model training pipeline with GroupKFold splitting, SMOTE, and post-hoc threshold tuning. | 14.3 KB | 2026-07-03 16:54 | YES | NO | YES |
| `predict_role.py` | Python Script | Core production API export for predicting speaker roles from transcript segments. | 6.8 KB | 2026-07-03 16:54 | YES | YES | YES |
| `02_feature_engineering.py` | Python Script | Batch script for aggregating raw utterances to speaker-level features and fitting global TF-IDF/SVD. | 7.3 KB | 2026-07-03 16:54 | YES | NO | PARTIAL / LEAKED |
| `requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2025-12-11 05:37 | YES | NO | YES |
| `model_validation_report.md` | Documentation / Report | Previous validation report evaluating 90 synthetic benchmark samples. | 19.2 KB | 2026-07-03 16:54 | YES | NO | YES |
| `agentic_router.py` | Python Script | Hybrid Role Router with XGBoost probability threshold check and Gemini LLM fallback. | 4.3 KB | 2026-07-03 16:54 | YES | YES | YES |
| `README.md` | Documentation / Report | Repository support artifact. | 3.3 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlflow.db` | Database File | Repository support artifact. | 1468.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `.gitignore` | Other | Repository support artifact. | 0.4 KB | 2026-07-03 17:00 | NO | NO | YES |
| `.env` | Other | Repository support artifact. | 0.1 KB | 2026-06-06 17:27 | YES | NO | YES |
| `app.py` | Python Script | Interactive Streamlit web application for real-time testing and batch inference. | 16.4 KB | 2026-07-03 16:54 | YES | YES | YES |
| `inference.py` | Python Script | Standalone CLI script demonstrating feature extraction and hybrid routing. | 2.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `demo.py` | Python Script | CLI demo script. | 0.6 KB | 2025-12-11 08:22 | YES | NO | YES |
| `feature_pipeline.py` | Python Script | Deterministic Feature Extractor class (RoleFeatureExtractor). | 4.7 KB | 2026-07-03 16:54 | YES | YES | YES |
| `json_to_csv/json_to_csv.py` | Python Script | Repository support artifact. | 1.4 KB | 2025-12-11 03:59 | YES | NO | YES |
| `json_to_csv/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2025-12-11 04:15 | YES | NO | YES |
| `json_to_csv/data/meetings.json` | Data Artifact | Repository support artifact. | 81.4 KB | 2025-12-11 04:37 | YES | NO | YES |
| `json_to_csv/data/labeled_roles.csv` | Data Artifact | Raw utterance-level ground-truth dataset (1,555 rows). | 46.1 KB | 2025-12-11 04:34 | YES | NO | YES |
| `models/label_encoder.pkl` | Model Artifact | Trained LabelEncoder mapping ['hr', 'junior', 'manager', 'other']. | 0.5 KB | 2026-07-03 16:54 | YES | YES | YES |
| `models/tfidf_svd.joblib` | Model Artifact | Fitted TruncatedSVD artifact (global fit, 32 components). | 376.5 KB | 2026-06-07 11:35 | YES | YES | YES |
| `models/role_classifier.pkl` | Model Artifact | Trained production XGBoost model artifact. | 1114.2 KB | 2026-07-03 16:54 | YES | YES | YES |
| `models/tfidf_vectorizer.joblib` | Model Artifact | Fitted TfidfVectorizer artifact (global fit). | 57.4 KB | 2026-06-07 11:35 | YES | YES | YES |
| `mlruns/0/meta.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/1/models/m-dd4f4b63cae84cfca18f66abd77ea459/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-06-06 18:00 | YES | NO | YES |
| `mlruns/1/models/m-dd4f4b63cae84cfca18f66abd77ea459/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-06-06 18:00 | YES | NO | YES |
| `mlruns/1/models/m-dd4f4b63cae84cfca18f66abd77ea459/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-06-06 18:00 | YES | NO | YES |
| `mlruns/1/models/m-dd4f4b63cae84cfca18f66abd77ea459/artifacts/model.ubj` | Other | Repository support artifact. | 710.1 KB | 2026-06-06 18:00 | YES | NO | YES |
| `mlruns/1/models/m-dd4f4b63cae84cfca18f66abd77ea459/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-06-06 18:00 | YES | NO | YES |
| `mlruns/1/models/m-19c1223caf3f4ec0bccbca340ba982f9/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 10:35 | YES | NO | YES |
| `mlruns/1/models/m-19c1223caf3f4ec0bccbca340ba982f9/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 10:35 | YES | NO | YES |
| `mlruns/1/models/m-19c1223caf3f4ec0bccbca340ba982f9/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-06-07 10:35 | YES | NO | YES |
| `mlruns/1/models/m-19c1223caf3f4ec0bccbca340ba982f9/artifacts/model.ubj` | Other | Repository support artifact. | 668.5 KB | 2026-06-07 10:35 | YES | NO | YES |
| `mlruns/1/models/m-19c1223caf3f4ec0bccbca340ba982f9/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-06-07 10:35 | YES | NO | YES |
| `mlruns/1/models/m-ecfd32f86b8f44f588689c3903752990/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 08:37 | YES | NO | YES |
| `mlruns/1/models/m-ecfd32f86b8f44f588689c3903752990/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 08:37 | YES | NO | YES |
| `mlruns/1/models/m-ecfd32f86b8f44f588689c3903752990/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-06-07 08:37 | YES | NO | YES |
| `mlruns/1/models/m-ecfd32f86b8f44f588689c3903752990/artifacts/model.ubj` | Other | Repository support artifact. | 922.0 KB | 2026-06-07 08:37 | YES | NO | YES |
| `mlruns/1/models/m-ecfd32f86b8f44f588689c3903752990/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-06-07 08:37 | YES | NO | YES |
| `mlruns/1/models/m-4acd058fc60f48b29be733a4a404193f/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-06-06 23:55 | YES | NO | YES |
| `mlruns/1/models/m-4acd058fc60f48b29be733a4a404193f/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-06-06 23:55 | YES | NO | YES |
| `mlruns/1/models/m-4acd058fc60f48b29be733a4a404193f/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-06-06 23:55 | YES | NO | YES |
| `mlruns/1/models/m-4acd058fc60f48b29be733a4a404193f/artifacts/model.ubj` | Other | Repository support artifact. | 710.1 KB | 2026-06-06 23:55 | YES | NO | YES |
| `mlruns/1/models/m-4acd058fc60f48b29be733a4a404193f/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-06-06 23:55 | YES | NO | YES |
| `mlruns/1/models/m-70cb98c17e314e20a74bc4715e65fd38/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 10:48 | YES | NO | YES |
| `mlruns/1/models/m-70cb98c17e314e20a74bc4715e65fd38/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 10:48 | YES | NO | YES |
| `mlruns/1/models/m-70cb98c17e314e20a74bc4715e65fd38/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-06-07 10:48 | YES | NO | YES |
| `mlruns/1/models/m-70cb98c17e314e20a74bc4715e65fd38/artifacts/model.ubj` | Other | Repository support artifact. | 668.5 KB | 2026-06-07 10:48 | YES | NO | YES |
| `mlruns/1/models/m-70cb98c17e314e20a74bc4715e65fd38/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-06-07 10:48 | YES | NO | YES |
| `mlruns/1/models/m-81fc8ad3ff344a869df6315c25dc2b03/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 11:32 | YES | NO | YES |
| `mlruns/1/models/m-81fc8ad3ff344a869df6315c25dc2b03/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 11:32 | YES | NO | YES |
| `mlruns/1/models/m-81fc8ad3ff344a869df6315c25dc2b03/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-07-03 11:32 | YES | NO | YES |
| `mlruns/1/models/m-81fc8ad3ff344a869df6315c25dc2b03/artifacts/model.ubj` | Other | Repository support artifact. | 1109.6 KB | 2026-07-03 11:32 | YES | NO | YES |
| `mlruns/1/models/m-81fc8ad3ff344a869df6315c25dc2b03/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-07-03 11:32 | YES | NO | YES |
| `mlruns/1/models/m-d4f792b5aab748e0a440a86f5cd49aff/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 07:13 | YES | NO | YES |
| `mlruns/1/models/m-d4f792b5aab748e0a440a86f5cd49aff/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 07:13 | YES | NO | YES |
| `mlruns/1/models/m-d4f792b5aab748e0a440a86f5cd49aff/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-06-07 07:13 | YES | NO | YES |
| `mlruns/1/models/m-d4f792b5aab748e0a440a86f5cd49aff/artifacts/model.ubj` | Other | Repository support artifact. | 393.9 KB | 2026-06-07 07:13 | YES | NO | YES |
| `mlruns/1/models/m-d4f792b5aab748e0a440a86f5cd49aff/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-06-07 07:13 | YES | NO | YES |
| `mlruns/1/models/m-b31e3e7664124e62988453840118184d/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 08:40 | YES | NO | YES |
| `mlruns/1/models/m-b31e3e7664124e62988453840118184d/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 08:40 | YES | NO | YES |
| `mlruns/1/models/m-b31e3e7664124e62988453840118184d/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-06-07 08:40 | YES | NO | YES |
| `mlruns/1/models/m-b31e3e7664124e62988453840118184d/artifacts/model.ubj` | Other | Repository support artifact. | 922.0 KB | 2026-06-07 08:40 | YES | NO | YES |
| `mlruns/1/models/m-b31e3e7664124e62988453840118184d/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-06-07 08:40 | YES | NO | YES |
| `mlruns/1/models/m-14c14f9fd89846869deac222ac4bbb34/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 11:12 | YES | NO | YES |
| `mlruns/1/models/m-14c14f9fd89846869deac222ac4bbb34/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 11:12 | YES | NO | YES |
| `mlruns/1/models/m-14c14f9fd89846869deac222ac4bbb34/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-06-07 11:12 | YES | NO | YES |
| `mlruns/1/models/m-14c14f9fd89846869deac222ac4bbb34/artifacts/model.ubj` | Other | Repository support artifact. | 690.4 KB | 2026-06-07 11:12 | YES | NO | YES |
| `mlruns/1/models/m-14c14f9fd89846869deac222ac4bbb34/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-06-07 11:12 | YES | NO | YES |
| `mlruns/1/models/m-f90133567585439881745435a8c545ca/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 08:24 | YES | NO | YES |
| `mlruns/1/models/m-f90133567585439881745435a8c545ca/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 08:24 | YES | NO | YES |
| `mlruns/1/models/m-f90133567585439881745435a8c545ca/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-06-07 08:24 | YES | NO | YES |
| `mlruns/1/models/m-f90133567585439881745435a8c545ca/artifacts/model.ubj` | Other | Repository support artifact. | 370.2 KB | 2026-06-07 08:24 | YES | NO | YES |
| `mlruns/1/models/m-f90133567585439881745435a8c545ca/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-06-07 08:24 | YES | NO | YES |
| `mlruns/1/models/m-6a1c49deba2f4fcbbbf4532780e5c671/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 11:35 | YES | NO | YES |
| `mlruns/1/models/m-6a1c49deba2f4fcbbbf4532780e5c671/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-06-07 11:35 | YES | NO | YES |
| `mlruns/1/models/m-6a1c49deba2f4fcbbbf4532780e5c671/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-06-07 11:35 | YES | NO | YES |
| `mlruns/1/models/m-6a1c49deba2f4fcbbbf4532780e5c671/artifacts/model.ubj` | Other | Repository support artifact. | 1109.6 KB | 2026-06-07 11:35 | YES | NO | YES |
| `mlruns/1/models/m-6a1c49deba2f4fcbbbf4532780e5c671/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-06-07 11:35 | YES | NO | YES |
| `mlruns/1/models/m-416e2967fb294eb6809b100fcc853c4e/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 11:36 | YES | NO | YES |
| `mlruns/1/models/m-416e2967fb294eb6809b100fcc853c4e/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 11:36 | YES | NO | YES |
| `mlruns/1/models/m-416e2967fb294eb6809b100fcc853c4e/artifacts/MLmodel` | Other | Repository support artifact. | 0.7 KB | 2026-07-03 11:36 | YES | NO | YES |
| `mlruns/1/models/m-416e2967fb294eb6809b100fcc853c4e/artifacts/model.ubj` | Other | Repository support artifact. | 1109.6 KB | 2026-07-03 11:36 | YES | NO | YES |
| `mlruns/1/models/m-416e2967fb294eb6809b100fcc853c4e/artifacts/conda.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-07-03 11:36 | YES | NO | YES |
| `mlruns/603949218934757692/meta.yaml` | Other | Repository support artifact. | 0.3 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/meta.yaml` | Other | Repository support artifact. | 0.5 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/metrics/junior_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/metrics/other_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/metrics/macro_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/metrics/manager_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/artifacts/artifacts/label_encoder.pkl` | Model Artifact | Trained LabelEncoder mapping ['hr', 'junior', 'manager', 'other']. | 0.5 KB | 2026-07-03 16:54 | YES | YES | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/artifacts/artifacts/tfidf_svd.joblib` | Model Artifact | Fitted TruncatedSVD artifact (global fit, 32 components). | 215.5 KB | 2025-12-11 06:57 | YES | YES | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/artifacts/artifacts/tfidf_vectorizer.joblib` | Model Artifact | Fitted TfidfVectorizer artifact (global fit). | 32.8 KB | 2025-12-11 06:57 | YES | YES | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/artifacts/artifacts/eval_metrics.json` | Data Artifact | Saved metrics from train_pipeline.py (Macro F1 = 0.6351). | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/tags/mlflow.user` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/tags/mlflow.runName` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/tags/mlflow.source.name` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/tags/mlflow.source.type` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/max_depth` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/learning_rate` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/n_features` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/calibrated` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/n_estimators` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/feature_engineering_version` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/val_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/model` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/train_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/class_weight_strategy` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/test_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/params/subsample` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/8978c21fc0d24c229f3072a97c05e8f8/outputs/m-5940d153693247f585da70246fa520a2/meta.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/meta.yaml` | Other | Repository support artifact. | 0.5 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/metrics/junior_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/metrics/other_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/metrics/macro_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/metrics/manager_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/artifacts/artifacts/label_encoder.pkl` | Model Artifact | Trained LabelEncoder mapping ['hr', 'junior', 'manager', 'other']. | 0.5 KB | 2026-07-03 16:54 | YES | YES | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/artifacts/artifacts/tfidf_svd.joblib` | Model Artifact | Fitted TruncatedSVD artifact (global fit, 32 components). | 215.5 KB | 2025-12-11 06:57 | YES | YES | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/artifacts/artifacts/tfidf_vectorizer.joblib` | Model Artifact | Fitted TfidfVectorizer artifact (global fit). | 32.8 KB | 2025-12-11 06:57 | YES | YES | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/artifacts/artifacts/eval_metrics.json` | Data Artifact | Saved metrics from train_pipeline.py (Macro F1 = 0.6351). | 0.2 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/tags/mlflow.user` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/tags/mlflow.runName` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/tags/mlflow.source.name` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/tags/mlflow.source.type` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/max_depth` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/learning_rate` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/n_features` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/calibrated` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/n_estimators` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/feature_engineering_version` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/val_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/model` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/train_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/class_weight_strategy` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/test_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/params/subsample` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/19616f78518143a1bf22ccea1bee16e1/outputs/m-a4173289b31d4df0831dcb3334caf055/meta.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/meta.yaml` | Other | Repository support artifact. | 0.5 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/metrics/junior_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/metrics/other_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/metrics/macro_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/metrics/manager_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/artifacts/MLmodel` | Other | Repository support artifact. | 3.8 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/artifacts/serving_input_example.json` | Data Artifact | Repository support artifact. | 2.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/artifacts/model.pkl` | Model Artifact | Repository support artifact. | 1163.9 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/artifacts/input_example.json` | Data Artifact | Repository support artifact. | 1.4 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/artifacts/conda.yaml` | Other | Repository support artifact. | 0.3 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/tags/mlflow.user` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/tags/mlflow.source.name` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/tags/mlflow.source.type` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/max_depth` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/learning_rate` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/n_features` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/calibrated` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/n_estimators` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/feature_engineering_version` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/val_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/model` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/train_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/class_weight_strategy` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/test_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-5940d153693247f585da70246fa520a2/params/subsample` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/meta.yaml` | Other | Repository support artifact. | 0.5 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/metrics/junior_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/metrics/other_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/metrics/macro_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/metrics/manager_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/artifacts/MLmodel` | Other | Repository support artifact. | 3.8 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/artifacts/serving_input_example.json` | Data Artifact | Repository support artifact. | 2.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/artifacts/model.pkl` | Model Artifact | Repository support artifact. | 1163.9 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/artifacts/input_example.json` | Data Artifact | Repository support artifact. | 1.4 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/artifacts/conda.yaml` | Other | Repository support artifact. | 0.3 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/tags/mlflow.user` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/tags/mlflow.source.name` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/tags/mlflow.source.type` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/max_depth` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/learning_rate` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/n_features` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/calibrated` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/n_estimators` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/feature_engineering_version` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/val_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/model` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/train_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/class_weight_strategy` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/test_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-e84cf45d8df94a739ca16e81c030bb06/params/subsample` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/meta.yaml` | Other | Repository support artifact. | 0.5 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/metrics/junior_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/metrics/other_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/metrics/macro_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/metrics/manager_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/artifacts/python_env.yaml` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/artifacts/requirements.txt` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/artifacts/MLmodel` | Other | Repository support artifact. | 3.8 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/artifacts/serving_input_example.json` | Data Artifact | Repository support artifact. | 2.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/artifacts/model.pkl` | Model Artifact | Repository support artifact. | 1163.9 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/artifacts/input_example.json` | Data Artifact | Repository support artifact. | 1.4 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/artifacts/conda.yaml` | Other | Repository support artifact. | 0.3 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/tags/mlflow.user` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/tags/mlflow.source.name` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/tags/mlflow.source.type` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/max_depth` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/learning_rate` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/n_features` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/calibrated` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/n_estimators` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/feature_engineering_version` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/val_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/model` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/train_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/class_weight_strategy` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/test_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/models/m-a4173289b31d4df0831dcb3334caf055/params/subsample` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/tags/mlflow.experimentKind` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/meta.yaml` | Other | Repository support artifact. | 0.5 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/metrics/junior_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/metrics/other_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/metrics/macro_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/metrics/manager_f1` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/artifacts/artifacts/label_encoder.pkl` | Model Artifact | Trained LabelEncoder mapping ['hr', 'junior', 'manager', 'other']. | 0.5 KB | 2026-07-03 16:54 | YES | YES | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/artifacts/artifacts/tfidf_svd.joblib` | Model Artifact | Fitted TruncatedSVD artifact (global fit, 32 components). | 215.5 KB | 2025-12-11 06:57 | YES | YES | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/artifacts/artifacts/tfidf_vectorizer.joblib` | Model Artifact | Fitted TfidfVectorizer artifact (global fit). | 32.8 KB | 2025-12-11 06:57 | YES | YES | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/artifacts/artifacts/eval_metrics.json` | Data Artifact | Saved metrics from train_pipeline.py (Macro F1 = 0.6351). | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/tags/mlflow.user` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/tags/mlflow.runName` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/tags/mlflow.source.name` | Other | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/tags/mlflow.source.type` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/max_depth` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/learning_rate` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/n_features` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/calibrated` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/n_estimators` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/feature_engineering_version` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/val_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/model` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/train_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/class_weight_strategy` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/test_samples` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/params/subsample` | Other | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/603949218934757692/6fe60e91ba7844df965d7a25b21eb5c5/outputs/m-e84cf45d8df94a739ca16e81c030bb06/meta.yaml` | Other | Repository support artifact. | 0.2 KB | 2026-07-03 16:54 | YES | NO | YES |
| `mlruns/2/traces/tr-3a9b9ebb834436141702e754b5764495/artifacts/attachments/d2cf5d0b-8725-4bdb-bc75-2fdc971b7620` | Other | Repository support artifact. | 0.1 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-61f375538e10fe38c3887cf105bcfab7/artifacts/traces.json` | Data Artifact | Repository support artifact. | 2.7 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-6ebad81061bf3a8f108d9774215f6e76/artifacts/traces.json` | Data Artifact | Repository support artifact. | 3.0 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-fe8dd17075dfd8a15488ebbda53a4b3c/artifacts/attachments/bc2aafaa-1175-41e7-ac3c-9010c14e1d06` | Other | Repository support artifact. | 3.9 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-89c5946fff22b88421b8acd908bd5ace/artifacts/traces.json` | Data Artifact | Repository support artifact. | 4.3 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-81d381ead712da3ab842563979cf545a/artifacts/traces.json` | Data Artifact | Repository support artifact. | 4.7 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-bddc8ae8a548cc44bdd354345e8758ed/artifacts/traces.json` | Data Artifact | Repository support artifact. | 2.8 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-68d9cdcd9f86cb5db7cb1cbe221de154/artifacts/traces.json` | Data Artifact | Repository support artifact. | 1.2 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-68d9cdcd9f86cb5db7cb1cbe221de154/artifacts/attachments/886bd057-eae6-41a2-adf6-e424b14db52a` | Other | Repository support artifact. | 0.1 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-d44cb38cb5773a283cb5601e2fe9e51f/artifacts/attachments/182c5d07-c5df-4a27-a905-bac8b9c5ca86` | Other | Repository support artifact. | 0.1 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-36cb52ab6920ee7d6847382f1a0bcf96/artifacts/attachments/b49bcdfc-5c94-4218-b98f-1c03e4faae04` | Other | Repository support artifact. | 3.9 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-15240959fe9b288953c4a981124ba991/artifacts/traces.json` | Data Artifact | Repository support artifact. | 1.9 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-c035bdf6b8081b08932ff458d46419cf/artifacts/traces.json` | Data Artifact | Repository support artifact. | 1.9 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-f37f85a41ec265802116531cb454ce28/artifacts/traces.json` | Data Artifact | Repository support artifact. | 3.2 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-5e12b05e1a8c697da83f1ba181b7239f/artifacts/traces.json` | Data Artifact | Repository support artifact. | 1.4 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-5e12b05e1a8c697da83f1ba181b7239f/artifacts/attachments/e2232f81-b207-44fa-b784-030c49377123` | Other | Repository support artifact. | 3.9 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-4659c09583ead2acb4dd7b45bce1aed7/artifacts/attachments/6b9d4d70-1beb-4e37-a3cd-aa8cd1844326` | Other | Repository support artifact. | 3.9 KB | 2026-06-06 20:35 | YES | NO | YES |
| `mlruns/2/traces/tr-06a9261b8b9090c07ffd961383d61e44/artifacts/attachments/0318767b-c469-4d13-bfc4-f5265d54d933` | Other | Repository support artifact. | 0.1 KB | 2026-06-06 20:35 | YES | NO | YES |
| `audit/predictions.csv` | Data Artifact | Repository support artifact. | 24.6 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/AUDIT_02_DATASET_PROFILE.md` | Documentation / Report | Repository support artifact. | 1.0 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/AUDIT_02_DATASET_PROFILE.csv` | Data Artifact | Repository support artifact. | 0.0 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/PRESENTATION_VERIFIED_RESULTS.md` | Documentation / Report | Repository support artifact. | 1.0 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/classification_report.csv` | Data Artifact | Repository support artifact. | 0.7 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/feature_ablation.csv` | Data Artifact | Repository support artifact. | 0.5 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/baseline_comparison.csv` | Data Artifact | Repository support artifact. | 0.4 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/evaluation_results.csv` | Data Artifact | Repository support artifact. | 0.6 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/CLAIM_AUDIT.md` | Documentation / Report | Repository support artifact. | 0.9 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/LEAKAGE_AUDIT.md` | Documentation / Report | Repository support artifact. | 1.5 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/generate_repo_inventory.py` | Python Script | Repository support artifact. | 5.9 KB | 2026-08-15 01:43 | YES | NO | YES |
| `audit/leader_identification_results.csv` | Data Artifact | Repository support artifact. | 2.0 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/run_audit.py` | Python Script | Repository support artifact. | 30.9 KB | 2026-08-15 01:38 | YES | NO | YES |
| `audit/confusion_matrix.csv` | Data Artifact | Repository support artifact. | 0.1 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/FINAL_ROLE_INFERENCE_AUDIT_REPORT.md` | Documentation / Report | Repository support artifact. | 4.2 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/REPRODUCIBILITY_AUDIT.md` | Documentation / Report | Repository support artifact. | 0.9 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/figures/confusion_matrix.png` | Image / Visualization | Repository support artifact. | 77.4 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/figures/confidence_distribution.png` | Image / Visualization | Repository support artifact. | 80.1 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/figures/normalized_confusion_matrix.png` | Image / Visualization | Repository support artifact. | 106.7 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/figures/class_distribution.png` | Image / Visualization | Repository support artifact. | 69.5 KB | 2026-08-15 01:40 | YES | NO | YES |
| `audit/figures/feature_ablation.png` | Image / Visualization | Repository support artifact. | 99.2 KB | 2026-08-15 01:40 | YES | NO | YES |
| `notebook/eval_on_test.ipynb` | Jupyter Notebook | Notebook evaluating model on test set (contains split bug with 0 HR samples). | 43.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `notebook/02_feature_engineering.ipynb` | Jupyter Notebook | Repository support artifact. | 15.5 KB | 2026-07-03 16:54 | YES | NO | YES |
| `notebook/01_eda.ipynb` | Jupyter Notebook | Repository support artifact. | 32.3 KB | 2026-07-03 16:54 | YES | NO | YES |
| `notebook/04_model_training_mlflow_integrated.ipynb` | Jupyter Notebook | Repository support artifact. | 102.3 KB | 2026-07-03 16:54 | YES | NO | YES |
| `notebook/03_model_training.ipynb` | Jupyter Notebook | Repository support artifact. | 290.3 KB | 2026-07-03 16:54 | YES | NO | YES |
| `data/demo.csv` | Data Artifact | Repository support artifact. | 15.7 KB | 2026-07-03 16:54 | YES | NO | YES |
| `data/features.csv` | Data Artifact | Speaker-level aggregated dataset with handcrafted and SVD features (1,133 rows). | 796.3 KB | 2026-07-03 16:54 | YES | NO | YES |
| `data/labeled_roles.csv` | Data Artifact | Raw utterance-level ground-truth dataset (1,555 rows). | 210.7 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/confusion_matrix.png` | Image / Visualization | Repository support artifact. | 32.7 KB | 2025-12-11 07:51 | YES | NO | YES |
| `reports/role_purity.csv` | Data Artifact | Repository support artifact. | 9.2 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/role_distribution.png` | Image / Visualization | Repository support artifact. | 12.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/validation_calibration_stats.csv` | Data Artifact | Repository support artifact. | 0.2 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/role_dist.png` | Image / Visualization | Repository support artifact. | 10.7 KB | 2025-12-11 05:46 | YES | NO | YES |
| `reports/validation_edge_shap.csv` | Data Artifact | Repository support artifact. | 4.6 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/speakers_per_meeting.csv` | Data Artifact | Repository support artifact. | 3.5 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/validation_classification_report.txt` | Other | Repository support artifact. | 0.4 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/mistakes_examples.csv` | Data Artifact | Repository support artifact. | 5.4 KB | 2025-12-11 07:51 | YES | NO | YES |
| `reports/speaker_label_counts.csv` | Data Artifact | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/role_distribution_speaker_level.png` | Image / Visualization | Repository support artifact. | 16.2 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/validation_stress_shap.csv` | Data Artifact | Repository support artifact. | 4.5 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/xgb_global_importances.csv` | Data Artifact | Repository support artifact. | 2.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/label_counts.csv` | Data Artifact | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/eval_metrics_full.json` | Data Artifact | Saved metrics from eval_on_test.ipynb (Macro F1 = 0.8836, 3-class only). | 1.0 KB | 2025-12-11 07:51 | YES | NO | PARTIAL / LEAKED |
| `reports/eda_report.md` | Documentation / Report | Repository support artifact. | 0.7 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/validation_standard.csv` | Data Artifact | Repository support artifact. | 10.9 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/validation_confusion_matrix.csv` | Data Artifact | Repository support artifact. | 0.1 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/feature_stats.csv` | Data Artifact | Repository support artifact. | 6.2 KB | 2025-12-11 06:57 | YES | NO | YES |
| `reports/validation_stress.csv` | Data Artifact | Repository support artifact. | 2.9 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/validation_standard_shap.csv` | Data Artifact | Repository support artifact. | 16.8 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/validation_edge.csv` | Data Artifact | Repository support artifact. | 2.8 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/eval_metrics.json` | Data Artifact | Saved metrics from train_pipeline.py (Macro F1 = 0.6351). | 0.7 KB | 2026-07-03 16:54 | YES | NO | YES |
| `reports/utterance_label_counts.csv` | Data Artifact | Repository support artifact. | 0.0 KB | 2026-07-03 16:54 | YES | NO | YES |
