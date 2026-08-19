# REPRODUCIBILITY AUDIT REPORT

## Audit Status: PARTIAL / FAIL

### 1. Requirements & Dependency Verification
- `requirements.txt` lists: `scikit-learn`, `xgboost`, `pandas`, `joblib`.
- **Missing Dependencies:** `imbalanced-learn`, `textblob`, `google-genai`, `streamlit`, `mlflow`, `matplotlib`, `seaborn`.
- **Python Version:** Tested on Python 3.10 / 3.11.

### 2. Artifact Reproducibility
- **Model Artifact:** `models/role_classifier.pkl` is fully loadable.
- **Label Encoder:** `models/label_encoder.pkl` is loadable (`['hr', 'junior', 'manager', 'other']`).
- **TF-IDF & SVD Artifacts:** `tfidf_vectorizer.joblib` and `tfidf_svd.joblib` are loadable.

### 3. Notebook & Script Reproducibility
- `train_pipeline.py`: Runs cleanly, produces `eval_metrics.json` (Macro F1 = 0.6351).
- `eval_on_test.ipynb`: Reproducible, but suffers from split artifact where HR class has 0 test samples (reporting 89.47% 3-class accuracy).
