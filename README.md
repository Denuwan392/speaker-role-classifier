# 🗣️ Speaker Role Classifier

An AI-powered **speaker role classification system** for software engineering meetings. The model predicts whether a speaker is acting as a **Manager**, **HR**, **Junior Developer**, or **Other** based solely on their spoken utterances.

Built using **XGBoost**, handcrafted linguistic features, **TF-IDF + TruncatedSVD semantic representations**, and a modular production-ready inference pipeline.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Multi--Class-green)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)

---

# ✨ Features

- 🎯 Multi-class speaker role classification
  - Manager
  - HR
  - Junior Developer
  - Other

- 🧠 Modular feature engineering pipeline

- 📊 40-dimensional feature representation
  - 8 handcrafted linguistic features
  - 32 TF-IDF + TruncatedSVD semantic features

- ⚡ Production-ready inference pipeline

- 📈 MLflow experiment tracking

- 📋 Automatic evaluation reports

- 🔍 Evidence sentence extraction

- 📊 Interactive Streamlit testing dashboard

- 📁 Fully modular training pipeline

---

# 🚀 Quick Start

## 1. Clone Repository

```bash
git clone https://github.com/Denuwan392/speaker-role-classifier.git
cd speaker-role-classifier
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Launch Interactive Demo

```bash
streamlit run app.py
```

Open

```
http://localhost:8501
```

---

## 5. Run Standalone Inference

```bash
python inference.py
```

---

## 6. Train Model

```bash
python train_pipeline.py
```

---

## 7. Launch MLflow Dashboard

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open

```
http://localhost:5000
```

---

# Example Usage

```python
from predict_role import predict_role

segments = [
    {
        "speaker_id": "spk_1",
        "text": "Please complete the API implementation before Friday."
    },
    {
        "speaker_id": "spk_2",
        "text": "I'm still trying to understand OAuth."
    }
]

results = predict_role(segments)

print(results)
```

---

# Example Output

```json
{
  "spk_1": {
    "role": "manager",
    "probability": 0.98,
    "probs": {
      "manager": 0.98,
      "junior": 0.01,
      "hr": 0.00,
      "other": 0.01
    },
    "evidence": [
      "Please complete the API implementation before Friday."
    ]
  }
}
```

---

# Model Pipeline

```
Diarized Transcript
        │
        ▼
Speaker Aggregation
        │
        ▼
RoleFeatureExtractor
        │
        ├───────────────┐
        │               │
        ▼               ▼
Handcrafted      TF-IDF Vectorizer
 Features              │
        │              ▼
        │        Truncated SVD
        └───────┬──────────────
                ▼
      40-Dimensional Feature Vector
                │
                ▼
        XGBoost Classifier
                │
                ▼
 Role Prediction + Confidence
                │
                ▼
 Evidence Sentence Extraction
```

---

# Feature Engineering

The model generates a **40-dimensional feature vector** for every speaker.

### Handcrafted Features (8)

- Word Count
- Average Sentence Length
- Question Count
- Hard Directive Count
- Soft Directive Count
- Uncertainty Count
- HR Keyword Count
- Greeting Count

### Semantic Features (32)

- TF-IDF Vectorization
- TruncatedSVD Dimensionality Reduction

The exact same feature pipeline is shared between training and inference through the reusable **RoleFeatureExtractor**.

---

# Training Pipeline

The project includes a fully automated training workflow.

## Dataset Processing

- Feature preprocessing
- Label encoding
- Missing value handling

## Dataset Split

- Group-aware Train / Validation / Test split
- Meeting-level separation
- 70 / 15 / 15 ratio

## Class Imbalance

SMOTE oversampling is applied only to the training set.

## Model

- XGBoost Multi-Class Classifier

Training includes:

- Early stopping
- Validation monitoring
- Automatic model serialization
- MLflow experiment tracking
- Evaluation report generation

---

# Evaluation

The pipeline automatically produces:

- Classification Report
- Macro F1 Score
- Confusion Matrix
- Class-wise Metrics
- Validation Reports
- SHAP Feature Importance
- Stress Testing Results
- Calibration Statistics

All outputs are stored inside the **reports/** directory.

---

# Project Structure

```
speaker-role-classifier/

│
├── app.py
├── predict_role.py
├── inference.py
├── train_pipeline.py
├── feature_pipeline.py
├── agentic_router.py
│
├── models/
│   ├── role_classifier.pkl
│   ├── label_encoder.pkl
│   └── tfidf_svd.joblib
│
├── reports/
│
├── data/
│
├── notebook/
│
├── mlflow.db
│
└── requirements.txt
```

---

# Technologies Used

- Python
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- MLflow
- Streamlit
- Joblib
- TF-IDF
- TruncatedSVD
- imbalanced-learn (SMOTE)

---

# Future Improvements

- Probability Calibration
- Sentence Transformer Embeddings
- Context-aware Multi-turn Classification
- Incremental Model Updating
- Larger Multi-domain Dataset
- SpeechInSight Pipeline Integration

---

# License

MIT License

---

## Author

**Darshana Denuwan Wijesinghe**

BSc (Hons) Artificial Intelligence  
University of Moratuwa

---

> This repository contains the standalone Speaker Role Classification module developed for the SpeechInSight project. It is designed to integrate into larger speech analytics pipelines while remaining independently trainable and deployable.
