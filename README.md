# Speaker Role Classifier
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Multi--Class-green)
![MLflow](https://img.shields.io/badge/MLflow-Experiment_Tracking-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-red)

This repository presents a speaker role classification system designed to identify the functional roles of participants in software engineering meetings based solely on their spoken utterances. The system categorizes speakers into four distinct roles: Lead, Human Resources (HR), Junior Developer, and Other.

The classification is performed using a hybrid feature representation that combines handcrafted linguistic heuristics with semantic embeddings derived from TF-IDF and Truncated Singular Value Decomposition (SVD). An XGBoost classifier is employed to map these 40-dimensional feature vectors to the respective role labels.

This module is developed as a standalone component for the broader SpeechInSight project, aiming to facilitate automated speech analytics and meeting summarization in professional software development environments.

## Overview

- **Project Objective**: To accurately predict the functional role of a speaker in a software engineering meeting using only textual utterances.
- **Research Motivation**: Understanding speaker dynamics and role distributions in meetings is critical for automated meeting analysis, discourse tracking, and organizational behavior studies.
- **Problem Statement**: Identifying speaker roles from unstructured, multi-turn conversational data without relying on explicit metadata or speaker identities.
- **SpeechInSight Integration**: This repository contains the standalone Speaker Role Classification module developed for the SpeechInSight project. It is designed to integrate into larger speech analytics pipelines while remaining independently trainable and deployable.

## Methodology

- **Data Aggregation**: Utterances are aggregated per speaker across the meeting context to form a comprehensive textual representation for each participant.
- **Feature Engineering**: A 40-dimensional feature vector is constructed for each speaker, combining domain-specific linguistic heuristics with distributional semantic representations.
- **TF-IDF & Truncated SVD**: Textual data is vectorized using Term Frequency-Inverse Document Frequency (TF-IDF) and subsequently reduced to 32 dimensions via Truncated Singular Value Decomposition (SVD) to capture latent semantic structures.
- **Handcrafted Linguistic Features**: Eight domain-specific features are extracted, including word count, sentence length, directive frequency, and uncertainty markers.
- **XGBoost Classifier**: A multi-class XGBoost model is trained to classify the aggregated feature vectors into the predefined role categories.
- **Inference Pipeline**: A modular inference engine processes raw diarized transcripts through the feature extraction and classification stages to output role predictions with confidence scores and supporting evidence.

## Model Architecture

```text
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

## Features

- Multi-class speaker role classification (Lead, HR, Junior Developer, Other).
- Hybrid feature extraction combining linguistic heuristics and semantic embeddings.
- Group-aware data splitting to prevent data leakage across meetings.
- Automated experiment tracking and evaluation report generation via MLflow.
- Modular and reusable inference pipeline for standalone or integrated deployment.
- Interactive visualization dashboard for qualitative model testing.

## Repository Structure

```text
speaker-role-classifier/
├── app.py
├── predict_role.py
├── inference.py
├── train_pipeline.py
├── feature_pipeline.py
├── agentic_router.py
├── models/
│   ├── role_classifier.pkl
│   ├── label_encoder.pkl
│   └── tfidf_svd.joblib
├── reports/
├── data/
├── notebook/
├── mlflow.db
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Denuwan392/speaker-role-classifier.git
cd speaker-role-classifier

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Training

To execute the training pipeline, which includes data preprocessing, feature extraction, model training, and evaluation:

```bash
python train_pipeline.py
```

To monitor training experiments and metrics:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Inference

For standalone inference using the trained model:

```bash
python inference.py
```

Example usage in Python:

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

Example output:

```json
{
  "spk_1": {
    "role": "lead",
    "probability": 0.98,
    "probs": {
      "lead": 0.98,
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

## Streamlit Demo

To launch the interactive testing dashboard:

```bash
streamlit run app.py
```

Access the interface at `http://localhost:8501`.

## Feature Engineering

The model utilizes a 40-dimensional feature vector for each speaker, partitioned into two categories:

**Handcrafted Linguistic Features (8 dimensions)**
- Word Count
- Average Sentence Length
- Question Count
- Hard Directive Count
- Soft Directive Count
- Uncertainty Count
- HR Keyword Count
- Greeting Count

**Semantic Features (32 dimensions)**
- TF-IDF Vectorization
- TruncatedSVD Dimensionality Reduction

The identical feature pipeline is instantiated during both training and inference to ensure consistency.

## Training Pipeline

- **Group-Aware Splitting**: Data is partitioned into training, validation, and test sets (70/15/15 ratio) at the meeting level to prevent data leakage.
- **Class Imbalance Handling**: Synthetic Minority Over-sampling Technique (SMOTE) is applied exclusively to the training set.
- **Model Training**: An XGBoost multi-class classifier is trained with early stopping based on validation metrics.
- **Experiment Tracking**: Training runs, hyperparameters, and artifacts are logged using MLflow.
- **Serialization**: The trained model, label encoder, and feature extractors are serialized for deployment.

## Evaluation

The pipeline automatically generates comprehensive evaluation artifacts stored in the `reports/` directory:
- Classification reports and macro F1 scores
- Confusion matrices
- Class-wise performance metrics
- SHAP feature importance analysis
- Stress testing results
- Probability calibration statistics

## Technologies

- Python
- XGBoost
- Scikit-learn
- Pandas & NumPy
- MLflow
- Streamlit
- Joblib
- imbalanced-learn (SMOTE)

## Citation

```bibtex
@misc{wijesinghe2026speakerrole,
  author = {Wijesinghe, Darshana Denuwan},
  title = {Speaker Role Classifier: A Hybrid Feature Approach for Meeting Participant Classification},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Denuwan392/speaker-role-classifier}}
}
```

## License

This project is licensed under the MIT License.

## Author

**Darshana Denuwan Wijesinghe**  
BSc (Hons) in Artificial Intelligence  
University of Moratuwa
