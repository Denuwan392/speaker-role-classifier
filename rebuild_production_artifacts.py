#!/usr/bin/env python3
"""
Rebuilds production model artifacts in models/ using an un-leaked train-only fitting pipeline.
- Fits TfidfVectorizer and TruncatedSVD on training meeting texts ONLY.
- Fits SMOTE on training features ONLY.
- Fits XGBClassifier on resampled training set.
- Saves clean artifacts to models/
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

def main():
    print("🚀 Rebuilding production artifacts cleanly (un-leaked train-only pipeline)...")

    df_raw = pd.read_csv(DATA_DIR / "labeled_roles.csv")
    df_feat = pd.read_csv(DATA_DIR / "features.csv")

    # Aggregate full text per speaker per meeting
    spk_text = df_raw.groupby(['meeting_id', 'speaker_id'])['full_text'].apply(
        lambda x: ' '.join(x.astype(str))
    ).reset_index()
    spk_text.rename(columns={'full_text': 'text'}, inplace=True)
    df_all = df_feat.merge(spk_text, on=['meeting_id', 'speaker_id'], how='left')

    # Group-aware split by meeting_id (Seed 0)
    meetings = df_all['meeting_id'].unique()
    np.random.seed(0)
    shuffled = meetings.copy()
    np.random.shuffle(shuffled)

    n_m = len(shuffled)
    n_tr = int(0.70 * n_m)
    tr_meetings = shuffled[:n_tr]

    train_df = df_all[df_all['meeting_id'].isin(tr_meetings)].copy()
    print(f"Training on {len(train_df)} samples across {len(tr_meetings)} meetings")

    # 1. Fit LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['role'])
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
    print(f"Saved LabelEncoder: {list(le.classes_)}")

    # 2. Fit TF-IDF on TRAIN ONLY
    tfidf = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.9)
    X_tr_tfidf = tfidf.fit_transform(train_df['text']).toarray()
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
    print("Saved TfidfVectorizer (fit on train text only)")

    # 3. Fit SVD on TRAIN ONLY
    svd = TruncatedSVD(n_components=32, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_tfidf)
    joblib.dump(svd, MODELS_DIR / "tfidf_svd.joblib")
    print("Saved TruncatedSVD (32 components, fit on train TF-IDF only)")

    # 4. Combine handcrafted base features and SVD features
    base_cols = [
        'word_count', 'avg_sentence_len', 'question_count',
        'hard_directive_count', 'soft_help_count', 'directive_count',
        'uncertainty_count', 'sentiment_score'
    ]
    X_tr_base = train_df[base_cols].values
    X_tr_clean = np.hstack([X_tr_base, X_tr_svd])

    # 5. Fit SMOTE on TRAIN ONLY
    smote = SMOTE(random_state=42)
    X_tr_res, y_tr_res = smote.fit_resample(X_tr_clean, y_train)

    # 6. Train XGBClassifier
    feature_names = base_cols + [f"tfidf_{i}" for i in range(32)]
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=len(le.classes_),
        max_depth=5,
        learning_rate=0.1,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )

    # Set feature names on XGBoost DataFrame input
    df_X_tr_res = pd.DataFrame(X_tr_res, columns=feature_names)
    model.fit(df_X_tr_res, y_tr_res)

    joblib.dump(model, MODELS_DIR / "role_classifier.pkl")
    print(f"Saved role_classifier.pkl ({len(feature_names)} feature names bound)")
    print("✅ Clean production artifacts rebuilt successfully!")

if __name__ == "__main__":
    main()
