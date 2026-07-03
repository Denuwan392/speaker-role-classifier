#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# 02_feature_engineering.py
# Speaker-Level Role Detection (Leader / HR / Junior / Other)
# ============================================================

import re
from pathlib import Path
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------- Config --------------------
INPUT_PATH = "data/labeled_roles.csv"
OUTPUT_PATH = "data/features.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Load Data --------------------
df = pd.read_csv(INPUT_PATH)
print(f"✅ Loaded {len(df)} utterances")

required_cols = {"meeting_id", "speaker_id", "full_text", "role"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------------------- Text Normalization --------------------
df["full_text"] = df["full_text"].fillna("").astype(str)
df["text"] = df["full_text"].str.replace(r"\s+", " ", regex=True).str.strip()

# -------------------- Basic Linguistic Features --------------------
df["word_count"] = df["text"].str.split().str.len().astype(int)
df["sentence_count"] = df["text"].str.count(r"[.!?]+")
df["avg_sentence_len"] = np.where(
    df["sentence_count"] > 0,
    df["word_count"] / df["sentence_count"],
    0.0
)
df["question_count"] = df["text"].str.count(r"\?")

# -------------------- Directive Features --------------------
HARD_DIRECTIVE_PATTERNS = [
    r"\byou should\b", r"\byou need to\b", r"\bhave to\b",
    r"\bdeadline\b", r"\bby friday\b", r"\bblocker\b",
    r"\bi expect\b", r"\bensure\b"
]

SOFT_HELP_PATTERNS = [
    r"\bi['’]?ll help\b", r"\bi can help\b", r"\blet me help\b",
    r"\bwe can\b", r"\bi can walk you through\b"
]

hard_re = re.compile("|".join(HARD_DIRECTIVE_PATTERNS), re.IGNORECASE)
soft_re = re.compile("|".join(SOFT_HELP_PATTERNS), re.IGNORECASE)

df["hard_directive_count"] = df["text"].apply(lambda x: len(hard_re.findall(x)))
df["soft_help_count"] = df["text"].apply(lambda x: len(soft_re.findall(x)))
df["directive_count"] = df["hard_directive_count"] + df["soft_help_count"]

# -------------------- Uncertainty --------------------
UNCERTAINTY_PATTERNS = [
    r"\bnot sure\b", r"\bstuck\b", r"\btrying to\b",
    r"\bi think\b", r"\bmaybe\b", r"\bconfused\b"
]

uncertainty_re = re.compile("|".join(UNCERTAINTY_PATTERNS), re.IGNORECASE)
df["uncertainty_count"] = df["text"].apply(lambda x: len(uncertainty_re.findall(x)))

# -------------------- Greeting-only --------------------
greeting_re = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening)[\s!.,]*$",
    re.IGNORECASE
)

df["is_greeting_only"] = df["text"].apply(
    lambda x: 1 if greeting_re.match(x.strip()) else 0
)

# -------------------- Sentiment --------------------
df["sentiment_score"] = df["text"].apply(
    lambda x: TextBlob(x).sentiment.polarity if x else 0.0
)

# ============================================================
# 🔥 SPEAKER-LEVEL AGGREGATION (CORE UPGRADE)
# ============================================================

speaker_df = (
    df.groupby(["meeting_id", "speaker_id", "role"])
    .agg({
        "text": " ".join,
        "word_count": "sum",
        "sentence_count": "sum",
        "avg_sentence_len": "mean",
        "question_count": "sum",
        "hard_directive_count": "sum",
        "soft_help_count": "sum",
        "directive_count": "sum",
        "uncertainty_count": "sum",
        "is_greeting_only": "sum",
        "sentiment_score": "mean"
    })
    .reset_index()
)

# Count turns per speaker
turn_counts = (
    df.groupby(["meeting_id", "speaker_id"])
    .size()
    .reset_index(name="turn_count")
)

speaker_df = speaker_df.merge(
    turn_counts,
    on=["meeting_id", "speaker_id"],
    how="left"
)

print(f"✅ Speaker-level rows: {len(speaker_df)}")

# ============================================================
# RELATIVE FEATURES (PER MEETING)
# ============================================================

# grouped = speaker_df.groupby("meeting_id")

# speaker_df["word_count_share"] = (
#     speaker_df["word_count"] /
#     (grouped["word_count"].transform("sum") + 1e-9)
# )

# speaker_df["directive_share"] = (
#     speaker_df["directive_count"] /
#     (grouped["directive_count"].transform("sum") + 1e-9)
# )

# speaker_df["turn_share"] = (
#     speaker_df["turn_count"] /
#     (grouped["turn_count"].transform("sum") + 1e-9)
# )

# ============================================================
# TF-IDF + SVD (ON SPEAKER TEXT)
# ============================================================

tfidf = TfidfVectorizer(
    max_features=1500,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9
)

tfidf_matrix = tfidf.fit_transform(speaker_df["text"])

n_components = min(32, tfidf_matrix.shape[1] - 1) if tfidf_matrix.shape[1] > 1 else 1
svd = TruncatedSVD(n_components=n_components, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_reduced.shape[1])]

speaker_df = pd.concat(
    [speaker_df,
     pd.DataFrame(tfidf_reduced, columns=tfidf_cols, index=speaker_df.index)],
    axis=1
)

joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.joblib")
joblib.dump(svd, MODEL_DIR / "tfidf_svd.joblib")

# ============================================================
# FINAL FEATURE SET
# ============================================================

feature_cols = [
    "meeting_id", "speaker_id", "role",
    "word_count", "avg_sentence_len", "question_count",
    "hard_directive_count", "soft_help_count", "directive_count",
    "uncertainty_count", "sentiment_score"
] + tfidf_cols

df_features = speaker_df[feature_cols]
df_features.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Saved speaker-level features → {OUTPUT_PATH}")
print(df_features.head())


# In[2]:


import pandas as pd

df = pd.read_csv("data/features.csv")

# ------------------------------
# Basic shape check
# ------------------------------
print("📦 Dataset shape:", df.shape)

print("\n📋 Columns:")
print(df.columns.tolist())

# ------------------------------
# Role distribution (IMPORTANT)
# ------------------------------
print("\n🎯 Role distribution:")
print(df["role"].value_counts())

# ------------------------------
# Relative feature sanity check
# ------------------------------
#print("\n📊 Relative feature stats:")
#print(df[[
   # "word_count_share",
  #  "directive_share",
 #   "turn_share"
#]].describe())

# ------------------------------
# Dominance sanity check
# ------------------------------
#print("\n🔎 Check that word_count_share sums to ~1 per meeting:")

#meeting_sums = (
#    df.groupby("meeting_id")["word_count_share"]
#    .sum()
#    .describe()
#)

#print(meeting_sums)

# ------------------------------
# Turn distribution insight
# ------------------------------
#print("\n🗣️ Turn count distribution:")
#print(df["turn_count"].describe())

# ------------------------------
# Speaker count per meeting
# ------------------------------
print("\n👥 Speakers per meeting:")
print(df.groupby("meeting_id")["speaker_id"].nunique().describe())


# In[ ]:




