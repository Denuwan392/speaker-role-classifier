# predict_role.py (robust version)
import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
from textblob import TextBlob
import joblib

# --- Paths (adjust if needed) ---
MODEL_DIR = Path("models")
CACHE_DIR = Path("cache")
LLM_CACHE_PATH = CACHE_DIR / "llm_manager_prob_cache.json"

# Ensure cache dir exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Load artifacts with safe fallbacks ---
print("📦 Loading model artifacts...")
label_encoder_path = MODEL_DIR / "label_encoder.pkl"
tfidf_path = MODEL_DIR / "tfidf_vectorizer.joblib"
svd_path = MODEL_DIR / "tfidf_svd.joblib"
calibrated_model_path = MODEL_DIR / "role_classifier_calibrated.pkl"
plain_model_path = MODEL_DIR / "role_classifier.pkl"

if not label_encoder_path.exists():
    raise FileNotFoundError(f"Missing label encoder: {label_encoder_path}")
label_encoder = joblib.load(label_encoder_path)

if not tfidf_path.exists() or not svd_path.exists():
    raise FileNotFoundError("Missing TF-IDF / SVD artifacts in models/")

tfidf = joblib.load(tfidf_path)
svd = joblib.load(svd_path)

# Prefer calibrated model, fallback to plain model
if calibrated_model_path.exists():
    model = joblib.load(calibrated_model_path)
    print(f"Loaded calibrated model: {calibrated_model_path.name}")
elif plain_model_path.exists():
    model = joblib.load(plain_model_path)
    print(f"Loaded model: {plain_model_path.name} (no calibration)")
else:
    raise FileNotFoundError("No model found in models/ (expected calibrated or plain model)")

# Load LLM cache if exists
if LLM_CACHE_PATH.exists():
    with open(LLM_CACHE_PATH, "r") as f:
        try:
            llm_cache = json.load(f)
        except Exception:
            llm_cache = {}
else:
    llm_cache = {}

# --- Helpers ---
def _normalize_text_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def _get_llm_score(text: str) -> float:
    key = _normalize_text_key(text)
    return float(llm_cache.get(key, 0.5))

def _safe_div(a, b, eps=1e-9):
    return a / (b + eps)

# directive / uncertainty patterns (shared)
HARD_DIRECTIVE_PATTERNS = [
    r"\byou should\b", r"\bdo this\b", r"\bmake sure\b", r"\bneed to\b",
    r"\bhave to\b", r"\bplease complete\b", r"\bthis is your task\b",
    r"\bassign\b", r"\breview\b.*\bPR\b", r"\bblocker\b", r"\bdeadline\b",
    r"\bI expect\b", r"\bensure\b", r"\bby EOD\b"
]
UNCERTAINTY_PATTERNS = [
    r"\bnot sure\b", r"\bstuck\b", r"\bconfused\b", r"\btrying to\b",
    r"\bi think\b", r"\bmaybe\b", r"\bkind of\b", r"\bsort of\b",
    r"\bdon't know\b", r"\bcan't figure\b", r"\bstruggling\b"
]
hard_re = re.compile("|".join(HARD_DIRECTIVE_PATTERNS), re.IGNORECASE)
uncertainty_re = re.compile("|".join(UNCERTAINTY_PATTERNS), re.IGNORECASE)

# tfidf dimension (from saved svd)
EXPECTED_TFIDF_DIM = svd.n_components if hasattr(svd, "n_components") else (svd.components_.shape[0] if hasattr(svd, "components_") else 32)

def _extract_features_for_speaker(text: str, meeting_df: pd.DataFrame) -> dict:
    text = (text or "").strip()
    words = text.split()
    word_count = len(words)
    sentence_count = len(re.findall(r"[.!?]+", text))
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0.0
    question_count = len(re.findall(r"\?", text))
    sentiment_score = TextBlob(text).sentiment.polarity if text else 0.0

    hard_directive_count = len(hard_re.findall(text))
    uncertainty_count = len(uncertainty_re.findall(text))

    # meeting_df expected to contain columns 'word_count' and 'hard_directive_count'
    mean_word = meeting_df["word_count"].mean() if not meeting_df["word_count"].empty else 0.0
    mean_hard = meeting_df["hard_directive_count"].mean() if not meeting_df["hard_directive_count"].empty else 0.0
    total_words = meeting_df["word_count"].sum() if not meeting_df["word_count"].empty else word_count

    word_count_rel = _safe_div(word_count, mean_word)
    directive_count_rel = _safe_div(hard_directive_count, mean_hard)
    word_count_share = _safe_div(word_count, total_words)

    # tfidf + svd vector (pad or trim to expected dim)
    vec = tfidf.transform([text])
    sv = svd.transform(vec).flatten()
    if len(sv) < EXPECTED_TFIDF_DIM:
        sv = np.concatenate([sv, np.zeros(EXPECTED_TFIDF_DIM - len(sv))])
    elif len(sv) > EXPECTED_TFIDF_DIM:
        sv = sv[:EXPECTED_TFIDF_DIM]

    lm_manager_prob = _get_llm_score(text)

    base_features = {
        "word_count": float(word_count),
        "avg_sentence_len": float(avg_sentence_len),
        "question_count": float(question_count),
        "hard_directive_count": float(hard_directive_count),
        "uncertainty_count": float(uncertainty_count),
        "sentiment_score": float(sentiment_score),
        "lm_manager_prob": float(lm_manager_prob),
        "word_count_rel": float(word_count_rel),
        "directive_count_rel": float(directive_count_rel),
        "word_count_share": float(word_count_share),
    }
    tfidf_features = {f"tfidf_{i}": float(sv[i]) for i in range(EXPECTED_TFIDF_DIM)}
    return {**base_features, **tfidf_features}

def _find_evidence_sentence(full_text: str, role: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", full_text.strip())
    if not sentences:
        return full_text
    if role == "manager":
        HARD_DIRECTIVE_RE = hard_re
        scores = [len(HARD_DIRECTIVE_RE.findall(s)) for s in sentences]
    elif role == "junior":
        UNCERTAINTY_RE = uncertainty_re
        scores = [len(UNCERTAINTY_RE.findall(s)) for s in sentences]
    else:
        scores = [len(s.split()) for s in sentences]
    best_idx = int(np.argmax(scores))
    return sentences[best_idx].strip()

# --- Main API function ---
def predict_role(transcript_segments):
    if not transcript_segments:
        return {}

    # Aggregate by speaker
    speaker_texts = {}
    for seg in transcript_segments:
        spk = seg.get("speaker_id")
        txt = seg.get("text", "")
        if not spk:
            continue
        speaker_texts.setdefault(spk, []).append(txt.strip())

    speaker_full_texts = {spk: " ".join(txts).strip() for spk, txts in speaker_texts.items()}

    temp_df = pd.DataFrame([{"speaker_id": spk, "full_text": txt} for spk, txt in speaker_full_texts.items()])
    temp_df["word_count"] = temp_df["full_text"].str.split().str.len().fillna(0).astype(float)
    # compute hard_directive_count for meeting context
    temp_df["hard_directive_count"] = temp_df["full_text"].apply(lambda x: len(hard_re.findall(x)))

    results = {}
    # Build features and predict
    # Determine feature order: try model.feature_names_in_, fallback to expected list
    feature_order = getattr(model, "feature_names_in_", None)
    if feature_order is None:
        base_cols = [
            "word_count", "avg_sentence_len", "question_count",
            "hard_directive_count", "uncertainty_count", "sentiment_score", "lm_manager_prob",
            "word_count_rel", "directive_count_rel", "word_count_share"
        ]
        tfidf_cols = [f"tfidf_{i}" for i in range(EXPECTED_TFIDF_DIM)]
        feature_order = base_cols + tfidf_cols

    for spk, full_text in speaker_full_texts.items():
        # meeting context: other speakers
        meeting_context = temp_df[temp_df["speaker_id"] != spk]
        if meeting_context.empty:
            meeting_context = temp_df  # fallback to self

        feats = _extract_features_for_speaker(full_text, meeting_context)
        X_row = np.array([[feats.get(c, 0.0) for c in feature_order]], dtype=float)

        proba = model.predict_proba(X_row)[0]
        pred_idx = int(np.argmax(proba))
        role = label_encoder.inverse_transform([pred_idx])[0]
        probability = float(proba[pred_idx])
        evidence = [_find_evidence_sentence(full_text, role)]
        results[spk] = {"role": role, "probability": probability, "evidence": evidence}

    return results

# CLI demo
if __name__ == "__main__":
    import json as _json
    segments = [
        {"speaker_id": "spk_1", "text": "Morning everyone. Let’s keep it quick—what did you do yesterday?"},
        {"speaker_id": "spk_1", "text": "Sure, check the /docs/api folder. And feel better!"},
        {"speaker_id": "spk_2", "text": "Yesterday I finished the auth middleware. Today I’ll start rate-limiting."},
        {"speaker_id": "spk_3", "text": "I was out sick yesterday. I’m not sure where the doc files live—can someone point me?"},
    ]
    print(_json.dumps(predict_role(segments), indent=2))
