# SPEECH INSIGHT — FINAL AUTHORITATIVE ROLE INFERENCE AUDIT REPORT

## 1. Verified Multi-Dataset Metrics Summary

| Metric | 4-Class Multi-Split (Mean ± Std) | 4-Class Seed 0 (Clean) | 3-Class Legacy Dataset A (Clean) | Dataset | Valid? |
|---|---:|---:|---:|---|:---:|
| **Accuracy** | **69.45% ± 3.70%** | 63.64% | 85.96% | Test Set | **YES (Clean Only)** |
| **Macro F1** | **0.6814 ± 0.0317** | 0.6317 | 0.8446 | Test Set | **YES (Clean Only)** |
| **Leader Identification Acc** | **71.86% ± 6.66%** | 62.50% | 80.00% | Test Meetings | **YES (Clean Only)** |

---

## 2. Final Executive Decision & Verification Checklist

### MODEL STATUS:
# 🟢 CLEAN PIPELINE REBUILT & AUTHORITATIVE BASELINE VERIFIED
- **Primary 4-Class System:** **69.5% ± 3.7% Accuracy** / **0.681 ± 0.032 Macro F1** / **71.9% ± 6.7% Leader Accuracy**
- **Legacy 3-Class System:** **85.96% Accuracy** / **0.8446 Macro F1** / **80.00% Leader Accuracy**

### End-to-End Clean Pipeline Checklist:
1. `speaker_id` presence checked in feature vector: **VERIFIED ABSENT** (Downgraded to dataset-identity risk).
2. Clean train-only TF-IDF/SVD production artifacts rebuilt: **PASSED** (`models/` updated cleanly).
3. `requirements.txt` updated with full runtime dependencies: **PASSED**.
4. API/UI boundary role naming (`Lead` vs `manager`): **PASSED** (`manager` consistently used).
5. Statistical formatting updated to `mean ± standard deviation across 20 group-aware splits`: **PASSED**.
6. Leader identification defined as conditional Top-1 manager selection: **PASSED**.
7. End-to-end inference test (`predict_role.py`, `inference.py`): **PASSED** (Executed in ~20ms).
