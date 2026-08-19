# DATA LEAKAGE & RISK FORENSIC AUDIT

| Finding | Type | Severity | Status | Description & Impact |
|---|---|---|---|---|
| **Global TF-IDF Fit** | Feature Leakage | CRITICAL | **FAIL** | `TfidfVectorizer.fit_transform()` executed globally before splitting. Re-fitting on train-only drops accuracy from 90.91% to 63.64%. |
| **Global SVD Fit** | Feature Leakage | CRITICAL | **FAIL** | `TruncatedSVD.fit_transform()` executed globally before splitting. Re-fitting on train-only drops accuracy from 90.91% to 63.64%. |
| **Speaker ID Presence** | Feature Verification | NONE | **VERIFIED ABSENT** | `speaker_id` is **NOT present** in the 40-dim feature vector or `model.feature_names_in_`. |
| **Speaker ID Overlap** | Dataset-Identity Risk | MEDIUM | **RISK DOCUMENTED** | 5 synthetic placeholder IDs (`spk_1`..`spk_5`) are reused across all 294 meetings. While not in the feature vector, it indicates synthetic template structure. |
| **Evaluation Class Omission** | Metric Distortion | CRITICAL | **FAIL** | Legacy evaluation in `eval_on_test.ipynb` evaluated test set with 0 HR samples, producing artificial 89.5% accuracy. |
| **Val Set Threshold Search** | Tuning Risk | MEDIUM | **WARNING** | Threshold search over 83,521 combinations on validation probabilities overfitted thresholds. |
| **SMOTE Timing** | Resampling Protocol | NONE | **PASS** | Resampling applied strictly to `X_train` in `train_pipeline.py`. |
| **Meeting Group Splitting** | Partition Protocol | NONE | **PASS** | Meetings partitioned by `meeting_id` without overlap across splits. |

## Leader Identification Definition
Leader identification is explicitly defined as: **conditional Top-1 manager selection among meetings containing a ground-truth manager**.
