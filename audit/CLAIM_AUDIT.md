# CLAIM AUDIT TABLE

| Claimed Metric / Statement | Location | Empirical Forensic Verification | Status |
|---|---|---|---|
| "89.5% accuracy" | `eval_metrics_full.json` / Notebook | Calculated on test split with **0 HR samples** (3-class only) | **UNSUPPORTED / INVALID** |
| "0.896 weighted F1" | `eval_metrics_full.json` | Calculated on 3-class test split excluding HR | **UNSUPPORTED / INVALID** |
| "F1 > 0.87" | `README.md` line 9 | Un-leaked 4-class Macro F1 is **0.6317** | **UNSUPPORTED** |
| "78.3% accuracy" | `model_validation_report.md` | Measured on 60-sample synthetic standard suite | **PARTIALLY SUPPORTED** |
| "Calibrated probabilities" | `README.md` line 11 | Uncalibrated; 11.55% calibration gap on high confidence | **UNSUPPORTED** |
| "Group-aware split 70/15/15" | `train_pipeline.py` | Verified in code | **VERIFIED** |
| "4-class role prediction" | Code & Models | `['hr', 'junior', 'manager', 'other']` verified | **VERIFIED** |
