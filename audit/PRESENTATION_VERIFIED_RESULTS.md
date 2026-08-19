# PRESENTATION VERIFIED RESULTS (AUTHORITATIVE SUMMARY)

## Academic Headline
> "Under a strict un-leaked meeting-grouped evaluation, the 4-class Speaker Role Classifier achieves a **mean ± standard deviation across 20 group-aware splits** of **69.5% ± 3.7% accuracy** (mean **0.681 ± 0.032 Macro F1**) and **71.9% ± 6.7% Leader identification accuracy**."

## Verified Core Metrics Table
- **Primary Evaluated Dataset:** Dataset B (4-Class Expanded Standup Dataset, 1,133 speaker-level samples, 294 meetings)
- **Evaluation Protocol:** Group-aware meeting split (70% train / 15% val / 15% test) with train-only TF-IDF/SVD fitting.
- **Overall Accuracy:** `69.45% ± 3.70%` (mean ± standard deviation across 20 group-aware splits)
- **Macro F1 Score:** `0.6814 ± 0.0317` (mean ± standard deviation across 20 group-aware splits)
- **Leader Identification Accuracy:** `71.86% ± 6.66%` (conditional Top-1 manager selection among meetings containing a ground-truth manager)
- **Seed 0 Single Reference Split:** Accuracy = `63.64%`, Macro F1 = `0.6317`, Leader Acc = `62.50%`.

---

## Legacy 3-Class Benchmark (Dataset A: 394 samples, 110 meetings)
- **Classes:** `['junior', 'manager', 'other']` (**HR Omitted**)
- **Clean Un-leaked Accuracy:** `85.96%`
- **Clean Un-leaked Macro F1:** `0.8446`
- **Explanation:** The legacy 394-sample dataset omitted the HR class completely, yielding higher classification scores.

---

## ⚠️ NUMBERS WE MUST NEVER PRESENT IN ACADEMIC PAPERS / SLIDES
1. **DO NOT PRESENT "89.5% Accuracy" or "0.896 F1"**: Derived from `eval_metrics_full.json` on a 3-class test split excluding HR.
2. **DO NOT PRESENT "90.9% Test Accuracy"**: Result of severe data leakage caused by fitting TF-IDF and SVD globally on the full dataset before splitting.
