# AUDIT 02 — DATASET PROFILE & INTEGRITY REPORT

## 1. Executive Data Profile
- **Raw Utterance-level Dataset:** `data/labeled_roles.csv` (1555 samples)
- **Speaker-level Feature Dataset:** `data/features.csv` (1133 aggregated samples)
- **Total Unique Meetings:** 294 meetings
- **Unique Speaker Identifiers:** 5 placeholder IDs (`spk_1` to `spk_5`)

## 2. Speaker-Level Class Distribution
| Role | Sample Count | Percentage |
|---|---:|---:|
| `other` | 475 | 41.92% |
| `manager` | 275 | 24.27% |
| `junior` | 268 | 23.65% |
| `hr` | 115 | 10.15% |
| **Total** | **1,133** | **100.00%** |

## 3. Data Integrity & Anomalies
- **Missing Values:** 0 across all columns in `labeled_roles.csv` and `features.csv`.
- **Exact Duplicate Rows:** 6 duplicate rows in `labeled_roles.csv`.
- **Placeholder Speaker ID Leakage:** Only 5 unique `speaker_id` strings (`spk_1`, `spk_2`, `spk_3`, `spk_4`, `spk_5`) are reused across all 294 meetings.
- **Word Count Distribution:** Mean = 25.0 words, Median = 17.0 words, Min = 4 words, Max = 292 words per speaker turn aggregation.
