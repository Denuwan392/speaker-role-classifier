# Speech Insight — Document Consistency Audit Report

This report documents the consistency audit performed between the primary source documents (**Group 03\_Interim Report.pdf**, **Group 03\_SRS Document.pdf**, **Project 3\_Embla Asia.pdf**, and the initial Word/LaTeX draft).

## Consistency Matrix

| Issue | Source Document(s) | Current Draft Wording | Recommended Action / Resolution |
| :--- | :--- | :--- | :--- |
| **Project Title Formatting** | Interim Report vs. SRS | Mixed `SpeechInsight` and `Speech Insight` | Standardized to **Speech Insight** across all document headers, titles, and text. |
| **Relational Database Technology** | Interim Report vs. SRS | Interim Report draft listed MongoDB in one passage alongside PostgreSQL, whereas SRS specifies PostgreSQL for structured records. | Enforced **PostgreSQL** (with `pgvector` extension) as the primary relational database, and vector store embeddings for RAG. |
| **LLM Model Name** | Draft LaTeX vs. SRS | Draft referenced generic Gemini / Gemini-1.5, whereas SRS specifies Gemini-2.5-flash. | Standardized to **Google Gemini-2.5-flash** across architecture descriptions and AI technology sections. |
| **Team Member Role Titles** | Interim Report vs. SRS | Roles were inconsistently formatted as `Lead`, `Team Lead`, or `ML Engineer`. | Standardized to formal titles: **Team Lead / Software Architect & ML Engineer**, **AI Researcher / Multimodal Engineer**, **AI Researcher / NLP Specialist**, **Audio Processing & Backend Engineer**. |
| **Chapter & Section Hierarchy** | Converted Draft LaTeX | Pandoc conversion generated unnumbered `\subsubsection` tags with manual dot leaders (`Introduction.............. 9`). | Replaced with native LaTeX structural hierarchy (`\chapter`, `\section`, `\subsection`, `\subsubsection`) with automatic numbering. |
| **Figure & Table Numbering** | Converted Draft LaTeX | Manually hardcoded labels like `Figure 1`, `Figure 5.1`, `Figure 2`. | Replaced with automatic `\caption{}` and `\label{}` cross-references (`\ref{fig:...}`). |
| **Mathematical Equations** | Converted Draft LaTeX | MAE and RMSE equations were inserted as raster images (`image8.png`, `image1.png`). | Replaced raster image equations with native LaTeX mathematical environments (`\begin{equation}` / `\sum`). |
| **Supervisor Details** | Title Page vs. Signatures | Titles formatted inconsistently (`Dr. ALARR Thanuja`, `Mr. Sivakumar Tharsan`). | Preserved official university & industry supervisor titles and appended formal designation (University Supervisor vs. Industry Supervisor). |

## Preserved Technical Facts

- Project Aim, Scope, and Architecture (Pyannote, Whisper, Gemini-2.5-flash, FastAPI, React).
- All quantitative evaluation results ($MAE = 0.42$, $RMSE = 0.58$, Role Accuracy $= 82.5\%$, $DER = 12.4\%$).
- 6-Stage Leadership Feedback Model (Warmup, Praise, Positive Suggestion, Negative Suggestion, Listen, Direct).
- All student index numbers (235503P, 235516H, 235535N, 235546A) and signatures.
