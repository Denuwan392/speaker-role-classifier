# 🗣️ Speaker Role Classifier

Predicts speaker roles (**manager**, **junior**, or **other**) from **diarized meeting transcripts** in software engineering stand-ups. Built with XGBoost, linguistic features, and LLM-enhanced signals.

![Example Output](https://img.shields.io/badge/Output-Manager_Junior_Other-blue)

## ✨ Features

- **Accurate role detection**: F1 > 0.87 on real-world-like data
- **Evidence extraction**: Highlights the key sentence that influenced the prediction
- **Calibrated probabilities**: Reliable confidence scores (e.g., `0.93`)
- **LLM-enhanced**: Uses cached Groq/Llama-3 scores for semantic nuance
- **Interactive demo**: Streamlit UI for real-time testing
- **Production-ready**: Modular, tested, and documented

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/your-username/speaker-role-classifier.git
cd speaker-role-classifier
```

### 2. Set up environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Set Groq API key for LLM features
```bash
echo "GROQ_API_KEY=your_api_key_here" > .env
```
> 💡 The model works without it (uses cached scores), but new texts will default to 0.5.

### 5. Run the Streamlit demo
```bash
streamlit run role_detection/app.py
```
Then open the URL shown (usually `http://localhost:8501`).

### 6. Or use the API directly
```python
from role_detection.predict_role import predict_role

segments = [
    {"speaker_id": "spk_1", "text": "What did you do yesterday?"},
    {"speaker_id": "spk_2", "text": "I'm stuck on the test case..."},
]

result = predict_role(segments)
print(result)
```

## 📂 Project Structure

```
role_detection/
├── predict_role.py          # Core inference function (main export)
├── demo.py                  # CLI example
├── app.py                   # Streamlit UI
├── models/                  # Trained model artifacts (.pkl, .joblib)
├── cache/                   # LLM response cache (auto-generated)
├── data/                    # Sample datasets (labeled_roles.csv, features.csv)
├── 01_eda.ipynb             # Exploratory data analysis
├── 02_feature_engineering.ipynb
├── 03_model_training.ipynb
└── ...
requirements.txt            # Python dependencies
.env.example                # Environment template
```

## 📊 Sample Output

```json
{
  "spk_1": {
    "role": "manager",
    "probability": 0.994,
    "evidence": ["Morning everyone. Let’s keep it quick—what did you do yesterday?"]
  },
  "spk_2": {
    "role": "junior",
    "probability": 0.69,
    "evidence": ["I’m not sure where the doc files live—can someone point me?"]
  }
}
```

## 🧠 How It Works

1. **Aggregates** all turns by `speaker_id`
2. **Extracts 48+ features** per speaker:
   - Linguistic: directives, uncertainty, questions
   - Relative: word share, speaking dominance
   - Semantic: TF-IDF + LLM score (via Groq)
3. **Predicts role** using a calibrated XGBoost classifier
4. **Returns evidence** (most indicative sentence)

Trained on **394 speaker samples** from realistic daily scrums.

## 📜 License

MIT License (see `LICENSE` file).

---

> 💡 **Note**: This model is designed for **software engineering stand-up meetings**. Performance may vary in other domains.

---

### ✅ Next Steps

1. Save this as `README.md` in your **project root** (`se_project/`)
2. Replace `your-username` in the clone URL with your actual GitHub username
3. Commit and push:
   ```bash
   git add README.md
   git commit -m "docs: add comprehensive README"
   git push
   ```
