# app.py (upgraded)
import streamlit as st
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# import the predictor (your file)
from predict_role import predict_role

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="🗣️ Role Classifier – JSON Input", layout="wide")
st.title("🗣️ Speaker Role Classifier (Manager / Junior / Other)")
st.caption("Paste a JSON array of diarized segments below (or upload a file). Each object must have `speaker_id` and `text`.")

# -----------------------
# Session state helpers
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []  # each entry: dict with timestamp, input, results

if "last_input" not in st.session_state:
    st.session_state.last_input = None

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Controls")
    use_llm = st.checkbox("Use cached LLM scores (no live calls)", value=False, help="If your predictor uses cached LLM scores, toggle this off to ignore them at inference (faster).")
    st.markdown("---")
    st.subheader("Batch / Demo")
    run_demo_suite = st.button("Run demo suite (10 tests)")
    clear_history = st.button("Clear run history")
    st.markdown("---")
    st.subheader("Export")
    export_history_csv = st.button("Download history CSV")
    export_history_json = st.button("Download history JSON")
    st.markdown("---")
    st.caption("Tips: Use the demo suite to exercise edge cases quickly.")

if clear_history:
    st.session_state.history = []
    st.success("Run history cleared.")

# -----------------------
# Main layout
# -----------------------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📌 Input (JSON)")
    default_json = """[
  {"speaker_id": "spk_1", "text": "Morning team. What did you do yesterday?"},
  {"speaker_id": "spk_1", "text": "Please check the API docs."},
  {"speaker_id": "spk_2", "text": "I'm stuck on the login bug..."},
  {"speaker_id": "spk_3", "text": "I reviewed the PR and left comments."}
]"""
    user_input = st.text_area("Paste diarized segments (JSON array):", value=default_json, height=240)

    st.markdown("**Or upload a file** (JSON or CSV with `speaker_id`, `text` columns).")
    uploaded_file = st.file_uploader("Upload JSON / CSV", type=["json", "csv"])

    run_button = st.button("🎯 Predict Roles")

with col_right:
    st.subheader("🔎 Quick actions")
    st.info("You can: paste JSON, upload a file, or run the demo suite in the sidebar.")
    st.write("History: ", len(st.session_state.history), "runs")
    st.markdown("---")
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.write("Last run:", last["timestamp"])
        st.button("Re-run last", key="re_run_last")
        if st.session_state.get("re_run_last_clicked", False):
            pass

# -----------------------
# Utility functions
# -----------------------
def load_segments_from_upload(uploaded):
    if uploaded is None:
        return None, "No file uploaded"
    try:
        b = uploaded.read()
        # try JSON first
        try:
            obj = json.loads(b.decode("utf-8"))
            # obj should be a list of dicts (or dict mapping)
            if isinstance(obj, dict):
                # maybe newline-delimited JSON? take values
                return list(obj.values()), None
            return obj, None
        except Exception:
            # try CSV
            df = pd.read_csv(uploaded)
            if "speaker_id" not in df.columns or "text" not in df.columns:
                return None, "CSV must contain 'speaker_id' and 'text' columns"
            segments = df[["speaker_id", "text"]].to_dict("records")
            return segments, None
    except Exception as e:
        return None, f"Failed to read file: {e}"

def validate_segments(segments):
    valid = []
    if not isinstance(segments, list):
        return None, "Top-level JSON must be an array of segment objects"
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            return None, f"Item {i} is not an object"
        if "speaker_id" not in seg or "text" not in seg:
            return None, f"Item {i} missing 'speaker_id' or 'text'"
        sid = str(seg["speaker_id"])
        txt = str(seg["text"]).strip()
        if txt == "":
            continue
        valid.append({"speaker_id": sid, "text": txt})
    if not valid:
        return None, "No valid segments (non-empty text required)"
    return valid, None

def pretty_results_to_df(result):
    # result: {spk: {"role": .., "probability":.., "evidence":[..]}}
    rows = []
    for spk, info in result.items():
        rows.append({
            "speaker_id": spk,
            "role": info.get("role"),
            "probability": info.get("probability"),
            "evidence": (info.get("evidence") or [""])[0]
        })
    return pd.DataFrame(rows)

# -----------------------
# Demo suite (quick batch tests)
# -----------------------
demo_cases = [
    ("Strong manager directive", [
        {"speaker_id": "spk_1", "text": "Team, please escalate blockers early. You need to update the release branch before EOD."},
        {"speaker_id": "spk_2", "text": "I completed the schema migration yesterday. No blockers."},
        {"speaker_id": "spk_3", "text": "Still trying to debug the 500s. Might need help later."},
    ]),
    ("Clear junior", [
        {"speaker_id": "spk_1", "text": "I'm not sure how to resolve this dependency issue. Maybe I'm missing something."},
        {"speaker_id": "spk_2", "text": "Yesterday I implemented the caching layer; today I’ll write tests."},
    ]),
    ("Only greetings", [
        {"speaker_id": "spk_1", "text": "Good morning team."},
        {"speaker_id": "spk_2", "text": "Morning!"},
        {"speaker_id": "spk_3", "text": "Hello everyone."}
    ]),
    ("Multiple directives", [
        {"speaker_id": "spk_1", "text": "Please review the PR today."},
        {"speaker_id": "spk_1", "text": "Make sure you run the migration before deploying."},
        {"speaker_id": "spk_2", "text": "Got it. I'll fix the failing tests."},
    ]),
    ("Senior uncertainty", [
        {"speaker_id": "spk_1", "text": "Not sure if the Grafana alert is correct — metrics look weird today, investigating."},
        {"speaker_id": "spk_2", "text": "Working on the infra pipeline improvements."}
    ]),
]

def run_demo_suite_fn():
    suite_results = []
    for name, segments in demo_cases:
        res = predict_role(segments)
        df = pretty_results_to_df(res)
        suite_results.append((name, df))
    return suite_results

# -----------------------
# Handle run actions
# -----------------------
def run_predict_and_record(segments):
    # call predictor
    ts = datetime.utcnow().isoformat()
    start = time.time()
    try:
        res = predict_role(segments)
        elapsed = time.time() - start
        # record
        st.session_state.history.append({
            "timestamp": ts,
            "input": segments,
            "result": res,
            "elapsed": elapsed
        })
        st.session_state.last_input = segments
        return res, None
    except Exception as e:
        return None, str(e)

# If file uploaded, prefer it over pasted JSON
uploaded_segments = None
if uploaded_file is not None:
    uploaded_segments, err = load_segments_from_upload(uploaded_file)
    if err:
        st.error(err)
        uploaded_segments = None

# Trigger predict
if run_button or uploaded_segments is not None or run_demo_suite:
    segments = None
    if uploaded_segments is not None:
        segments = uploaded_segments
    else:
        # parse pasted text
        try:
            segments = json.loads(user_input)
        except Exception as e:
            st.error("Invalid JSON paste: " + str(e))
            segments = None

    if run_demo_suite:
        st.info("Running demo suite...")
        suite = run_demo_suite_fn()
        for name, df in suite:
            st.markdown(f"### 🔁 Demo: {name}")
            st.dataframe(df)
        st.success("Demo suite finished.")
    else:
        # validate
        valid_segments, v_err = validate_segments(segments) if segments is not None else (None, "No input")
        if v_err:
            st.error(v_err)
        else:
            # run prediction
            with st.spinner("🧠 Running prediction..."):
                result, err = run_predict_and_record(valid_segments)
            if err:
                st.exception("Prediction failed: " + err)
            else:
                st.success("Prediction complete")
                df_res = pretty_results_to_df(result)
                st.dataframe(df_res)

                # Display per-speaker probability bars
                st.markdown("### Probability per speaker")
                prob_df = df_res[["speaker_id","probability"]].set_index("speaker_id")
                st.bar_chart(prob_df)

                # Show evidence and allow download
                st.markdown("### Evidence & details")
                for _, row in df_res.iterrows():
                    emoji = {"manager":"🟢", "junior":"🟡", "other":"🔵"}.get(row["role"], "⚪")
                    st.markdown(f"**{emoji} {row['speaker_id']} — {row['role'].upper()} ({row['probability']:.1%})**")
                    st.code(row["evidence"], language="text")
                    st.markdown("---")

# -----------------------
# Export / history actions
# -----------------------
if export_history_csv and st.session_state.history:
    # flatten to DataFrame
    rows = []
    for run in st.session_state.history:
        for sid, info in run["result"].items():
            rows.append({
                "timestamp": run["timestamp"],
                "speaker_id": sid,
                "role": info.get("role"),
                "prob": info.get("probability"),
                "evidence": (info.get("evidence") or [""])[0]
            })
    df_hist = pd.DataFrame(rows)
    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download history CSV", csv, file_name="role_predictions_history.csv", mime="text/csv")

if export_history_json and st.session_state.history:
    j = json.dumps(st.session_state.history, indent=2)
    st.download_button("⬇️ Download history JSON", j, file_name="role_predictions_history.json", mime="application/json")

# small footer
st.markdown("---")
st.caption("Built by you — iterate on patterns and add rules or retrain as you see failure modes.")
