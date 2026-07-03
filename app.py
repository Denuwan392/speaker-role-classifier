# app.py (upgraded)
import streamlit as st
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Force reload of predict_role to prevent Streamlit caching old module codes
import importlib
import predict_role
importlib.reload(predict_role)
from predict_role import predict_role

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="🗣️ Role Classifier – JSON Input", layout="wide")
st.title("🗣️ Speaker Role Classifier (Manager / HR / Junior / Other)")
st.caption("Paste a JSON array of diarized segments below (or upload a file). Each object must have `speaker_id` and `text`.")

# -----------------------
# Session state helpers
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []  # each entry: dict with timestamp, input, results, elapsed

if "staged_segments" not in st.session_state:
    st.session_state.staged_segments = None

# -----------------------
# Export / history actions preparation
# -----------------------
csv_data = None
json_data = None
if st.session_state.history:
    rows = []
    for run in st.session_state.history:
        for sid, info in run["result"].items():
            probs = info.get("probs", {})
            details = info.get("prediction_details", {})
            row = {
                "timestamp": run["timestamp"],
                "speaker_id": sid,
                "role": info.get("role"),
                "confidence": max(probs.values()) if probs else info.get("probability"),
                "evidence": (info.get("evidence") or [""])[0],
                "source": details.get("source", "N/A"),
                "llm_used": details.get("llm_used", False),
                "router_threshold": details.get("router_threshold", "N/A"),
                "xgb_confidence": details.get("xgb_confidence", "N/A"),
                "llm_confidence": details.get("llm_confidence", "N/A"),
                "fallback_reason": details.get("fallback_reason", "N/A"),
                "model_name": details.get("model_name", "N/A"),
                "feature_count": details.get("feature_count", "N/A"),
                "inference_time_ms": details.get("inference_time_ms", "N/A")
            }
            # Add class probabilities directly
            for label, p_val in probs.items():
                row[f"p_{label}"] = p_val
            rows.append(row)

    df_hist = pd.DataFrame(rows)
    csv_data = df_hist.to_csv(index=False).encode("utf-8")
    json_data = json.dumps(st.session_state.history, indent=2)

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Controls")
    use_llm = st.checkbox("Use cached LLM scores (no live calls)", value=False, help="If your predictor uses cached LLM scores, toggle this off to ignore them at inference (faster).")
    st.markdown("---")
    st.subheader("Batch / Demo")
    run_demo_suite = st.button("Run demo suite (10 tests)")
    if st.button("Clear run history"):
        st.session_state.history = []
        st.success("Run history cleared.")
        st.rerun()

    st.markdown("---")
    st.subheader("Export")
    if csv_data is not None:
        st.download_button("⬇️ Download history CSV", csv_data, file_name="role_predictions_history.csv", mime="text/csv")
    else:
        st.button("Download history CSV (Empty)", disabled=True, key="btn_csv_empty")

    if json_data is not None:
        st.download_button("⬇️ Download history JSON", json_data, file_name="role_predictions_history.json", mime="application/json")
    else:
        st.button("Download history JSON (Empty)", disabled=True, key="btn_json_empty")
    
    st.markdown("---")
    st.caption("Tips: Use the demo suite to exercise edge cases quickly.")

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

# -----------------------
# Re-run Last button handler
# -----------------------
re_run_clicked = False

with col_right:
    st.subheader("🔎 Quick actions")
    st.info("You can: paste JSON, upload a file, or run the demo suite in the sidebar.")
    st.write("History: ", len(st.session_state.history), "runs")
    st.markdown("---")
    if st.session_state.history:
        last = st.session_state.history[-1]
        st.write("Last run:", last["timestamp"])
        if st.button("Re-run last", key="re_run_last"):
            re_run_clicked = True

# -----------------------
# Utility functions
# -----------------------
def load_segments_from_upload(uploaded):
    if uploaded is None:
        return None, "No file uploaded"
    try:
        b = uploaded.read()
        filename = uploaded.name.lower()
        if filename.endswith(".json"):
            try:
                obj = json.loads(b.decode("utf-8"))
                if isinstance(obj, dict):
                    return list(obj.values()), None
                return obj, None
            except Exception as e:
                return None, f"Malformed JSON file: {e}"
        elif filename.endswith(".csv"):
            try:
                uploaded.seek(0)
                df = pd.read_csv(uploaded)
                if "speaker_id" not in df.columns or "text" not in df.columns:
                    return None, "CSV must contain 'speaker_id' and 'text' columns"
                segments = df[["speaker_id", "text"]].to_dict("records")
                return segments, None
            except Exception as e:
                return None, f"Malformed CSV file: {e}"
        else:
            return None, "Unsupported file extension (JSON or CSV required)"
    except Exception as e:
        return None, f"Failed to read file: {e}"

def validate_segments(segments):
    valid = []
    if not isinstance(segments, list):
        return None, "Top-level JSON must be an array of segment objects"
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            return None, f"Item {i} is not a valid JSON object"
        if "speaker_id" not in seg or "text" not in seg:
            return None, f"Item {i} is missing required fields 'speaker_id' or 'text'"
        
        sid = seg["speaker_id"]
        txt = seg["text"]
        
        if sid is None or txt is None:
            return None, f"Item {i} contains null values in speaker_id or text"
            
        sid_str = str(sid).strip()
        txt_str = str(txt).strip()
        
        if sid_str.lower() in ["nan", "null"] or txt_str.lower() in ["nan", "null"]:
            return None, f"Item {i} contains invalid/NaN values in speaker_id or text"
            
        if txt_str == "":
            continue
            
        valid.append({"speaker_id": sid_str, "text": txt_str})
        
    if not valid:
        return None, "No valid transcript segments found (non-empty text required)"
    return valid, None

def pretty_results_to_df(result):
    rows = []
    for spk, info in result.items():
        probs = info.get("probs", {})
        rows.append({
            "speaker_id": spk,
            "role": info.get("role"),
            "probability": max(probs.values()) if probs else info.get("probability"),
            "p_manager": probs.get("manager", 0.0),
            "p_hr": probs.get("hr", 0.0),
            "p_junior": probs.get("junior", 0.0),
            "p_other": probs.get("other", 0.0),
            "evidence": (info.get("evidence") or [""])[0]
        })
    return pd.DataFrame(rows)

# -----------------------
# Demo suite
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
        res = predict_role(segments, use_llm=use_llm)
        suite_results.append((name, res))
    return suite_results

# -----------------------
# Handle run actions
# -----------------------
def run_predict_and_record(segments):
    ts = datetime.utcnow().isoformat()
    start = time.time()
    try:
        res = predict_role(segments, use_llm=use_llm)
        elapsed = time.time() - start
        
        st.session_state.history.append({
            "timestamp": ts,
            "input": segments,
            "result": res,
            "elapsed": elapsed
        })
        
        # Limit history to 20 items to prevent memory issues
        if len(st.session_state.history) > 20:
            st.session_state.history = st.session_state.history[-20:]
            
        return res, None
    except Exception as e:
        return None, str(e)

# -----------------------
# File Upload Staging
# -----------------------
if uploaded_file is not None:
    uploaded_segments, err = load_segments_from_upload(uploaded_file)
    if err:
        st.error(err)
        st.session_state.staged_segments = None
    else:
        st.session_state.staged_segments = uploaded_segments
        st.info(f"📁 File '{uploaded_file.name}' loaded successfully. Click '🎯 Predict Roles' to run.")
else:
    st.session_state.staged_segments = None

# -----------------------
# Execution Block
# -----------------------
if run_button or run_demo_suite or re_run_clicked:
    segments = None
    
    if run_demo_suite:
        st.info("Running demo suite...")
        suite = run_demo_suite_fn()
        for name, result in suite:
            st.markdown(f"### 🔁 Demo: {name}")
            df = pretty_results_to_df(result)
            st.dataframe(df)
            
            # Display diagnostics for each demo case
            with st.expander("🔍 Demo Prediction Diagnostics"):
                for spk, info in result.items():
                    details = info.get("prediction_details", {})
                    st.markdown(f"**Speaker: {spk}**")
                    st.write(f"- Source: {details.get('source', 'N/A')}")
                    st.write(f"- XGBoost Conf: {details.get('xgb_confidence', 0.0):.2%}")
                    st.write(f"- LLM Used: {details.get('llm_used', False)}")
                    st.write(f"- Inference Time: {details.get('inference_time_ms', 0.0):.2f} ms")
                    st.markdown("---")
        st.success("Demo suite finished.")
    else:
        if re_run_clicked:
            if st.session_state.history:
                segments = st.session_state.history[-1]["input"]
            else:
                st.warning("No run history to re-run.")
        elif st.session_state.staged_segments is not None:
            segments = st.session_state.staged_segments
        else:
            try:
                segments = json.loads(user_input)
            except Exception as e:
                st.error("Invalid JSON paste: " + str(e))
                segments = None

        if segments is not None:
            valid_segments, v_err = validate_segments(segments)
            if v_err:
                st.error(v_err)
            else:
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
                    prob_cols = [c for c in df_res.columns if c.startswith("p_")]
                    if prob_cols:
                        prob_df = df_res.set_index("speaker_id")[prob_cols]
                        st.bar_chart(prob_df)
                    else:
                        prob_df = df_res[["speaker_id", "probability"]].set_index("speaker_id")
                        st.bar_chart(prob_df)

                    # Show evidence and diagnostics
                    st.markdown("### Evidence & details")
                    for _, row in df_res.iterrows():
                        emoji = {
                            "manager": "🟢",
                            "hr": "🟣",
                            "junior": "🟡",
                            "other": "🔵"
                        }.get(row["role"], "⚪")

                        st.markdown(f"**{emoji} {row['speaker_id']} — {row['role'].upper()} ({row['probability']:.1%})**")
                        st.code(row["evidence"], language="text")
                        
                        # Expandable section for diagnostics
                        with st.expander("🔍 Prediction Diagnostics"):
                            spk_res = result.get(row["speaker_id"], {})
                            details = spk_res.get("prediction_details", {})
                            
                            st.markdown("### Prediction Source")
                            st.markdown("-------------------------")
                            st.write(f"**Source**            : {details.get('source', 'N/A')}")
                            st.write(f"**Model**             : {details.get('model_name', 'N/A')}")
                            
                            llm_used_str = "Yes" if details.get('llm_used', False) else "No"
                            st.write(f"**LLM Used**          : {llm_used_str}")
                            st.write("")
                            
                            if details.get('llm_used', False):
                                st.write(f"**Reason**            : {details.get('fallback_reason', 'N/A')}")
                                st.write(f"**XGBoost Confidence**: {details.get('xgb_confidence', 0.0):.2%}")
                                
                                llm_conf = details.get('llm_confidence', 'N/A')
                                if isinstance(llm_conf, float):
                                    st.write(f"**LLM Confidence**    : {llm_conf:.2%}")
                                else:
                                    st.write(f"**LLM Confidence**    : {llm_conf}")
                            else:
                                st.write(f"**Confidence**        : {details.get('xgb_confidence', 0.0):.2%}")
                                st.write(f"**Threshold**         : {details.get('router_threshold', 0.0):.2%}")
                                
                            st.write("")
                            st.write(f"**Inference Time**    : {details.get('inference_time_ms', 0.0):.2f} ms")
                            st.write(f"**Feature Count**     : {details.get('feature_count', 'N/A')}")

                        st.markdown("---")

st.markdown("---")
st.caption("Built by you — iterate on patterns and add rules or retrain as you see failure modes.")
