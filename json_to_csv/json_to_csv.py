import json
import pandas as pd
from pathlib import Path

def json_to_speaker_csv(json_path: str, csv_path: str):
    """
    Converts LLM-generated meeting JSON → speaker-level CSV for training.
    
    Input JSON: list of meetings, each with turn-level segments.
    Output CSV: one row per speaker per meeting (aggregated text).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        meetings = json.load(f)

    rows = []
    for meeting in meetings:
        meeting_id = meeting["meeting_id"]
        # Group segments by speaker_id
        speaker_turns = {}
        speaker_role = {}
        for seg in meeting["segments"]:
            spk = seg["speaker_id"]
            if spk not in speaker_turns:
                speaker_turns[spk] = []
                speaker_role[spk] = seg["role"]  # assume consistent label per speaker
            speaker_turns[spk].append(seg["text"])

        # Create one row per speaker
        for spk, texts in speaker_turns.items():
            full_text = " ".join(texts)
            rows.append({
                "meeting_id": meeting_id,
                "speaker_id": spk,
                "full_text": full_text,
                "role": speaker_role[spk]
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(df)} speaker samples to {csv_path}")

# Example usage
if __name__ == "__main__":
    json_to_speaker_csv("data/meetings.json", "data/labeled_roles.csv")