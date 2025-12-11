# demo.py
from predict_role import predict_role
import json

segments = [
    {"speaker_id": "spk_1", "text": "Morning everyone. Let’s keep it quick—what did you do yesterday?"},
    {"speaker_id": "spk_1", "text": "Sure, check the /docs/api folder. And feel better!"},
    {"speaker_id": "spk_2", "text": "Yesterday I finished the auth middleware. Today I’ll start rate-limiting."},
    {"speaker_id": "spk_3", "text": "I was out sick yesterday. I’m not sure where the doc files live—can someone point me?"},
]

result = predict_role(segments)
print(json.dumps(result, indent=2))