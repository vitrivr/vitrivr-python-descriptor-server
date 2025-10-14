import base64
from transformers import pipeline
import torch
from apiflask import APIBlueprint
from flask import request, jsonify

emotions = APIBlueprint('emotions', __name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_text_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")


@emotions.post("/extract/emotions_text")
@emotions.doc(summary="Emotions endpoint that extract emotions from text")
def extract_emotions_from_text():
    data = request.form.get("data", "")
    print(f"data: {data}")
    header, encoded = data.split("utf-8,", 1)
    try:
        text = base64.b64decode(encoded).decode("utf-8")
        print(text)
    except Exception as e:
        print(f"[ERROR] Could not base64 encode text: {e}")
        return "[]"

    try:
        result = emotion_text_classifier(text, top_k=None, truncation=True)
        scores = result if isinstance(result, list) and isinstance(result[0], dict) else result[0]
        scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)
        return jsonify(scores_sorted)

    except Exception as e:
        print(f"[ERROR] classification failed: {e}")
        return jsonify([])