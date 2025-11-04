import base64
from collections import defaultdict
from io import BytesIO

from PIL import Image
import cv2
from deepface import DeepFace
from transformers import pipeline
import torch
from apiflask import APIBlueprint
from flask import request, jsonify
import numpy as np
import pytesseract
import tempfile

emotions = APIBlueprint('emotions', __name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_text_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
emotion_face_classifier = pipeline("image-classification", model="trpakov/vit-face-expression")

EMOTION_LABEL_ORDER = ["angry","disgust","fear","happy","neutral","sad","surprise"]

_CANON_MAP = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "joy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
}

def _safe_decode_data_url(data: str, marker: str):
    """
    Extracts payload from base64 encoded data
    :param data: base64 encoded data
    :param marker: substring that marks the start of the base64 encoded data
    :return: the raw bytes
    """
    if not data or marker not in data:
        raise ValueError("Invalid data URL")
    _, encoded = data.split(marker, 1)
    return base64.b64decode(encoded)

def _normalize_label(lbl: str) -> str:
    """
    Normalizes the labels and employs fixed label order.
    :param lbl: label as string
    :return: normalized label as string
    """
    if not isinstance(lbl, str):
        return ""
    return _CANON_MAP.get(lbl.strip().lower(), lbl.strip().lower())


def _vectorize_scores(list_of_label_score_dicts):
    """
    Vectorizes model outputs into a fixed label order.
    Accepts a list of dicts like {"label": str, "score": float}.
    Unknown/missing labels are ignored; missing entries are 0.0.
    """
    agg = defaultdict(float)

    for item in list_of_label_score_dicts or []:
        if not isinstance(item, dict):
            continue
        raw = item.get("label", "")
        label = _normalize_label(raw)
        if not label:
            continue
        try:
            agg[label] += float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue

    confidences = [float(agg.get(label, 0.0)) for label in EMOTION_LABEL_ORDER]
    return EMOTION_LABEL_ORDER, confidences


def get_emotion_for_image(image, top_k=None):
    preds = emotion_face_classifier(image, top_k=top_k)
    if isinstance(preds, dict):
        preds = [preds]
    labels = [p["label"] for p in preds]
    scores = [p["score"] for p in preds]
    return labels, scores


@emotions.post("/extract/emotions_face")
@emotions.doc(summary="Emotions endpoint that checks if face exists (with DeepFace) and then predicts emotion "
                      "for those faces")
def detect_faces_and_get_emotion():
    data = request.form.get('data', '')
    if not data or "base64," not in data:
        return jsonify({"error": "No valid data URL provided"}), 400

    try:
        raw = _safe_decode_data_url(data, "base64,")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()
            try:
                faces = DeepFace.extract_faces(img_path=tmp.name, detector_backend="retinaface", enforce_detection=True)
            except Exception as e:
                print(f"[ERROR] DeepFace found no face on {tmp.name}: {e}")
                return [0.0 for _ in range(len(EMOTION_LABEL_ORDER))]

            img_bgr = cv2.imread(tmp.name)
            if img_bgr is None:
                return jsonify({"error": "Failed to read temp image"}), 500
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            agg_scores = defaultdict(list)
            for face_data in faces:
                x1 = face_data["facial_area"]["x"]
                y1 = face_data["facial_area"]["y"]
                w  = face_data["facial_area"]["w"]
                h  = face_data["facial_area"]["h"]
                x2, y2 = x1 + w, y1 + h

                x1c = max(0, x1); y1c = max(0, y1)
                x2c = min(img_rgb.shape[1], x2); y2c = min(img_rgb.shape[0], y2)
                face_roi = img_rgb[y1c:y2c, x1c:x2c]
                if face_roi.size == 0:
                    continue

                preds = emotion_face_classifier(Image.fromarray(face_roi), top_k=None)
                if isinstance(preds, dict):
                    preds = [preds]
                for p in preds:
                    agg_scores[p["label"]].append(float(p["score"]))

        if not agg_scores:
            return jsonify([0.0 for _ in EMOTION_LABEL_ORDER]), 200

        averaged = [{"label": lbl, "score": float(np.mean(vals))} for lbl, vals in agg_scores.items()]
        labels, confidences = _vectorize_scores(averaged)
        return jsonify(confidences), 200

    except Exception as e:
        print(f"[ERROR] emotions_face failed: {e}")
        return jsonify({"error": f"emotions_face failed: {e}"}), 422

@emotions.post("/extract/emotions_ocr")
@emotions.doc(summary="Endpoint that extract ocr text from images and then classifies into emotions. Returns vectors. ")
def extract_text_and_emotions():
    data = request.form.get('data', '')
    if not data or "base64," not in data:
        return jsonify({"error": "No valid data URL provided"}), 400

    header, encoded = data.split("base64,", 1)
    image = Image.open(BytesIO(base64.b64decode(encoded)))

    # Perform OCR using Tesseract
    extracted_text = pytesseract.image_to_string(image)
    print(extracted_text)
    try:
        result = emotion_text_classifier(extracted_text, top_k=None, truncation=True)
        scores = result if isinstance(result, list) and isinstance(result[0], dict) else result[0]
        labels, confidences = _vectorize_scores(scores)
        return confidences

    except Exception as e:
        print(f"[ERROR] classification failed: {e}")
        return [0.0 for _ in range(len(EMOTION_LABEL_ORDER))]

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@emotions.post("/extract/emotions_text")
@emotions.doc(summary="Emotions endpoint that extract emotions from text")
def extract_emotions_from_text():
    data = request.form.get("data", "")
    print(f"data: {data}")
    try:
        raw = _safe_decode_data_url(data, "utf-8,")
        text = raw.decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Could not base64 encode text: {e}")
        return [0.0 for _ in range(len(EMOTION_LABEL_ORDER))]

    try:
        result = emotion_text_classifier(text, top_k=None, truncation=True)
        scores = result if isinstance(result, list) and isinstance(result[0], dict) else result[0]
        labels, confidences = _vectorize_scores(scores)
        return confidences

    except Exception as e:
        print(f"[ERROR] classification failed: {e}")
        return [0.0 for _ in range(len(EMOTION_LABEL_ORDER))]