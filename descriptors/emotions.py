import base64
from collections import defaultdict
from PIL import Image
import cv2
from deepface import DeepFace
from transformers import pipeline
import torch
from apiflask import APIBlueprint
from flask import request, jsonify

emotions = APIBlueprint('emotions', __name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_text_classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
emotion_face_classifier = pipeline("image-classification", model="trpakov/vit-face-expression")


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
def detect_faces_and_get_emotion_with_plots():
    data = request.form.get('data', '')
    if not data or "base64," not in data:
        return jsonify({"error": "No valid data URL provided"}), 400

    _, encoded = data.split("base64,", 1)
    image_path = "emotion.jpg"

    try:
        with open(image_path, "wb") as fh:
            fh.write(base64.b64decode(encoded))
    except Exception as e:
        print(f"[ERROR] decoding/saving image: {e}")
        return jsonify({"error": "Could not decode image"}), 400

    try:
        faces = DeepFace.extract_faces(image_path, detector_backend="retinaface", enforce_detection=True)
    except Exception as e:
        print(f"[ERROR] DeepFace failed on {image_path}: {e}")
        return jsonify({'error': str(e)}), 500

    if not faces:
        return jsonify([])

    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "Failed to read saved image"}), 500
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[ERROR] cvtColor failed: {e}")
        return jsonify({"error": "Color conversion failed"}), 500

    emotion_sum = defaultdict(float)
    emotion_count = defaultdict(int)
    face_count = 0

    for face_data in faces:
        x1 = face_data["facial_area"]["x"]
        y1 = face_data["facial_area"]["y"]
        w  = face_data["facial_area"]["w"]
        h  = face_data["facial_area"]["h"]
        x2, y2 = x1 + w, y1 + h

        x1c = max(0, x1); y1c = max(0, y1)
        x2c = min(img_rgb.shape[1], x2); y2c = min(img_rgb.shape[0], y2)
        face = img_rgb[y1c:y2c, x1c:x2c]
        face_pil = Image.fromarray(face)
        emotions, confidences = get_emotion_for_image(face_pil, top_k=None)
        for label, score in zip(emotions, confidences):
            emotion_sum[label] += score
            emotion_count[label] += 1

        face_count += 1

    if face_count == 0 or not emotion_sum:
        return jsonify([])

    emotion_avg = {lbl: (emotion_sum[lbl] / max(1, emotion_count[lbl])) for lbl in emotion_sum}
    dominant = max(emotion_avg.items(), key=lambda kv: kv[1])[0]
    confidence = emotion_avg[dominant]

    return jsonify({
        "faces": face_count,
        "dominant_emotion": dominant,
        "dominant_confidence": confidence,
        "per_emotion_avg": emotion_avg,
        "per_emotion_sum": dict(emotion_sum),
    })

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