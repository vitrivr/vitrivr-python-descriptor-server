from __future__ import annotations

import os
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from apiflask import APIBlueprint
from flask import current_app, jsonify, request
from PIL import Image

from descriptors.adaface import (
    Box,
    _boolean_parameter,
    _decode_image,
    _get_service as _get_face_service,
    _normalize,
)


yolo_adaface_forward = APIBlueprint("yolo_adaface_forward", __name__)


def _clamp_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


class OpenVINOBodyReID:
    def __init__(self, model_path: Path, device: str = "AUTO") -> None:
        try:
            import openvino as ov
        except ImportError as error:
            raise RuntimeError(
                "OpenVINO is not installed; install the 'openvino' package."
            ) from error

        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"OpenVINO ReID model not found: {model_path}")

        core = ov.Core()
        self.compiled = core.compile_model(str(model_path), device)
        self.input_layer = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)
        # CompiledModel.__call__ reuses one internal InferRequest and therefore
        # cannot serve concurrent Flask requests. Use the device's recommended
        # number of independent requests so forward and two-pass inference can
        # run concurrently without creating an unbounded request per HTTP thread.
        try:
            request_count = max(
                1,
                int(self.compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")),
            )
        except RuntimeError:
            request_count = 1
        self.infer_requests: "queue.Queue[object]" = queue.Queue(request_count)
        for _ in range(request_count):
            self.infer_requests.put(self.compiled.create_infer_request())

    def embed(self, person_rgb: np.ndarray) -> Optional[np.ndarray]:
        if person_rgb.size == 0:
            return None
        height, width = person_rgb.shape[:2]
        if height < 50 or width < 20:
            return None

        # person-reidentification-retail-0286 expects BGR NCHW input at 256x128.
        import cv2

        person_bgr = person_rgb[:, :, ::-1]
        resized = cv2.resize(person_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
        tensor = np.transpose(resized.astype(np.float32), (2, 0, 1))[None, ...]
        infer_request = self.infer_requests.get()
        try:
            result = infer_request.infer({self.input_layer: tensor})
            embedding = np.asarray(result[self.output_layer]).reshape(-1).copy()
        finally:
            self.infer_requests.put(infer_request)
        return _normalize(embedding)


class NamedBodyGallery:
    """Body observations learned only from earlier face-confirmed frames."""

    def __init__(
        self,
        threshold: float,
        margin: float,
        max_embeddings_per_person: int,
    ) -> None:
        self.threshold = threshold
        self.margin = margin
        self.max_embeddings_per_person = max_embeddings_per_person
        self.people: Dict[str, List[np.ndarray]] = {}

    def add(self, name: str, embedding: Optional[np.ndarray]) -> None:
        if embedding is None:
            return
        embedding = _normalize(embedding)
        gallery = self.people.setdefault(name, [])
        if gallery and max(float(np.dot(embedding, old)) for old in gallery) > 0.985:
            return
        gallery.append(embedding)
        del gallery[: max(0, len(gallery) - self.max_embeddings_per_person)]

    def recognize(self, embedding: Optional[np.ndarray]) -> Tuple[Optional[str], Optional[float]]:
        if embedding is None or not self.people:
            return None, None

        query = _normalize(embedding)
        ranked = []
        for name, gallery in self.people.items():
            scores = sorted(
                (float(np.dot(query, known)) for known in gallery), reverse=True
            )
            ranked.append((float(np.mean(scores[:3])), name))

        ranked.sort(reverse=True)
        best_score, best_name = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else -1.0
        if best_score < self.threshold:
            return None, best_score
        if len(ranked) > 1 and best_score - second_score < self.margin:
            return None, best_score
        return best_name, best_score


class StreamState:
    def __init__(self, gallery: NamedBodyGallery) -> None:
        self.gallery = gallery
        self.last_access = time.monotonic()
        self.lock = threading.Lock()


class ForwardService:
    def __init__(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is not installed; install the 'ultralytics' package."
            ) from error

        self.yolo_model_name = os.getenv("YOLO_MODEL", "yolo26n.pt")
        self.detector = YOLO(self.yolo_model_name)
        self.detector_lock = threading.Lock()

        reid_path = os.getenv("BODY_REID_MODEL")
        self.body_reid = (
            OpenVINOBodyReID(
                Path(reid_path), os.getenv("OPENVINO_DEVICE", "AUTO")
            )
            if reid_path
            else None
        )
        self.body_threshold = float(os.getenv("BODY_REID_THRESHOLD", "0.55"))
        self.body_margin = float(os.getenv("BODY_REID_MARGIN", "0.08"))
        self.max_body_embeddings = int(
            os.getenv("BODY_REID_MAX_EMBEDDINGS", "50")
        )
        self.max_streams = int(os.getenv("FORWARD_MAX_STREAMS", "32"))
        self.stream_ttl = float(os.getenv("FORWARD_STREAM_TTL_SECONDS", "3600"))
        self.streams: "OrderedDict[str, StreamState]" = OrderedDict()
        self.streams_lock = threading.Lock()

    def detect_people(
        self, image: Image.Image, confidence: float, max_persons: int
    ) -> List[Tuple[Box, float]]:
        with self.detector_lock:
            result = self.detector.predict(
                image,
                classes=[0],
                conf=confidence,
                max_det=max_persons,
                verbose=False,
            )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        return [
            (
                _clamp_box(
                    tuple(int(round(value)) for value in raw_box),
                    image.width,
                    image.height,
                ),
                float(score),
            )
            for raw_box, score in zip(xyxy, scores)
        ]

    def stream(self, stream_id: str) -> StreamState:
        now = time.monotonic()
        with self.streams_lock:
            expired = [
                key
                for key, state in self.streams.items()
                if now - state.last_access > self.stream_ttl
            ]
            for key in expired:
                del self.streams[key]

            state = self.streams.pop(stream_id, None)
            if state is None:
                state = StreamState(
                    NamedBodyGallery(
                        self.body_threshold,
                        self.body_margin,
                        self.max_body_embeddings,
                    )
                )
            state.last_access = now
            self.streams[stream_id] = state
            while len(self.streams) > self.max_streams:
                self.streams.popitem(last=False)
            return state

    def reset(self, stream_id: str) -> bool:
        with self.streams_lock:
            return self.streams.pop(stream_id, None) is not None


_service: Optional[ForwardService] = None
_service_lock = threading.Lock()


def _get_service() -> ForwardService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ForwardService()
    return _service


def _face_for_person(
    person_box: Box,
    faces: List[Tuple[np.ndarray, Box]],
) -> Optional[Tuple[np.ndarray, Box]]:
    px1, py1, px2, py2 = person_box
    candidates = []
    for embedding, box in faces:
        fx1, fy1, fx2, fy2 = box
        center_x = (fx1 + fx2) / 2
        center_y = (fy1 + fy2) / 2
        if px1 <= center_x <= px2 and py1 <= center_y <= py2:
            candidates.append(((fx2 - fx1) * (fy2 - fy1), embedding, box))
    if not candidates:
        return None
    _, embedding, box = max(candidates, key=lambda item: item[0])
    return embedding, box


@yolo_adaface_forward.post("/extract/yolo_adaface_forward")
@yolo_adaface_forward.doc(
    summary="Detect people and propagate AdaFace identities forward within a video stream"
)
def extract_yolo_adaface_forward():
    try:
        image = _decode_image(request.form.get("data", ""))
        stream_id = request.form.get("stream_id", "").strip()
        confidence = float(request.form.get("person_confidence", "0.25"))
        face_threshold = float(request.form.get("face_threshold", "0.25"))
        max_persons = int(request.form.get("max_persons", "20"))
        include_face_embedding = _boolean_parameter("include_face_embedding", True)
        include_body_embedding = _boolean_parameter("include_body_embedding", False)
    except ValueError as error:
        return jsonify(error=str(error)), 400

    if not stream_id or len(stream_id) > 200:
        return jsonify(error="'stream_id' is required and must be at most 200 characters."), 400
    if not 0.0 <= confidence <= 1.0 or not -1.0 <= face_threshold <= 1.0:
        return jsonify(error="Confidence values are outside their valid range."), 400
    if not 1 <= max_persons <= 100:
        return jsonify(error="'max_persons' must be between 1 and 100."), 400

    try:
        service = _get_service()
        face_service = _get_face_service()
    except (FileNotFoundError, RuntimeError) as error:
        return jsonify(error=str(error)), 503

    try:
        people = service.detect_people(image, confidence, max_persons)
        faces = face_service.recognizer.detect_and_embed(image, max_persons)
    except Exception as error:
        current_app.logger.exception("YOLO + AdaFace extraction failed")
        return jsonify(error=f"YOLO + AdaFace extraction failed: {error}"), 500

    state = service.stream(stream_id)
    rgb = np.asarray(image)
    response_people = []
    with state.lock:
        for person_box, detection_score in people:
            x1, y1, x2, y2 = person_box
            body_embedding = (
                service.body_reid.embed(rgb[y1:y2, x1:x2])
                if service.body_reid is not None
                else None
            )
            face = _face_for_person(person_box, faces)
            name = None
            source = None
            face_score = None
            body_score = None
            face_payload = None

            if face is not None:
                face_embedding, face_box = face
                name, face_score = face_service.gallery.recognize(
                    face_embedding, face_threshold
                )
                if name is not None:
                    source = "face"
                    state.gallery.add(name, body_embedding)
                face_payload = {"box": list(face_box)}
                if include_face_embedding:
                    face_payload["embedding"] = face_embedding.tolist()

            if name is None and service.body_reid is not None:
                name, body_score = state.gallery.recognize(body_embedding)
                if name is not None:
                    source = "body"

            person_payload = {
                "box": list(person_box),
                "detection_score": detection_score,
                "person": name,
                "source": source,
                "face_score": face_score,
                "body_score": body_score,
                "face": face_payload,
            }
            if include_body_embedding and body_embedding is not None:
                person_payload["body_embedding"] = body_embedding.tolist()
            response_people.append(person_payload)

    return jsonify(
        stream_id=stream_id,
        image={"width": image.width, "height": image.height},
        persons=response_people,
        gallery_configured=bool(face_service.gallery.people),
        body_reid_configured=service.body_reid is not None,
    )


@yolo_adaface_forward.post("/extract/yolo_adaface_forward/reset")
@yolo_adaface_forward.doc(summary="Discard forward-only body identity state for one stream")
def reset_yolo_adaface_forward():
    stream_id = request.form.get("stream_id", "").strip()
    if not stream_id:
        return jsonify(error="'stream_id' is required."), 400
    try:
        removed = _get_service().reset(stream_id)
    except (FileNotFoundError, RuntimeError) as error:
        return jsonify(error=str(error)), 503
    return jsonify(stream_id=stream_id, reset=removed)
