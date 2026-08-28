from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from apiflask import APIBlueprint
from flask import current_app, jsonify, request

from descriptors.adaface import (
    Box,
    _boolean_parameter,
    _decode_image,
    _get_service as _get_face_service,
)
from descriptors.yolo_adaface_forward import (
    NamedBodyGallery,
    _face_for_person,
    _get_service as _get_forward_service,
)


yolo_adaface_two_pass = APIBlueprint("yolo_adaface_two_pass", __name__)


@dataclass
class StoredDetection:
    payload: Dict[str, object]
    body_embedding: Optional[np.ndarray]


@dataclass
class StoredFrame:
    frame_index: int
    timestamp_seconds: Optional[float]
    width: int
    height: int
    detections: List[StoredDetection]


class TwoPassStream:
    def __init__(self, gallery: NamedBodyGallery) -> None:
        self.gallery = gallery
        self.frames: Dict[int, StoredFrame] = {}
        self.last_access = time.monotonic()
        self.lock = threading.Lock()


class TwoPassStore:
    def __init__(self) -> None:
        forward = _get_forward_service()
        self.body_threshold = forward.body_threshold
        self.body_margin = forward.body_margin
        self.max_body_embeddings = forward.max_body_embeddings
        self.max_streams = forward.max_streams
        self.stream_ttl = forward.stream_ttl
        self.max_frames = 10000
        self.streams: "OrderedDict[str, TwoPassStream]" = OrderedDict()
        self.lock = threading.Lock()

    def stream(self, stream_id: str) -> TwoPassStream:
        now = time.monotonic()
        with self.lock:
            expired = [
                key
                for key, stream in self.streams.items()
                if now - stream.last_access > self.stream_ttl
            ]
            for key in expired:
                del self.streams[key]

            stream = self.streams.pop(stream_id, None)
            if stream is None:
                stream = TwoPassStream(
                    NamedBodyGallery(
                        self.body_threshold,
                        self.body_margin,
                        self.max_body_embeddings,
                    )
                )
            stream.last_access = now
            self.streams[stream_id] = stream
            while len(self.streams) > self.max_streams:
                self.streams.popitem(last=False)
            return stream

    def get(self, stream_id: str) -> Optional[TwoPassStream]:
        with self.lock:
            stream = self.streams.get(stream_id)
            if stream is not None:
                stream.last_access = time.monotonic()
            return stream

    def reset(self, stream_id: str) -> bool:
        with self.lock:
            return self.streams.pop(stream_id, None) is not None


_store: Optional[TwoPassStore] = None
_store_lock = threading.Lock()


def _get_store() -> TwoPassStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = TwoPassStore()
    return _store


def _validate_stream_id() -> str:
    stream_id = request.form.get("stream_id", "").strip()
    if not stream_id or len(stream_id) > 200:
        raise ValueError(
            "'stream_id' is required and must be at most 200 characters."
        )
    return stream_id


def _public_frame(frame: StoredFrame) -> Dict[str, object]:
    return {
        "frame_index": frame.frame_index,
        "timestamp_seconds": frame.timestamp_seconds,
        "image": {"width": frame.width, "height": frame.height},
        "persons": [detection.payload for detection in frame.detections],
    }


@yolo_adaface_two_pass.post("/extract/yolo_adaface_two_pass")
@yolo_adaface_two_pass.doc(
    summary="Add one video frame and return provisional two-pass person identities"
)
def extract_yolo_adaface_two_pass():
    try:
        image = _decode_image(request.form.get("data", ""))
        stream_id = _validate_stream_id()
        frame_index = int(request.form.get("frame_index", ""))
        timestamp_value = request.form.get("timestamp_seconds")
        timestamp_seconds = (
            float(timestamp_value) if timestamp_value not in (None, "") else None
        )
        confidence = float(request.form.get("person_confidence", "0.25"))
        face_threshold = float(request.form.get("face_threshold", "0.25"))
        max_persons = int(request.form.get("max_persons", "20"))
        include_face_embedding = _boolean_parameter("include_face_embedding", True)
        include_body_embedding = _boolean_parameter("include_body_embedding", False)
    except ValueError as error:
        return jsonify(error=str(error)), 400

    if frame_index < 0 or (timestamp_seconds is not None and timestamp_seconds < 0):
        return jsonify(error="Frame index and timestamp cannot be negative."), 400
    if not 0.0 <= confidence <= 1.0 or not -1.0 <= face_threshold <= 1.0:
        return jsonify(error="Confidence values are outside their valid range."), 400
    if not 1 <= max_persons <= 100:
        return jsonify(error="'max_persons' must be between 1 and 100."), 400

    try:
        forward = _get_forward_service()
        if forward.body_reid is None:
            return jsonify(
                error="BODY_REID_MODEL is required for the two-pass descriptor."
            ), 503
        face_service = _get_face_service()
        stream = _get_store().stream(stream_id)
    except (FileNotFoundError, RuntimeError) as error:
        return jsonify(error=str(error)), 503

    try:
        people = forward.detect_people(image, confidence, max_persons)
        faces = face_service.recognizer.detect_and_embed(image, max_persons)
    except Exception as error:
        current_app.logger.exception("Two-pass frame extraction failed")
        return jsonify(error=f"Two-pass frame extraction failed: {error}"), 500

    rgb = np.asarray(image)
    stored_detections: List[StoredDetection] = []
    with stream.lock:
        if frame_index in stream.frames:
            return jsonify(
                error=f"Frame {frame_index} already exists in stream '{stream_id}'."
            ), 409
        if len(stream.frames) >= _get_store().max_frames:
            return jsonify(error="The stream has reached its frame limit."), 413

        for person_box, detection_score in people:
            x1, y1, x2, y2 = person_box
            body_embedding = forward.body_reid.embed(rgb[y1:y2, x1:x2])
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
                    stream.gallery.add(name, body_embedding)
                face_payload: Dict[str, object] = {"box": list(face_box)}
                if include_face_embedding:
                    face_payload["embedding"] = face_embedding.tolist()

            if name is None:
                name, body_score = stream.gallery.recognize(body_embedding)
                if name is not None:
                    source = "body"

            payload: Dict[str, object] = {
                "box": list(person_box),
                "detection_score": detection_score,
                "person": name,
                "source": source,
                "face_score": face_score,
                "body_score": body_score,
                "face": face_payload,
            }
            if include_body_embedding and body_embedding is not None:
                payload["body_embedding"] = body_embedding.tolist()
            stored_detections.append(StoredDetection(payload, body_embedding))

        frame = StoredFrame(
            frame_index,
            timestamp_seconds,
            image.width,
            image.height,
            stored_detections,
        )
        stream.frames[frame_index] = frame

    return jsonify(
        stream_id=stream_id,
        provisional=True,
        frame=_public_frame(frame),
        stored_frame_count=len(stream.frames),
    )


@yolo_adaface_two_pass.post("/extract/yolo_adaface_two_pass/finalize")
@yolo_adaface_two_pass.doc(
    summary="Retroactively identify unknown observations using the completed body gallery"
)
def finalize_yolo_adaface_two_pass():
    try:
        stream_id = _validate_stream_id()
    except ValueError as error:
        return jsonify(error=str(error)), 400

    store = _get_store()
    stream = store.get(stream_id)
    if stream is None:
        return jsonify(error=f"Unknown stream: {stream_id}"), 404

    corrected = 0
    remaining_unknown = 0
    with stream.lock:
        for frame in stream.frames.values():
            for detection in frame.detections:
                if detection.payload["person"] is not None:
                    continue
                name, body_score = stream.gallery.recognize(
                    detection.body_embedding
                )
                detection.payload["body_score"] = body_score
                if name is None:
                    remaining_unknown += 1
                    continue
                detection.payload["person"] = name
                detection.payload["source"] = "body_retroactive"
                corrected += 1

        frames = [
            _public_frame(frame)
            for frame in sorted(stream.frames.values(), key=lambda item: item.frame_index)
        ]

    return jsonify(
        stream_id=stream_id,
        provisional=False,
        corrected=corrected,
        remaining_unknown=remaining_unknown,
        frames=frames,
    )


@yolo_adaface_two_pass.post("/extract/yolo_adaface_two_pass/reset")
@yolo_adaface_two_pass.doc(summary="Discard all retained two-pass state for one video")
def reset_yolo_adaface_two_pass():
    try:
        stream_id = _validate_stream_id()
    except ValueError as error:
        return jsonify(error=str(error)), 400
    return jsonify(stream_id=stream_id, reset=_get_store().reset(stream_id))
