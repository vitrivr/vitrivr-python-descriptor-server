from __future__ import annotations

import base64
import importlib
import os
import sys
import threading
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from apiflask import APIBlueprint
from flask import current_app, jsonify, request
from PIL import Image, UnidentifiedImageError


Box = Tuple[int, int, int, int]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

adaface = APIBlueprint("adaface", __name__)


class _CheckpointMetadata:
    pass


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    return vector if norm < 1e-12 else vector / norm


def _boolean_parameter(name: str, default: bool) -> bool:
    value = request.form.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _decode_image(data_url: str) -> Image.Image:
    if not data_url:
        raise ValueError("Missing form field 'data'.")

    encoded = data_url.split("base64,", 1)[-1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoded, validate=True)))
        image.load()
        return image.convert("RGB")
    except (ValueError, UnidentifiedImageError, OSError) as error:
        raise ValueError("The 'data' field is not a valid base64 image.") from error


class AdaFaceRecognizer:
    def __init__(
        self,
        source_dir: Path,
        checkpoint: Path,
        architecture: str,
        max_image_size: int = 1280,
    ):
        self.source_dir = source_dir.resolve()
        self.checkpoint = checkpoint.resolve()
        self.max_image_size = max_image_size

        if not self.source_dir.is_dir():
            raise FileNotFoundError(
                f"AdaFace source directory not found: {self.source_dir}"
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"AdaFace checkpoint not found: {self.checkpoint}"
            )

        source = str(self.source_dir)
        if source not in sys.path:
            sys.path.insert(0, source)

        net = importlib.import_module("net")
        mtcnn = importlib.import_module("face_alignment.mtcnn")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        detector_device = "cuda:0" if self.device.type == "cuda" else "cpu"

        previous_directory = Path.cwd()
        try:
            os.chdir(self.source_dir / "face_alignment")
            self.detector = mtcnn.MTCNN(
                device=detector_device,
                crop_size=(112, 112),
            )
        finally:
            os.chdir(previous_directory)

        self.model = net.build_model(architecture)

        safe_metadata = (
            _CheckpointMetadata,
            "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
        )
        with torch.serialization.safe_globals([safe_metadata]):
            state = torch.load(
                self.checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        state_dict = state.get("state_dict", state)
        cleaned = {
            key[len("model.") :] if key.startswith("model.") else key: value
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(cleaned, strict=False)
        self.model.eval().to(self.device)

    def _embed_many(self, aligned_faces: List[Image.Image]) -> List[np.ndarray]:
        tensors = []
        for aligned_face in aligned_faces:
            rgb = np.asarray(aligned_face.resize((112, 112)), dtype=np.float32)
            bgr = rgb[:, :, ::-1].copy()
            tensors.append(torch.from_numpy(bgr).permute(2, 0, 1))

        tensor = torch.stack(tensors)
        tensor = ((tensor / 255.0) - 0.5) / 0.5

        with torch.inference_mode():
            feature, _ = self.model(tensor.to(self.device))
            feature = F.normalize(feature, p=2, dim=1)

        return [
            _normalize(vector)
            for vector in feature.detach().cpu().numpy()
        ]

    def detect_and_embed(
        self, image: Image.Image, max_faces: int
    ) -> List[Tuple[np.ndarray, Box]]:
        if image.width < 40 or image.height < 40:
            return []

        original_width, original_height = image.size
        detection_image = image
        longest_side = max(image.size)
        if self.max_image_size > 0 and longest_side > self.max_image_size:
            scale = self.max_image_size / longest_side
            detection_image = image.resize(
                (
                    max(1, round(original_width * scale)),
                    max(1, round(original_height * scale)),
                ),
                Image.Resampling.BOX,
            )

        boxes, faces = self.detector.align_multi(detection_image, limit=max_faces)
        if boxes is None or faces is None:
            return []

        faces = list(faces)
        if not faces:
            return []

        embeddings = self._embed_many(faces)
        scale_x = original_width / detection_image.width
        scale_y = original_height / detection_image.height
        result: List[Tuple[np.ndarray, Box]] = []
        for raw_box, embedding in zip(boxes, embeddings):
            values = np.asarray(raw_box).reshape(-1)
            box = (
                int(round(values[0] * scale_x)),
                int(round(values[1] * scale_y)),
                int(round(values[2] * scale_x)),
                int(round(values[3] * scale_y)),
            )
            result.append((embedding, box))
        return result


class FaceGallery:
    def __init__(self, recognizer: AdaFaceRecognizer, directory: Optional[Path]):
        self.people: Dict[str, np.ndarray] = {}
        if directory is not None:
            self._load(recognizer, directory.resolve())

    def _load(self, recognizer: AdaFaceRecognizer, directory: Path) -> None:
        if not directory.is_dir():
            raise FileNotFoundError(f"Face gallery directory not found: {directory}")

        for person_directory in sorted(directory.iterdir()):
            if not person_directory.is_dir():
                continue

            embeddings: List[np.ndarray] = []
            for image_path in sorted(person_directory.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    with Image.open(image_path) as image:
                        detections = recognizer.detect_and_embed(
                            image.convert("RGB"), max_faces=1
                        )
                except (OSError, UnidentifiedImageError):
                    continue
                if detections:
                    embeddings.append(detections[0][0])

            if embeddings:
                self.people[person_directory.name] = _normalize(
                    np.mean(np.stack(embeddings), axis=0)
                )

    def recognize(self, embedding: np.ndarray, threshold: float) -> Tuple[Optional[str], Optional[float]]:
        if not self.people:
            return None, None

        query = _normalize(embedding)
        name, score = max(
            (
                (person, float(np.dot(query, template)))
                for person, template in self.people.items()
            ),
            key=lambda item: item[1],
        )
        return (name if score >= threshold else None), score


class FaceService:
    def __init__(self) -> None:
        source_dir = Path(os.getenv("ADAFACE_DIR", "models/AdaFace"))
        checkpoint = Path(
            os.getenv(
                "ADAFACE_CHECKPOINT",
                str(source_dir / "pretrained" / "adaface_ir50_ms1mv2.ckpt"),
            )
        )
        architecture = os.getenv("ADAFACE_ARCH", "ir_50")
        max_image_size = int(os.getenv("ADAFACE_MAX_IMAGE_SIZE", "1280"))
        gallery_value = os.getenv("FACE_GALLERY_DIR")

        self.recognizer = AdaFaceRecognizer(
            source_dir,
            checkpoint,
            architecture,
            max_image_size=max_image_size,
        )
        self.gallery = FaceGallery(
            self.recognizer,
            Path(gallery_value) if gallery_value else None,
        )


_service: Optional[FaceService] = None
_service_lock = threading.Lock()


def _get_service() -> FaceService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = FaceService()
    return _service


@adaface.post("/extract/adaface")
@adaface.doc(
    summary=(
        "Detect faces in an image and return AdaFace embeddings with optional "
        "server-side identity matching"
    )
)
def extract_adaface():
    try:
        image = _decode_image(request.form.get("data", ""))
        max_faces = int(request.form.get("max_faces", "20"))
        threshold = float(request.form.get("threshold", "0.25"))
        include_embedding = _boolean_parameter("include_embedding", True)
        identify = _boolean_parameter("identify", True)
    except ValueError as error:
        return jsonify(error=str(error)), 400

    if not 1 <= max_faces <= 100:
        return jsonify(error="'max_faces' must be between 1 and 100."), 400
    if not -1.0 <= threshold <= 1.0:
        return jsonify(error="'threshold' must be between -1 and 1."), 400

    try:
        service = _get_service()
        detections = service.recognizer.detect_and_embed(image, max_faces)
    except FileNotFoundError as error:
        return jsonify(
            error=str(error),
            hint=(
                "Configure ADAFACE_DIR and ADAFACE_CHECKPOINT. "
                "FACE_GALLERY_DIR is optional."
            ),
        ), 503
    except Exception as error:
        current_app.logger.exception("AdaFace extraction failed")
        return jsonify(error=f"AdaFace extraction failed: {error}"), 500

    faces = []
    for embedding, box in detections:
        person: Optional[str] = None
        score: Optional[float] = None
        if identify:
            person, score = service.gallery.recognize(embedding, threshold)

        face = {
            "box": list(box),
            "person": person,
            "score": score,
        }
        if include_embedding:
            face["embedding"] = embedding.tolist()
        faces.append(face)

    return jsonify(
        faces=faces,
        image={"width": image.width, "height": image.height},
        gallery_configured=bool(service.gallery.people),
    )
