# vitrivr Python descriptor server

This server exposes general image/text features as well as face extraction and
person-identification descriptors. The examples below cover all implemented
endpoints. CLIP is enabled in `main.py`; DINOv2 is currently disabled and can
be enabled as described below.

## Start the server

```bash
FACE_GALLERY_DIR=face-gallery \
ADAFACE_MAX_IMAGE_SIZE=1920 \
BODY_REID_MODEL=models/person-reidentification-retail-0286/person-reidentification-retail-0286.xml \
OPENVINO_DEVICE=CPU \
venv/bin/python main.py
```

The server listens on `http://localhost:8888`. On a machine where OpenVINO
should choose the device, use `OPENVINO_DEVICE=AUTO` instead.

These default paths do not need to be set explicitly:

```text
ADAFACE_DIR=models/AdaFace
ADAFACE_CHECKPOINT=models/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt
YOLO_MODEL=yolo26n.pt
```

The face gallery contains one directory per known person:

```text
face-gallery/
├── Alice/
│   ├── image-1.jpg
│   └── image-2.jpg
└── Bob/
    └── image-1.jpg
```

## Prepare an image for curl

All image endpoints use a base64 data URL in the `data` form field:

```bash
IMAGE_PATH=/absolute/path/to/frame.jpg
IMAGE_BASE64="$(base64 < "$IMAGE_PATH" | tr -d '\n')"
```

The vectors in the example responses are abbreviated. Real AdaFace embeddings
contain 512 floating-point values.

## Health check

```bash
curl --silent http://localhost:8888/health | jq
```

Response:

```json
{
  "status": "ok"
}
```

## CLIP and DINOv2 endpoints

The CLIP descriptors are enabled. DINOv2 is implemented, but its blueprint
registration is currently commented out in `main.py`. Enable DINOv2 before
starting the server if that endpoint is needed:

```python
def register_modules():
    app.register_blueprint(open_clip_lion_text)
    app.register_blueprint(open_clip_lion_image)
    app.register_blueprint(dino_v2)
    app.register_blueprint(adaface)
    app.register_blueprint(yolo_adaface_forward)
    app.register_blueprint(yolo_adaface_two_pass)
```

CLIP and DINOv2 load their models during server startup. The first startup may
therefore take longer while model files are downloaded.

### CLIP image embedding

```bash
curl --silent -X POST http://localhost:8888/extract/clip_image \
  -F "data=data:image/jpeg;base64,$IMAGE_BASE64" | jq
```

Response, abbreviated from the full feature vector:

```json
[
  0.0182,
  -0.0271,
  0.0043
]
```

The returned CLIP vector is L2-normalized.

### CLIP text embedding

Prepare a UTF-8 text data URL:

```bash
TEXT="a person wearing a red jacket"
TEXT_BASE64="$(printf '%s' "$TEXT" | base64 | tr -d '\n')"
```

Send it to the text endpoint:

```bash
curl --silent -X POST http://localhost:8888/extract/clip_text \
  -F "data=data:text/plain;utf-8,$TEXT_BASE64" | jq
```

Response, abbreviated from the full feature vector:

```json
[
  -0.0064,
  0.0318,
  0.0127
]
```

The image and text endpoints use the same CLIP embedding space, and both
vectors are L2-normalized.

### DINOv2 image embedding

```bash
curl --silent -X POST http://localhost:8888/extract/dino \
  -F "data=data:image/jpeg;base64,$IMAGE_BASE64" | jq
```

Response, abbreviated from the full feature vector:

```json
[
  0.7341,
  -1.2085,
  0.0917
]
```

The configured `dinov2_vits14` model returns a 384-dimensional image feature
vector. Unlike the CLIP descriptor implementation, this endpoint does not
explicitly L2-normalize its output.

## AdaFace image extraction

This stateless endpoint detects visible faces, identifies known people, and
returns face embeddings:

```bash
curl --silent -X POST http://localhost:8888/extract/adaface \
  -F "data=data:image/jpeg;base64,$IMAGE_BASE64" \
  -F "max_faces=20" \
  -F "threshold=0.25" \
  -F "identify=true" \
  -F "include_embedding=true" | jq
```

Response:

```json
{
  "faces": [
    {
      "box": [1362, 757, 2163, 1813],
      "person": "Alice",
      "score": 0.78,
      "embedding": [0.012, -0.031, 0.008]
    }
  ],
  "gallery_configured": true,
  "image": {
    "width": 3520,
    "height": 4400
  }
}
```

If no gallery is configured or no match passes the threshold, `person` is
`null`. To omit the large feature vectors:

```bash
curl --silent -X POST http://localhost:8888/extract/adaface \
  -F "data=data:image/jpeg;base64,$IMAGE_BASE64" \
  -F "include_embedding=false" | jq
```

## Forward YOLO + AdaFace + body ReID

This endpoint processes successive frames from one video. YOLO detects people,
AdaFace identifies visible faces, and body ReID can propagate a face-confirmed
identity to later frames. Use one unique `stream_id` per video and submit its
frames sequentially.

### Submit a frame

```bash
curl --silent -X POST http://localhost:8888/extract/yolo_adaface_forward \
  -F "stream_id=video-001" \
  -F "data=data:image/jpeg;base64,$IMAGE_BASE64" \
  -F "person_confidence=0.25" \
  -F "face_threshold=0.25" \
  -F "max_persons=20" \
  -F "include_face_embedding=true" \
  -F "include_body_embedding=false" | jq
```

Response when AdaFace recognizes a face:

```json
{
  "stream_id": "video-001",
  "image": {
    "width": 1920,
    "height": 1080
  },
  "gallery_configured": true,
  "body_reid_configured": true,
  "persons": [
    {
      "box": [340, 120, 890, 1050],
      "detection_score": 0.96,
      "person": "Alice",
      "source": "face",
      "face_score": 0.78,
      "body_score": null,
      "face": {
        "box": [510, 155, 680, 360],
        "embedding": [0.012, -0.031, 0.008]
      }
    }
  ]
}
```

In a later frame where the face is hidden, the relevant person fields can be:

```json
{
  "person": "Alice",
  "source": "body",
  "face_score": null,
  "body_score": 0.72,
  "face": null
}
```

### Reset forward state

Call this after the final frame:

```bash
curl --silent -X POST http://localhost:8888/extract/yolo_adaface_forward/reset \
  -F "stream_id=video-001" | jq
```

Response:

```json
{
  "stream_id": "video-001",
  "reset": true
}
```

## Retroactive two-pass identification

This endpoint stores descriptors and provisional results for sampled frames.
After all frames arrive, finalization retries earlier unknown observations using
body identities learned anywhere in the video.

`BODY_REID_MODEL` must be configured. Feature vectors are never rewritten;
finalization only revises identity labels and body scores.

### Submit the first sampled frame

```bash
curl --silent -X POST http://localhost:8888/extract/yolo_adaface_two_pass \
  -F "stream_id=video-002" \
  -F "frame_index=0" \
  -F "timestamp_seconds=0.0" \
  -F "data=data:image/jpeg;base64,$IMAGE_BASE64" \
  -F "person_confidence=0.25" \
  -F "face_threshold=0.25" \
  -F "max_persons=20" \
  -F "include_face_embedding=true" \
  -F "include_body_embedding=false" | jq
```

Provisional response:

```json
{
  "stream_id": "video-002",
  "provisional": true,
  "stored_frame_count": 1,
  "frame": {
    "frame_index": 0,
    "timestamp_seconds": 0.0,
    "image": {
      "width": 1920,
      "height": 1080
    },
    "persons": [
      {
        "box": [340, 120, 890, 1050],
        "detection_score": 0.96,
        "person": null,
        "source": null,
        "face_score": null,
        "body_score": null,
        "face": null
      }
    ]
  }
}
```

### Submit another sampled frame

Use the same stream ID and a unique frame index:

```bash
IMAGE_PATH=/absolute/path/to/frame-125.jpg
IMAGE_BASE64="$(base64 < "$IMAGE_PATH" | tr -d '\n')"

curl --silent -X POST http://localhost:8888/extract/yolo_adaface_two_pass \
  -F "stream_id=video-002" \
  -F "frame_index=125" \
  -F "timestamp_seconds=5.0" \
  -F "data=data:image/jpeg;base64,$IMAGE_BASE64" | jq
```

Repeat this for every sampled frame. A duplicate `frame_index` within the same
stream returns HTTP `409`.

### Finalize and obtain authoritative results

```bash
curl --silent -X POST http://localhost:8888/extract/yolo_adaface_two_pass/finalize \
  -F "stream_id=video-002" | jq
```

Final response:

```json
{
  "stream_id": "video-002",
  "provisional": false,
  "corrected": 1,
  "remaining_unknown": 0,
  "frames": [
    {
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "image": {
        "width": 1920,
        "height": 1080
      },
      "persons": [
        {
          "box": [340, 120, 890, 1050],
          "detection_score": 0.96,
          "person": "Alice",
          "source": "body_retroactive",
          "face_score": null,
          "body_score": 0.71,
          "face": null
        }
      ]
    },
    {
      "frame_index": 125,
      "timestamp_seconds": 5.0,
      "image": {
        "width": 1920,
        "height": 1080
      },
      "persons": [
        {
          "box": [355, 115, 905, 1050],
          "detection_score": 0.97,
          "person": "Alice",
          "source": "face",
          "face_score": 0.79,
          "body_score": null,
          "face": {
            "box": [525, 150, 695, 355],
            "embedding": [0.012, -0.031, 0.008]
          }
        }
      ]
    }
  ]
}
```

The finalization response is authoritative. Earlier HTTP responses cannot be
modified after they have been returned.

### Reset two-pass state

```bash
curl --silent -X POST http://localhost:8888/extract/yolo_adaface_two_pass/reset \
  -F "stream_id=video-002" | jq
```

Response:

```json
{
  "stream_id": "video-002",
  "reset": true
}
```

## Processing several videos

Use a different stream ID for every video:

```text
video-a.mp4 -> stream_id=video-a
video-b.mp4 -> stream_id=video-b
video-c.mp4 -> stream_id=video-c
```

AdaFace identities from `FACE_GALLERY_DIR` are global. Body-ReID observations
are scoped to one stream because clothing can change between videos. Stream
state is held in server memory and is not shared between multiple server
processes or replicas.
