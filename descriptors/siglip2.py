"""Image and text descriptors in a shared SigLIP 2 embedding space."""

import base64
import binascii
import os
from io import BytesIO
from threading import Lock

import open_clip
import torch
from apiflask import APIBlueprint
from flask import jsonify, request
from PIL import Image, UnidentifiedImageError


siglip2 = APIBlueprint('siglip2', __name__)
_model_lock = Lock()
_components = None


def _get_components():
    """Load one model for both modalities, on the first valid request."""
    global _components
    with _model_lock:
        if _components is None:
            model_name = os.environ.get('SIGLIP2_MODEL', 'hf-hub:timm/ViT-B-16-SigLIP2')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, _, preprocess = open_clip.create_model_and_transforms(model_name)
            model = model.to(device)
            model.eval()
            tokenizer = open_clip.get_tokenizer(model_name)
            _components = model, preprocess, tokenizer, device
    return _components


def _decode_data_url(data, media_type):
    header, encoded = data.split(',', 1)
    if not header.startswith('data:' + media_type):
        raise ValueError('Unexpected data URL media type')
    # CLIP clients also use ;utf-8, for base64-encoded text.
    if ';base64' not in header and not (media_type == 'text/' and header.endswith(';utf-8')):
        raise ValueError('Expected a base64 data URL')
    return base64.b64decode(encoded, validate=True)


@siglip2.post('/extract/siglip2_image')
@siglip2.doc(summary='SigLIP 2 image embedding from a data URL in the data form field')
def siglip2_image():
    try:
        payload = _decode_data_url(request.form.get('data', ''), 'image/')
        with Image.open(BytesIO(payload)) as image:
            image = image.convert('RGB')
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError):
        return jsonify(error='Expected an image encoded as a base64 data URL'), 400

    model, preprocess, _, device = _get_components()
    with torch.inference_mode():
        features = model.encode_image(preprocess(image).unsqueeze(0).to(device))
        features = torch.nn.functional.normalize(features, dim=-1)
    return jsonify(features.float().cpu().flatten().tolist())


@siglip2.post('/extract/siglip2_text')
@siglip2.doc(summary='SigLIP 2 text embedding from a data URL in the data form field')
def siglip2_text():
    try:
        text = _decode_data_url(request.form.get('data', ''), 'text/').decode('utf-8')
    except (ValueError, binascii.Error, UnicodeDecodeError):
        return jsonify(error='Expected UTF-8 text encoded as a base64 data URL'), 400

    model, _, tokenizer, device = _get_components()
    with torch.inference_mode():
        features = model.encode_text(tokenizer([text]).to(device))
        features = torch.nn.functional.normalize(features, dim=-1)
    return jsonify(features.float().cpu().flatten().tolist())
