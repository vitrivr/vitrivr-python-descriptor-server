"""Common request decoding and inference for image-embedding batches."""

import base64
from io import BytesIO

import torch
from PIL import Image


MAX_IMAGE_BATCH_SIZE = 32


def decode_image_batch(payload):
    """Decode a bounded list of base64 image data URLs in input order."""
    images = payload.get('images') if isinstance(payload, dict) else None
    if not isinstance(images, list) or not 1 <= len(images) <= MAX_IMAGE_BATCH_SIZE:
        raise ValueError(f'Expected between 1 and {MAX_IMAGE_BATCH_SIZE} images')
    decoded = []
    for index, data in enumerate(images):
        try:
            if not isinstance(data, str):
                raise ValueError('Image must be a string')
            header, encoded = data.split(',', 1)
            if not header.startswith('data:image/') or not header.endswith(';base64'):
                raise ValueError('Expected a base64 image data URL')
            with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
                decoded.append(image.convert('RGB'))
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError(f'Invalid image at index {index}') from exc
    return decoded


def encode_images(images, model, preprocess, device):
    """Run one model forward pass and return one normalized vector per image."""
    inputs = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.inference_mode():
        features = model.encode_image(inputs)
        return torch.nn.functional.normalize(features, dim=-1).float().cpu().tolist()
