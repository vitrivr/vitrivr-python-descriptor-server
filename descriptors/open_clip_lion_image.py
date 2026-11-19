import base64
from io import BytesIO

import open_clip
import torch
from PIL import Image
from apiflask import APIBlueprint
from flask import jsonify, request

from descriptors.image_batch import decode_image_batch, encode_images

#from main import device #FIXME cyclic dependency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

open_clip_lion_image = APIBlueprint('open_clip_lion_image', __name__)

model, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32',
                                                             pretrained='laion5b_s13b_b90k')
model = model.to(device).eval()
tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')


@open_clip_lion_image.post("/extract/clip_image")
@open_clip_lion_image.doc(
    summary="CLIP endpoint for feature extraction on image, where the image is transmitted in the body by a data URL")
def clip_image():
    data = request.form.get('data', '')
    header, encoded = data.split("base64,", 1)

    try:
        image = Image.open(BytesIO(base64.b64decode(encoded)))
    except Exception as e:
        print(f"Error decoding image: {e}")
        return "[]"

    return jsonify(encode_images([image], model, preprocess, device)[0])


@open_clip_lion_image.post("/extract/clip_image_batch")
@open_clip_lion_image.doc(summary="CLIP image embeddings for an ordered JSON array of image data URLs")
def clip_image_batch():
    try:
        images = decode_image_batch(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(encode_images(images, model, preprocess, device))
