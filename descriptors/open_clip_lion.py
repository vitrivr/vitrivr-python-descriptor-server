import base64
from io import BytesIO
import json
from apiflask import APIBlueprint
from flask import request
from core.main import device
import open_clip
import torch
from PIL import Image
import urllib

open_clip_lion = APIBlueprint('open_clip_lion', __name__)

model, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32',
                                                             pretrained='laion5b_s13b_b90k')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')


@open_clip_lion.post("/extract/clip/text")
@open_clip_lion.doc(summary="CLIP endpoint for feature extraction on text")
def clip_text():
    data = request.form.get('data', '')
    header, encoded = data.split("utf-8,", 1)

    try:
        text = urllib.parse.unquote(encoded)
    except Exception as e:
        print(f"Error decoding text: {e}")
        return "[]"
    feature = "[]" if text is None else json.dumps(feature_text("this is just a first example").tolist())
    return feature


@open_clip_lion.post("/extract/clip/image")
@open_clip_lion.doc(
    summary="CLIP endpoint for feature extraction on image, where the image is transmitted in the body by a data URL"
)
def clip_image():
    data = request.form.get('data', '')
    header, encoded = data.split("base64,", 1)

    try:
        image = Image.open(BytesIO(base64.b64decode(encoded)))
    except Exception as e:
        print(f"Error decoding image: {e}")
        return "[]"

    feature = "[]" if image is None else json.dumps(feature_image(image).tolist())
    return feature


def feature_text(query):
    text = tokenizer(query).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()


def feature_image(image):
    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
