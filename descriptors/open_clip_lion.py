# my_blueprint.py
from apiflask import APIBlueprint

from core.main import device as device

open_clip_lion = APIBlueprint('open_clip_lion', __name__)

import json
import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32',
                                                             pretrained='laion5b_s13b_b90k')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')


@open_clip_lion.get("/retrieve/clip/text/<string:text>")
@open_clip_lion.doc(summary="CLIP endpoint for feature extraction on text")
def clip_text(text):
    if text is None:
        feature = "[]"
    else:
        feature = json.dumps(feature_text(text).tolist())
    return feature


@open_clip_lion.get("/retrieve/clip/image/<string:path>")
@open_clip_lion.doc(summary="CLIP endpoint for feature extraction on image")
def clip_image(path):
    if path is None:
        feature = "[]"
    else:
        feature = json.dumps(feature_image(path).tolist())
    return feature


def feature_text(query):
    text = tokenizer(query).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()


def feature_image(img):
    print("Image " + img)
    img = preprocess(Image.open(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
