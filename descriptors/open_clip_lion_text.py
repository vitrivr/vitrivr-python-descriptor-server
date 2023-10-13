import json
import urllib

import open_clip
import torch
from apiflask import APIBlueprint
from flask import request

from core.main import device

open_clip_lion_text = APIBlueprint('open_clip_lion_text', __name__)

model, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32',
                                                             pretrained='laion5b_s13b_b90k')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')


@open_clip_lion_text.post("/extract/cliptext")
@open_clip_lion_text.doc(summary="CLIP endpoint for feature extraction on text")
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


def feature_text(query):
    text = tokenizer(query).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
