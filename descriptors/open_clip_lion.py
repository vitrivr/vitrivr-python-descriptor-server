import json

import open_clip
import torch
from PIL import Image

import main

model, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32',
                                                             pretrained='laion5b_s13b_b90k')
model = model.to(main.device)
tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')


def text(query):
    if query is None:
        return "[]"

    return json.dumps(feature_text(query).tolist())


def feature_text(query):
    text = tokenizer(query).to(main.device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()


def image(img):
    if img is None:
        return "[]"

    return json.dumps(feature_image(img).tolist())


def feature_image(img):
    print("Image " + img)
    img = preprocess(Image.open(img)).unsqueeze(0).to(main.device)
    with torch.no_grad():
        image_features = model.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
