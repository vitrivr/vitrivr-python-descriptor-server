import base64
import json
from io import BytesIO

import open_clip
import torch
from PIL import Image
from apiflask import APIBlueprint
from flask import request
import gc

#from main import device #FIXME cyclic dependency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

open_clip_lion_image = APIBlueprint('open_clip_lion_image', __name__)

model, _, preprocess = open_clip.create_model_and_transforms('xlm-roberta-base-ViT-B-32',
                                                             pretrained='laion5b_s13b_b90k')
model = model.to(device)
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

    feature = "[]" if image is None else json.dumps(feature_image(image).tolist())
    return feature


def feature_image(image):
    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        gc.collect()
        return image_features.cpu().numpy().flatten()
