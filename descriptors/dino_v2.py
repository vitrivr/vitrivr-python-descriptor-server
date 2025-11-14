import base64
from io import BytesIO

import torch
import torchvision.transforms as transforms

from PIL import Image
from apiflask import APIBlueprint
from flask import request
import gc

dino_v2 = APIBlueprint('dino_v2', __name__)

# Load DINOv2 model
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dino_model = dinov2_model.to(device)

# Image transformation
transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(244),
    transforms.CenterCrop(224),
    transforms.Normalize([0.5], [0.5])
])


@dino_v2.post("/extract/dino")
@dino_v2.doc(
    summary="DINOv2 endpoint for feature extraction on image, where the image is transmitted in the body by a data URL"
)
def dino_image():
    data = request.form.get('data', '')
    header, encoded = data.split("base64,", 1)

    try:
        image = Image.open(BytesIO(base64.b64decode(encoded)))
    except Exception as e:
        print(f"Error decoding image: {e}")
        return "[]"

    transformed_img = transform_image(image)[:3].unsqueeze(0).to(device)
    feature = dino_model(transformed_img).detach().cpu().numpy().flatten().tolist()
    gc.collect()

    return feature
