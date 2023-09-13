from flask import request
from flask_openapi3 import Info, Tag
from flask_openapi3 import OpenAPI

from descriptors import open_clip_lion

info = Info(title="vitrivr Python Descriptor Server API", version="1.0.0")
app = OpenAPI(__name__, info=info)

text = Tag(name="text", description="Textual Description")
image = Tag(name="image", description="Image Representation")


@app.get("/retrieve/clip/text/<text>", summary="endpoint of CLIP for text", tags=[text])
def clip_text():
    query = request.view_args['text']
    feature = open_clip_lion.text(query)
    return feature


@app.get("/retrieve/clip/image/<path>", summary="endpoint of CLIP for images", tags=[image])
def clip_image():
    query = request.view_args['path']
    feature = open_clip_lion.image(query)
    return feature


def entrypoint(host, port):
    app.run(host=host, port=port)
