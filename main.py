import argparse

from apiflask import APIFlask
from flask import jsonify

app = APIFlask(__name__, title='ExternalPython Descriptor Server API for vitrivr', version='1.0.0')

@app.route("/health", methods=["GET"])  # Add thisAdd commentMore actions
def health():
    return jsonify(status="ok")

# import necessary modules
from descriptors.open_clip_lion_text import open_clip_lion_text
from descriptors.open_clip_lion_image import open_clip_lion_image
from descriptors.dino_v2 import dino_v2


# specify here all modules, that will be needed for feature extraction server
def register_modules():
    app.register_blueprint(open_clip_lion_text)
    app.register_blueprint(open_clip_lion_image)
    app.register_blueprint(dino_v2)


def entrypoint(host, port, args):
    app.run(host=host, port=port)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--host', type=str, help='Host to connect to.', default='localhost')
parser.add_argument('--port', type=int, help='Port to listen on.', default=8888)
parser.add_argument('--device', type=str, help='Device to use for feature extraction.', default='cpu')

args = parser.parse_args()
device = args.device

if __name__ == '__main__':
    register_modules()
    entrypoint(host=args.host, port=args.port, args=args.device)
