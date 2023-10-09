from apiflask import APIFlask

app = APIFlask(__name__, title='ExternalPython Descriptor Server API for vitrivr', version='1.0.0')

# import necessary modules
from descriptors.open_clip_lion import open_clip_lion
from descriptors.dino_v2 import dino_v2


# specify here all modules, that will be needed for feature extraction server
def register_modules():
    app.register_blueprint(open_clip_lion)
    app.register_blueprint(dino_v2)


def entrypoint(host, port, args):
    app.run(host=host, port=port)
