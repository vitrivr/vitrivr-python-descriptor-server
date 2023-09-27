from apiflask import APIFlask

app = APIFlask(__name__, title='ExternalPython Descriptor Server API for vitrivr', version='1.0.0')

# import necessary modules
from descriptors.open_clip_lion import open_clip_lion


# specify here all modules, that will be needed for feature extraction server
def register_modules():
    app.register_blueprint(open_clip_lion)


def entrypoint(host, port, args):
    app.run(host=host, port=port)
