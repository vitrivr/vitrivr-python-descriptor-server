import argparse

import app

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--host', type=str, help='Host to connect to.', default='localhost')
parser.add_argument('--port', type=int, help='Port to listen on.', default=8888)
parser.add_argument('--device', type=str, help='Device to use for feature extraction.', default='cpu')

args = parser.parse_args()

device = args.device

if __name__ == '__main__':
    app.entrypoint(host=args.host, port=args.port)
