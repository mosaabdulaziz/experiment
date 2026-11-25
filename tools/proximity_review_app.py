"""The app for the proximity review stack (tools/proximity_review.sh): create_app on a given config, served by uvicorn.
Usage: python tools/proximity_review_app.py <config.yaml> <port>"""
import sys
from pathlib import Path

import uvicorn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the repo root: run as a script from tools/
from app.server import create_app  # noqa: E402

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = yaml.safe_load(f)
    uvicorn.run(create_app(config), host="127.0.0.1", port=int(sys.argv[2]))
