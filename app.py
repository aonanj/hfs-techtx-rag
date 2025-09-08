from flask import Flask
from flask_cors import CORS
import os
import pathlib
from infrastructure.logger import setup_logger
from routes.web import web_bp
from routes.api import api_bp
import infrastructure.database as db


setup_logger()

def create_app():
    app = Flask(__name__, template_folder='static')
    CORS(app)
    app.config.from_object('config.Config')
    for k,v in {
        "PERSIST_DIRECTORY": "/data/chroma_db",
        "XDG_CACHE_HOME": "/data/.cache",
        "HF_HOME": "/data/.cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/data/.cache/huggingface/hub",
        "TRANSFORMERS_CACHE": "/data/.cache/hf",
        "SENTENCE_TRANSFORMERS_HOME": "/data/.cache/sentence-transformers",
        "HOME": "/data",
    }.items():
        os.environ.setdefault(k, v)
        pathlib.Path(v).mkdir(parents=True, exist_ok=True)

    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)

    with app.app_context():
        db.init_db()

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=7860)