from flask import Flask
from flask_cors import CORS
import os
import pathlib
import chromadb
from infrastructure.logger import setup_logger, get_logger
from routes.web import web_bp
from routes.api import api_bp
import infrastructure.database as db


setup_logger()

def create_app():
    app = Flask(__name__, template_folder='static')
    CORS(app)
    for k,v in {
        "PERSIST_DIRECTORY": "/data/chroma_db",
        "XDG_CACHE_HOME": "/data/.cache",
        "HF_HOME": "/data/.huggingface",
        "HUGGINGFACE_HUB_CACHE": "/data/.cache/huggingface/hub",
        "TRANSFORMERS_CACHE": "/data/.cache/hf",
        "SENTENCE_TRANSFORMERS_HOME": "/data/.cache/sentence-transformers",
        "HOME": "/data",
    }.items():
        os.environ.setdefault(k, v)
        pathlib.Path(v).mkdir(parents=True, exist_ok=True)


    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)
    client = chromadb.PersistentClient(path=os.environ["PERSIST_DIRECTORY"])

    with app.app_context():
        logger = get_logger()
        try:
            db.init_db()
        except Exception as e:
            # Log the error but try to continue with a database reset
            logger.error(f"Database initialization failed: {e}")
            logger.info("Attempting to reset database...")
            try:
                if db.reset_database():
                    logger.info("Database reset successful")
                else:
                    logger.error("Database reset failed - app may not function properly")
            except Exception as reset_e:
                logger.error(f"Database reset failed: {reset_e}")
                logger.error("App will continue but database functionality may be limited")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=7860)