from flask import Flask
from flask_cors import CORS
import os
import pathlib
from infrastructure.logger import setup_logger, get_logger
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
        try:
            pathlib.Path(v).mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            # If we can't create the directory (read-only filesystem), 
            # try fallback locations
            if "/data" in v:
                fallback = v.replace("/data", "/tmp")
                try:
                    pathlib.Path(fallback).mkdir(parents=True, exist_ok=True)
                    os.environ[k] = fallback
                    logger = get_logger() if 'get_logger' in globals() else None
                    if logger:
                        logger.warning(f"Fallback: {k} set to {fallback} due to: {e}")
                    else:
                        print(f"Warning: {k} set to {fallback} due to: {e}")
                except Exception:
                    # If fallback also fails, just continue without creating the directory
                    pass

    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)

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
                logger.warning("App will continue but database functionality may be limited")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=7860)