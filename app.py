from flask import Flask
from flask_cors import CORS
import os
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
        "XDG_CACHE_HOME": "/data/cache",
        "HF_HOME": "/data/.huggingface",
        "HOME": "/data",
        "UPLOAD_FOLDER": "/data/corpus_raw",
        "CLEANED_FOLDER": "/data/corpus_clean",
        "MANIFEST_DIR": "/data/manifest",
        "CHUNKS_DIR": "/data/chunks"
    }.items():
        os.environ.setdefault(k, v)
        abs_path = os.path.abspath(v)
        os.makedirs(abs_path, exist_ok=True)

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
                logger.error("App will continue but database functionality may be limited")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=7860)