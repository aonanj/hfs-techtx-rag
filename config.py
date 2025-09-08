from dotenv import load_dotenv
import os
from flask import current_app, has_app_context

load_dotenv()

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    CHAT_MODEL = os.getenv("CHAT_MODEL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CLAUDE_KEY = os.getenv("CLAUDE_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    MAX_CONTEXT_LENGTH = os.getenv("MAX_CONTEXT_LENGTH")
    TOP_K_RESULTS = os.getenv("TOP_K_RESULTS")
    MAX_QUERY_LENGTH = os.getenv("MAX_QUERY_LENGTH")
    REQUEST_TIMEOUT = os.getenv("REQUEST_TIMEOUT")



def cfg(key, default=None, cast=lambda x: x):
    val = current_app.config.get(key) if has_app_context() else None
    if val is None:
        val = os.getenv(key, default)
    return cast(val) if val is not None else val