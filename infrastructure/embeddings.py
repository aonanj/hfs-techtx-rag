"""Embedding utilities.

Generates embeddings for chunk records lacking a stored vector and writes
results to the SQLite `embeddings` table. Also upserts into a Vertex AI 
Matching Engine index.

Primary entrypoint:
    generate_and_store_embeddings(doc_id: str | None = None,
                                   model: str | None = None,
                                   batch_size: int = 64) -> int

Requirements / Assumptions:
* Uses OpenAI Embeddings API (model default: text-embedding-3-large).
* Stores vectors as little-endian float32 binary blobs (dimension recorded).
* Uses Vertex AI Matching Engine for vector indexing. Requires environment
  variables for project, region, and index endpoint.
* Current DB schema has `chunk_id` as PRIMARY KEY in embeddings table, so only
  one embedding per chunk is effectively stored despite a `model` column.

Graceful degradation:
* If openai or google-cloud-aiplatform packages are missing or API calls fail, 
  logs an error and aborts/skips steps.
* Batches requests; naive exponential backoff on rate / transient errors.

Future improvements:
* Streaming ingestion & async concurrency.
* Token counting for dynamic batch sizing.
"""
from __future__ import annotations

import os
from typing import List, Optional
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
import time

from .logger import get_logger

logger = get_logger()

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
CHROMA_DATA_PATH = "/data/chroma_db"
CHROMA_COLLECTION = "techtrans_rag"


def get_chroma_collection():
    """Return a ChromaDB collection instance."""
    # Get the API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Initialize the embedding function with the API key
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=DEFAULT_MODEL
    )
    
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    
    # Pass the embedding function to the get_or_create_collection method
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    
    return collection


def _ensure_openai_client() -> Optional[OpenAI]:
    """Ensure OpenAI client is available and configured."""
    if OpenAI is None:
        logger.error("openai package not installed; cannot generate embeddings")
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set; cannot generate embeddings")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:  # pragma: no cover
        logger.error("Failed to init OpenAI client: %s", e)
        return None


def generate_embeddings(
    texts: List[str],
    model: str | None = None,
    max_retries: int = 5,
) -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    model = model or DEFAULT_MODEL
    client = _ensure_openai_client()
    if not client:
        return []

    attempt = 0
    while True:
        try:
            resp = client.embeddings.create(model=model, input=texts)
            vectors = [d.embedding for d in resp.data]
            if len(vectors) != len(texts):
                raise RuntimeError("Mismatch between returned embeddings and input size")
            return vectors
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Failed embedding batch after %d attempts: %s", attempt - 1, e
                )
                return []
            sleep_for = min(2**attempt, 30)
            logger.warning(
                "Embedding batch error (%s). Retry %d/%d in %.1fs",
                e,
                attempt,
                max_retries,
                sleep_for,
            )
            time.sleep(sleep_for)


def generate_embedding(
    text: str,
    model: str | None = None,
    max_retries: int = 5,
) -> List[float]:
    """Generate an embedding for a single text."""
    embeddings = generate_embeddings([text], model=model, max_retries=max_retries)
    return embeddings[0] if embeddings else []


__all__ = [
    "generate_embeddings",
    "generate_embedding",
    "get_chroma_collection",
]
