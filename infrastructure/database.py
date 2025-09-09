"""
ChromaDB database layer.
"""
import os
import contextlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict, Any
import numpy as np
from infrastructure.logger import get_logger

_logger = get_logger()

# ---------------------------------------------------------------------------
# Versioning for corpus mechanics
# ---------------------------------------------------------------------------
TOK_VER = int(os.getenv("TOK_VER", "1"))
SEG_VER = int(os.getenv("SEG_VER", "1"))

# ---------------------------------------------------------------------------
# ChromaDB client setup (with HuggingFace / read-only filesystem resilience)
# ---------------------------------------------------------------------------
# Set a writable cache directory for ChromaDB with fallbacks
def _setup_cache_dir():
    """Set up a writable cache directory for ChromaDB with fallbacks."""
    cache_candidates = [
        "/data/chroma_cache",
        "/data/chromacache", 
        "/tmp/.cache/chroma"
    ]
    
    for cache_dir in cache_candidates:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Test if writable
            test_file = os.path.join(cache_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            os.environ["CHROMA_CACHE_DIR"] = cache_dir
            _logger.info(f"Using ChromaDB cache directory: {cache_dir}")
            return cache_dir
        except Exception:
            continue
    
    # If all fail, don't set the cache dir and let ChromaDB handle it
    _logger.error("Could not set up ChromaDB cache directory, using default")
    return None

_setup_cache_dir()

CHROMA_PATH = os.getenv("PERSIST_DIRECTORY", "/data/chroma_db")

def _init_chroma_client():
    """Initialize a Chroma client with robust write tests & fallbacks.

    New behaviour: after creating a PersistentClient we attempt an actual write
    (temp collection + add + delete). If that raises a *read-only* or similar
    error we immediately fall back to the next candidate path. This addresses
    the Hugging Face Spaces scenario where a volume exists but underlying
    database files were created with different permissions (so directory write
    test passed but sqlite writes fail with 'attempt to write a readonly database').

    Order of attempts:
      1. Env provided CHROMA_PATH
      2. /data/chdata   (persistent on HF Spaces when writable)
      3. /home/user/chroma_db (common writable home on Spaces)
      4. /tmp/chroma_db    (ephemeral but always writable)
      5. In-memory client  (last resort, non-persistent)
    """
    import time
    import chromadb  # local import so we can still return memory client last
    from chromadb.config import Settings

    # Allow a force-tmp override (e.g. CHROMA_FORCE_TMP=1)
    force_tmp = os.getenv("CHROMA_FORCE_TMP") in {"1", "true", "yes", "on"}

    env_path = os.getenv("PERSIST_DIRECTORY", "chroma_db")
    # Order of path attempts (most desired first)
    candidates: list[str] = []
    if force_tmp:
        candidates = ["/tmp/chroma_db"]
    else:
        if env_path:
            candidates.append(env_path)
        # Common writable locations on HF Spaces
        candidates.extend([
            "/data/chroma_db",
            "/home/user/chroma_db",
            "/data/db_chroma",
        ])

    tried: list[tuple[str, str]] = []

    def _dir_writeable(path: str) -> bool:
        try:
            test_file = os.path.join(path, f".write_test_{int(time.time()*1000)}")
            with open(test_file, "w", encoding="utf-8") as tf:
                tf.write("ok")
            os.remove(test_file)
            return True
        except Exception as e:  # pragma: no cover - env specific
            tried.append((path, f"dir not writable: {e}"))
            return False

    for cand in candidates:
        abs_path = os.path.abspath(cand)
        try:
            os.makedirs(abs_path, exist_ok=True)
        except Exception as e:
            tried.append((abs_path, f"makedirs failed: {e}"))
            continue

        _logger.info(f"Attempting Chroma PersistentClient at {abs_path}")
        if not _dir_writeable(abs_path):
            _logger.error(f"Directory not writable: {abs_path}")
            continue

        try:
            client = chromadb.PersistentClient(path=abs_path)
        except Exception as e:  # immediate failure
            tried.append((abs_path, f"create failed: {e}"))
            _logger.error(f"Chroma path {abs_path} not usable, trying next candidate…")
            continue

        # Basic list works?
        try:
            client.list_collections()
        except Exception as e:
            tried.append((abs_path, f"list failed: {e}"))
            continue

        # WRITE TEST (critical)
        test_coll_name = f"rw_test_{int(time.time()*10)}"
        try:
            tc = client.get_or_create_collection(name=test_coll_name)  # type: ignore
            tc.add(ids=["0"], metadatas=[{"t": "x"}], documents=["test"])
            client.delete_collection(name=test_coll_name)
            _logger.error(f"Chroma connected with write access at {abs_path}")
            return client
        except Exception as e:
            msg = str(e).lower()
            tried.append((abs_path, f"write test failed: {e}"))
            # Clean up partial collection if it exists
            try:
                client.delete_collection(name=test_coll_name)
            except Exception:
                pass
            # Detect read-only / permission style errors; if so, try next path
            if any(tok in msg for tok in ["read-only", "readonly", "permission", "attempt to write"]):
                _logger.error(f"Chroma path {abs_path} not writable for DB operations, trying next candidate…")
                continue
            # For corruption cases attempt a reset once
            if any(tok in msg for tok in ["corrupt", "corruption", "malformed", "disk image"]):
                _logger.error(f"Possible DB corruption at {abs_path}; attempting reset once")
                try:
                    client.reset()
                    # Re-run write test after reset
                    tc = client.get_or_create_collection(name=test_coll_name)  # type: ignore
                    tc.add(ids=["0"], metadatas=[{"t": "x"}], documents=["test"])
                    client.delete_collection(name=test_coll_name)
                    _logger.info(f"Recovered corrupted DB at {abs_path}")
                    return client
                except Exception as e2:
                    tried.append((abs_path, f"reset failed: {e2}"))
                    continue
            # Otherwise just continue to next
            continue

    _logger.error("Failed to initialize persistent Chroma client; attempts=%s. Using ephemeral in-memory fallback.", tried)
    # Final fallback: non-persistent in-memory client so the app can still run
    return chromadb.Client(Settings(is_persistent=False, anonymized_telemetry=False))  # type: ignore

_client = _init_chroma_client()

# ---------------------------------------------------------------------------
# Ensure writable HOME / cache dirs (HF Spaces & read-only root fix)
# ---------------------------------------------------------------------------
def _ensure_writable_caches():  # pragma: no cover (env specific)
    try:
        current_home = os.path.expanduser("~")
        # If home resolves to root or isn't writable, pick a fallback
        if current_home == "/" or not os.access(current_home, os.W_OK):
            for cand in ["/data", "/home/user", "/data/home"]:
                try:
                    os.makedirs(cand, exist_ok=True)
                    test_path = os.path.join(cand, "home_write_test")
                    with open(test_path, "a", encoding="utf-8") as f:
                        f.write("ok")
                    os.remove(test_path)
                    os.environ["HOME"] = cand
                    _logger.error(f"Reset HOME to writable directory: {cand}")
                    current_home = cand
                    break
                except Exception:  # continue trying fallbacks
                    continue

        # Set cache related env vars if absent
        cache_base = os.path.join(os.environ.get("HOME", current_home), "/data/cache")
        for var in ["XDG_CACHE_HOME", "HF_HOME", "TRANSFORMERS_CACHE"]:
            if not os.getenv(var):
                try:
                    os.makedirs(cache_base, exist_ok=True)
                    os.environ[var] = cache_base
                except Exception:
                    pass
    except Exception as e:
        _logger.error(f"Cache/HOME setup skipped due to error: {e}")

_ensure_writable_caches()

# Using a default embedding function for the collections, though we'll be providing our own embeddings.
# This is required by ChromaDB.
# The type hint for embedding_function is complex, and mypy has issues with it.
# Using type: ignore to suppress the error.

def _init_collections():
    """Initialize ChromaDB collections with error handling."""
    global _documents_collection, _chunks_collection
    try:
        _documents_collection = _client.get_or_create_collection(
            name="documents"
        )
        _chunks_collection = _client.get_or_create_collection(
            name="chunks"
        )
        _logger.error("Successfully initialized ChromaDB collections.")
    except Exception as e:
        _logger.error(f"Failed to initialize collections: {e}")
        # Re-raise the exception to be handled by the application
        raise

# Initialize collections
_init_collections()

# ---------------------------------------------------------------------------
# Data classes (replacing ORM)
# ---------------------------------------------------------------------------
@dataclass
class Document:
    doc_id: int
    sha256: str
    title: Optional[str] = None
    source_path: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    doc_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    governing_law: Optional[str] = None
    party_roles: Optional[str] = None
    industry: Optional[str] = None
    effective_date: Optional[datetime] = None
    chunks: List['Chunk'] = field(default_factory=list)

    @classmethod
    def from_chroma(cls, doc_id: int, metadata: Dict[str, Any]):
        if metadata.get('effective_date') and isinstance(metadata['effective_date'], str):
            metadata['effective_date'] = datetime.fromisoformat(metadata['effective_date'])
        if metadata.get('created_at') and isinstance(metadata['created_at'], str):
            metadata['created_at'] = datetime.fromisoformat(metadata['created_at'])
        
        # Filter metadata to only include fields expected by the dataclass
        known_fields = {f.name for f in fields(cls) if f.init}
        filtered_metadata = {k: v for k, v in metadata.items() if k in known_fields}

        return cls(doc_id=doc_id, **filtered_metadata)

@dataclass
class Chunk:
    chunk_id: int
    doc_id: int
    chunk_index: int
    text: str
    token_count: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    path: Optional[str] = None
    numbers_present: Optional[bool] = None
    definition_terms: Optional[str] = None
    clause_type: Optional[str] = None
    tok_ver: int = TOK_VER
    seg_ver: int = SEG_VER
    document: Optional[Document] = None
    embedding: Optional['Embedding'] = None

    @classmethod
    def from_chroma(cls, chunk_id: int, metadata: Dict[str, Any], document_text: str):
        metadata['text'] = document_text
        known_fields = {f.name for f in fields(cls) if f.init}
        filtered_metadata = {k: v for k, v in metadata.items() if k in known_fields}
        return cls(chunk_id=chunk_id, **filtered_metadata)

@dataclass
class Embedding:
    id: int
    chunk_id: int
    model: str
    dim: int
    vector: list[float]
    chunk: Optional[Chunk] = None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init_db():
    """Initializes ChromaDB collections."""
    try:
        _init_collections()
        _logger.info("ChromaDB collections ensured.")
    except Exception as e:
        _logger.error(f"Failed to initialize database collections: {e}")
        raise


def reset_chroma_collections() -> bool:
    """Delete known Chroma collections if they exist.

    Returns True if the operation completed (even if some collections did not exist).
    """
    try:
        for name in ["documents", "chunks", "techtrans_rag"]:
            try:
                _client.delete_collection(name=name)  # type: ignore
                _logger.info("Deleted Chroma collection '%s'", name)
            except Exception:
                # Collection might not exist – ignore
                pass
        return True
    except Exception as e:  # pragma: no cover - defensive
        _logger.error("Failed resetting Chroma collections: %s", e)
        return False


def reset_database() -> bool:
    """Reset the entire ChromaDB database.
    
    This is useful for recovering from database corruption issues.
    Returns True if successful.
    """
    try:
        _logger.info("Resetting entire ChromaDB database...")
        _client.reset()
        # Re-initialize collections after reset
        _init_collections()
        _logger.info("Database reset and collections re-initialized successfully.")
        return True
    except Exception as e:
        _logger.error(f"Failed to reset database: {e}")
        return False


def rebuild_chroma_client() -> None:
    """Rebuild the underlying Chroma client (e.g., after filesystem wipe)."""
    global _client
    try:
        _logger.info("Rebuilding Chroma client after data reset.")
        _client = _init_chroma_client()
        init_db()
    except Exception as e:  # pragma: no cover - defensive
        _logger.error("Failed to rebuild Chroma client: %s", e)


def full_reset_chroma() -> bool:
    """Convenience helper: delete collections and recreate them fresh."""
    ok = reset_chroma_collections()
    if not ok:
        return False
    init_db()
    return True

def get_session():
    return None

def get_db():
    return None

@contextlib.contextmanager
def session_scope():
    """Provide a transactional scope (no-op for ChromaDB)."""
    yield None

# --------------- CRUD ---------------
def add_document(*, sha256, title=None, source_path=None, doc_type=None, jurisdiction=None, industry=None, party_roles=None, governing_law=None, effective_date=None):
    existing = _documents_collection.get(where={"sha256": sha256})

    if doc_type and isinstance(doc_type, list):
        doc_type = ", ".join(doc_type)
    if jurisdiction and isinstance(jurisdiction, list):
        jurisdiction = ", ".join(jurisdiction)
    if industry and isinstance(industry, list):
        industry = ", ".join(industry)
    if party_roles and isinstance(party_roles, list):
        party_roles = ", ".join(party_roles)
    if governing_law and isinstance(governing_law, list):
        governing_law = ", ".join(governing_law)

    if existing['ids'] and existing['metadatas']:
        doc_id = int(existing['ids'][0])
        metadata = dict(existing['metadatas'][0])
        _logger.info("Document with sha256 %s already exists (doc_id=%s); checking for metadata enrichment", sha256, doc_id)
        
        updated = False
        if doc_type and not metadata.get("doc_type"):
            metadata["doc_type"] = doc_type
            updated = True
        if jurisdiction and not metadata.get("jurisdiction"):
            metadata["jurisdiction"] = jurisdiction
            updated = True
        if industry and not metadata.get("industry"):
            metadata["industry"] = industry
            updated = True
        if party_roles and not metadata.get("party_roles"):
            metadata["party_roles"] = party_roles
            updated = True
        if governing_law and not metadata.get("governing_law"):
            metadata["governing_law"] = governing_law
            updated = True
        if effective_date and not metadata.get("effective_date"):
            metadata["effective_date"] = effective_date.isoformat()
            updated = True

        if updated:
            _documents_collection.update(ids=[str(doc_id)], metadatas=[metadata])
            _logger.info("Document metadata enriched for doc_id=%s", doc_id)
        
        return Document.from_chroma(doc_id, metadata)

    all_docs = _documents_collection.get(include=[])
    max_id = 0
    if all_docs['ids']:
        max_id = max([int(i) for i in all_docs['ids']])
    new_doc_id = max_id + 1

    metadata = {
        "sha256": sha256, "title": title, "source_path": source_path, "doc_type": doc_type,
        "jurisdiction": jurisdiction, "industry": industry, "party_roles": party_roles,
        "governing_law": governing_law, "created_at": datetime.now(timezone.utc).isoformat(),
        "effective_date": effective_date.isoformat() if effective_date else None,
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    try:
        _documents_collection.add(ids=[str(new_doc_id)], metadatas=[metadata], documents=[title or ''])
    except Exception as e:
        msg = str(e).lower()
        if any(tok in msg for tok in ["read-only", "readonly", "attempt to write", "permission"]):
            _logger.error("Initial document add failed due to read-only DB; attempting fallback rebuild: %s", e)
            # Force tmp usage on rebuild
            os.environ["CHROMA_FORCE_TMP"] = "1"
            try:
                rebuild_chroma_client()
                # Re-attempt add (globals _documents_collection reinitialized)
                _documents_collection.add(ids=[str(new_doc_id)], metadatas=[metadata], documents=[title or ''])
                _logger.info("Retry succeeded after rebuilding Chroma client on tmp path.")
            except Exception as e2:
                _logger.error("Retry add_document failed after rebuild: %s", e2)
                raise
        else:
            raise
    # Normalize return via from_chroma to ensure datetime fields are parsed
    return Document.from_chroma(new_doc_id, dict(metadata))

def add_chunk(doc_id: int, chunk_id: int, text: str, chunk_index: int, token_count: int | None = None,
              page_start: int | None = None, page_end: int | None = None, section_number: str | None = None,
              section_title: str | None = None, path: str | None = None, numbers_present: bool | None = None, definition_terms: str | None = None,
              clause_type: str | None = None, tok_ver: int = TOK_VER, seg_ver: int = SEG_VER):

    # Convert definition_terms list to comma-separated string for ChromaDB metadata
    definition_terms_str = None
    if definition_terms is not None:
        if isinstance(definition_terms, list):
            definition_terms_str = ", ".join(str(term) for term in definition_terms)
        else:
            definition_terms_str = str(definition_terms)

    metadata: Dict[str, Any] = {
        "doc_id": doc_id, "chunk_index": chunk_index, "token_count": token_count,
        "page_start": page_start, "page_end": page_end, "section_number": section_number,
        "section_title": section_title, "path": path, "numbers_present": numbers_present,
        "definition_terms": definition_terms_str, "clause_type": clause_type, "tok_ver": tok_ver, "seg_ver": seg_ver,
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    _chunks_collection.add(ids=[str(chunk_id)], documents=[text], metadatas=[metadata])
    _logger.info("DB added chunk %s for doc_id=%s", chunk_id, doc_id)

    # Avoid passing duplicate values for parameters already provided explicitly
    safe_meta = {k: v for k, v in metadata.items() if k not in {"doc_id", "chunk_index"}}
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, chunk_index=chunk_index, **safe_meta)

def upsert_embedding(chunk_id: int, model: str, dim: int, vector_bytes: bytes):
    vector = np.frombuffer(vector_bytes, dtype=np.float32).tolist()

    existing_chunk = _chunks_collection.get(ids=[str(chunk_id)], include=["metadatas"])
    if not existing_chunk['ids'] or not existing_chunk['metadatas']:
        _logger.error(f"Cannot upsert embedding for non-existent chunk_id {chunk_id}")
        return None

    metadata = dict(existing_chunk['metadatas'][0])
    metadata['embedding_model'] = model
    metadata['embedding_dim'] = dim

    _chunks_collection.update(ids=[str(chunk_id)], embeddings=[vector], metadatas=[metadata])
    
    return Embedding(id=chunk_id, chunk_id=chunk_id, model=model, dim=dim, vector=vector)

def get_document(doc_id: int):
    doc_data = _documents_collection.get(ids=[str(doc_id)])
    if not doc_data['ids'] or not doc_data['metadatas']:
        return None
    return Document.from_chroma(doc_id, dict(doc_data['metadatas'][0]))

def get_document_by_sha(sha256: str):
    doc_data = _documents_collection.get(where={"sha256": sha256})
    if not doc_data['ids'] or not doc_data['metadatas']:
        return None
    doc_id = int(doc_data['ids'][0])
    return Document.from_chroma(doc_id, dict(doc_data['metadatas'][0]))

def get_chunks_for_doc(doc_id: int):
    chunk_data = _chunks_collection.get(where={"doc_id": doc_id})
    if not chunk_data['ids'] or not chunk_data['metadatas'] or not chunk_data['documents']:
        return []
    
    chunks = [Chunk.from_chroma(int(cid), dict(meta), text) for cid, meta, text in zip(chunk_data['ids'], chunk_data['metadatas'], chunk_data['documents'])]
    chunks.sort(key=lambda c: c.chunk_index)
    return chunks

def get_chunks_by_ids(chunk_ids: list[int]):
    if not chunk_ids:
        return []
    str_chunk_ids = [str(cid) for cid in chunk_ids]
    chunk_data = _chunks_collection.get(ids=str_chunk_ids, include=["metadatas", "documents"])
    
    chunks = []
    doc_ids_to_fetch = set()
    
    if chunk_data['ids'] and chunk_data['metadatas'] and chunk_data['documents']:
        for i, chunk_id_str in enumerate(chunk_data['ids']):
            metadata = chunk_data['metadatas'][i]
            text = chunk_data['documents'][i]
            chunk = Chunk.from_chroma(int(chunk_id_str), dict(metadata), text)
            chunks.append(chunk)
            if 'doc_id' in metadata:
                doc_ids_to_fetch.add(metadata['doc_id'])

    if doc_ids_to_fetch:
        str_doc_ids = [str(did) for did in doc_ids_to_fetch]
        docs_data = _documents_collection.get(ids=str_doc_ids)
        if docs_data['ids'] and docs_data['metadatas']:
            docs_map = {int(doc_id): Document.from_chroma(int(doc_id), dict(meta)) for doc_id, meta in zip(docs_data['ids'], docs_data['metadatas'])}
            
            for chunk in chunks:
                if chunk.doc_id in docs_map:
                    chunk.document = docs_map[chunk.doc_id]

    return chunks

def get_documents(limit: int = 100, offset: int = 0):
    all_docs_data = _documents_collection.get()
    if not all_docs_data['ids'] or not all_docs_data['metadatas']:
        return []
    docs = [Document.from_chroma(int(id), dict(meta)) for id, meta in zip(all_docs_data['ids'], all_docs_data['metadatas'])]
    docs.sort(key=lambda d: d.created_at, reverse=True)
    return docs[offset : offset + limit]

def get_document_chunk_counts(doc_ids: list[int]) -> dict[int, int]:
    if not doc_ids:
        return {}
    
    chunk_data = _chunks_collection.get(where={"doc_id": {"$in": doc_ids}}, include=['metadatas']) # type: ignore
    counts = {doc_id: 0 for doc_id in doc_ids}
    if chunk_data['metadatas']:
        for meta in chunk_data['metadatas']:
            doc_id = meta.get('doc_id')
            if doc_id in counts:
                counts[int(doc_id)] += 1
    return counts

def get_max_document_id():
    all_ids = _documents_collection.get(include=[])['ids']
    if not all_ids:
        return -1
    return max([int(i) for i in all_ids])

def get_all_chunks(limit: int = 200, offset: int = 0):
    all_ids = _chunks_collection.get(include=[])['ids']
    paginated_ids = sorted([int(i) for i in all_ids])[offset : offset + limit]
    
    if not paginated_ids:
        return []
        
    chunks = get_chunks_by_ids(paginated_ids)
    chunks.sort(key=lambda c: c.chunk_id)
    return chunks

def get_max_chunk_id():
    all_ids = _chunks_collection.get(include=[])['ids']
    if not all_ids:
        return -1
    return max([int(i) for i in all_ids])

def get_embedding_for_chunk(chunk_id: int):
    chunk_data = _chunks_collection.get(ids=[str(chunk_id)], include=["embeddings", "metadatas"])
    if not chunk_data['ids'] or not chunk_data['embeddings'] or not chunk_data['embeddings'][0] or not chunk_data['metadatas']:
        return None
    
    vector = chunk_data['embeddings'][0]
    meta = chunk_data['metadatas'][0]
    model = str(meta.get('embedding_model', 'unknown'))
    dim_val = meta.get('embedding_dim')
    dim = int(dim_val) if dim_val is not None else (len(vector) if vector else 0)
    
    float_vector = [float(v) for v in vector] if vector else []

    return Embedding(id=chunk_id, chunk_id=chunk_id, model=model, dim=dim, vector=float_vector)

def delete_document(doc_id: int):
    chunks_to_delete = _chunks_collection.get(where={"doc_id": doc_id})
    if chunks_to_delete['ids']:
        chunk_ids = [str(cid) for cid in chunks_to_delete['ids']]
        # Also delete corresponding embeddings from the external embeddings collection
        try:
            from .embeddings import get_chroma_collection  # local import to avoid cycles
            vec_coll = get_chroma_collection()
            try:
                vec_coll.delete(ids=chunk_ids)
            except Exception as e:  # pragma: no cover - defensive
                _logger.error("Failed to delete embeddings for chunk_ids=%s: %s", chunk_ids[:5], e)
        except Exception as e:  # pragma: no cover - env or import issues
            _logger.error("Could not access embeddings collection to delete vectors: %s", e)

        # Delete chunks (and their stored vectors in the chunks collection)
        _chunks_collection.delete(ids=chunk_ids)
    
    _documents_collection.delete(ids=[str(doc_id)])

def update_document(doc_id: int, updates: Dict[str, Any]) -> Optional[Document]:
    """Update metadata fields for a document.

    Only known metadata keys are updated. Returns the updated Document or None if not found.
    """
    allowed = {
        "title",
        "doc_type",
        "jurisdiction",
        "governing_law",
        "party_roles",
        "industry",
        "effective_date",
        "source_path",
    }
    try:
        existing = _documents_collection.get(ids=[str(doc_id)])
        if not existing['ids'] or not existing['metadatas']:
            return None
        meta = dict(existing['metadatas'][0])
        # Normalize list values to comma-separated strings
        def _norm(v):
            if isinstance(v, list):
                return ", ".join(str(x) for x in v)
            return v
        for k, v in updates.items():
            if k in allowed:
                if k == "effective_date" and isinstance(v, str):
                    # store as ISO string as-is; validation elsewhere
                    meta[k] = v
                else:
                    meta[k] = _norm(v)
        _documents_collection.update(ids=[str(doc_id)], metadatas=[meta])
        return Document.from_chroma(doc_id, meta)
    except Exception as e:  # pragma: no cover - defensive
        _logger.error(f"Failed to update document {doc_id}: {e}")
        return None

def clear_database():
    _client.delete_collection(name="documents")
    _client.delete_collection(name="chunks")
    init_db()

def ping():
    try:
        _client.heartbeat()
    except Exception as e:
        _logger.error(f"ChromaDB ping failed: {e}")
        raise

__all__ = [
    "init_db", "get_session", "get_db", "session_scope",
    "add_document", "add_chunk", "upsert_embedding",
    "get_document", "get_document_by_sha", "get_chunks_for_doc", "get_chunks_by_ids",
    "get_documents", "get_document_chunk_counts", "get_all_chunks", "get_max_chunk_id",
    "get_embedding_for_chunk", "delete_document", "clear_database", "update_document",
    "TOK_VER", "SEG_VER", "ping",
    "Document", "Chunk", "Embedding",
]
