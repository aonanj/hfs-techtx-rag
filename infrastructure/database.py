"""
ChromaDB database layer.
"""
import os
import contextlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict, Any
import numpy as np

from chromadb.utils import embedding_functions

try:
    from infrastructure.logger import get_logger  # reuse central logger if available
    _logger = get_logger()
except Exception:  # pragma: no cover - fallback minimal logger
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO)
    _logger = _logging.getLogger("db")

# ---------------------------------------------------------------------------
# Versioning for corpus mechanics
# ---------------------------------------------------------------------------
TOK_VER = int(os.getenv("TOK_VER", "1"))
SEG_VER = int(os.getenv("SEG_VER", "1"))

# ---------------------------------------------------------------------------
# ChromaDB client setup (with HuggingFace / read-only filesystem resilience)
# ---------------------------------------------------------------------------
CHROMA_PATH = os.getenv("CHROMA_PATH", "/data/chroma_db")

def _init_chroma_client():
    """Initialize a Chroma client with fallbacks.

    Order of attempts:
      1. Provided/ default CHROMA_PATH (relative or absolute)
      2. /tmp/chroma_db (writable in most containerized envs incl. HF Spaces)
      3. In-memory (non-persistent) client as last resort
    """
    candidates: list[Optional[str]] = [CHROMA_PATH, "/data/chroma_db"]
    tried: list[tuple[Optional[str], str]] = []

    for cand in candidates:
        try:
            if cand is None:
                _logger.warning("ChromaDB falling back to in-memory client (no persistence).")
                import chromadb
                return chromadb.Client()  # type: ignore
            # Ensure directory exists and is writable
            abs_path = os.path.abspath(cand)
            os.makedirs(abs_path, exist_ok=True)
            test_file = os.path.join(abs_path, ".write_test")
            with open(test_file, "a", encoding="utf-8") as tf:
                tf.write("ok")
            os.remove(test_file)
            _logger.info(f"Initializing Chroma PersistentClient at {abs_path}")
            import chromadb
            return chromadb.PersistentClient(path=abs_path)  # type: ignore
        except Exception as e:  # pragma: no cover - environment specific
            tried.append((cand, str(e)))
            # Continue to next candidate
            continue

    _logger.error("Failed to initialize persistent Chroma client; attempts=%s. Using in-memory fallback.", tried)
    import chromadb
    return chromadb.Client()  # type: ignore

_client = _init_chroma_client()

# ---------------------------------------------------------------------------
# Ensure writable HOME / cache dirs (HF Spaces & read-only root fix)
# ---------------------------------------------------------------------------
def _ensure_writable_caches():  # pragma: no cover (env specific)
    try:
        current_home = os.path.expanduser("~")
        # If home resolves to root or isn't writable, pick a fallback
        if current_home == "/" or not os.access(current_home, os.W_OK):
            for cand in ["/data", "/home/user", "/tmp"]:
                try:
                    os.makedirs(cand, exist_ok=True)
                    test_path = os.path.join(cand, ".home_write_test")
                    with open(test_path, "w", encoding="utf-8") as f:
                        f.write("ok")
                    os.remove(test_path)
                    os.environ["HOME"] = cand
                    _logger.warning(f"Reset HOME to writable directory: {cand}")
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
        _logger.warning(f"Cache/HOME setup skipped due to error: {e}")

_ensure_writable_caches()

# Using a default embedding function for the collections, though we'll be providing our own embeddings.
# This is required by ChromaDB.
# The type hint for embedding_function is complex, and mypy has issues with it.
# Using type: ignore to suppress the error.
default_ef = embedding_functions.DefaultEmbeddingFunction()

_documents_collection = _client.get_or_create_collection(
    name="documents",
    embedding_function=default_ef # type: ignore
)
_chunks_collection = _client.get_or_create_collection(
    name="chunks",
    embedding_function=default_ef # type: ignore
)

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
    global _documents_collection, _chunks_collection
    _documents_collection = _client.get_or_create_collection(name="documents", embedding_function=default_ef) # type: ignore
    _chunks_collection = _client.get_or_create_collection(name="chunks", embedding_function=default_ef) # type: ignore
    _logger.info("ChromaDB collections ensured.")


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

    _documents_collection.add(ids=[str(new_doc_id)], metadatas=[metadata], documents=[title or ''])
    return Document(doc_id=new_doc_id, **metadata)

def add_chunk(doc_id: int, chunk_id: int, text: str, chunk_index: int, token_count: int | None = None,
              page_start: int | None = None, page_end: int | None = None, section_number: str | None = None,
              section_title: str | None = None, clause_type: str | None = None, tok_ver: int = TOK_VER, seg_ver: int = SEG_VER):
    
    metadata: Dict[str, Any] = {
        "doc_id": doc_id, "chunk_index": chunk_index, "token_count": token_count,
        "page_start": page_start, "page_end": page_end, "section_number": section_number,
        "section_title": section_title, "clause_type": clause_type, "tok_ver": tok_ver, "seg_ver": seg_ver,
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    _chunks_collection.add(ids=[str(chunk_id)], documents=[text], metadatas=[metadata])
    _logger.info("DB added chunk %s for doc_id=%s", chunk_id, doc_id)
    
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, chunk_index=chunk_index, **metadata)

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
        _chunks_collection.delete(ids=chunks_to_delete['ids'])
    
    _documents_collection.delete(ids=[str(doc_id)])

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
    "get_embedding_for_chunk", "delete_document", "clear_database",
    "TOK_VER", "SEG_VER", "ping",
    "Document", "Chunk", "Embedding",
]
