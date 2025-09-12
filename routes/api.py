# api.py
import os
from flask import Blueprint, request, jsonify, send_from_directory
import json
from flask_cors import CORS
from werkzeug.utils import secure_filename
from infrastructure.logger import get_logger
from datetime import datetime

from infrastructure.database import (
    init_db,
    add_document,
    get_document_by_sha,
    get_chunks_for_doc,
    get_documents,
    get_document_chunk_counts,
    get_all_chunks,
    get_embedding_counts_for_chunks,
    full_reset_chroma,
    get_chunks_by_ids,
    update_document,
    update_chunk,
    delete_document,
)
from infrastructure.document_processor import (
    extract_text,
    extract_title_type_jurisdiction,
    sha256_text,
    upsert_manifest_record,
    get_manifest_info
)
from infrastructure.chunker import chunk_doc
from infrastructure.vector_search import find_nearest_neighbors
from services.gpt_service import refine_query_response

api_bp = Blueprint("api", __name__, url_prefix="/api")
CORS(api_bp)
logger = get_logger()
RESET_PASSWORD = os.getenv("RESET_PASSWORD")

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/data/corpus_raw")
CLEAN_FOLDER = os.getenv("CLEAN_FOLDER", "/data/corpus_clean")
MANIFEST_DIR = os.getenv("MANIFEST_DIR", "/data/manifest")
RAW_PREFIX = "/data/corpus_raw"
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "/data/chunks")

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@api_bp.route("/upload", methods=["POST"])
def add_doc():
    """Upload one document, extract text, persist metadata, and schedule chunking."""

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    doc_type = request.form.get("doc_type") or None

    logger.info("Received upload: filename=%s, doc_type=%s",
                file.filename, doc_type)

    clean_text = extract_text(file=file)
    content_sha = sha256_text(clean_text)
    doc = get_document_by_sha(content_sha)
    title = None
    doc_id_val = None


    if not doc:
        filename = secure_filename(file.filename)

        doc_info = extract_title_type_jurisdiction(clean_text)
        title = doc_info.get("title") or file.filename.split('.')[0]
        jurisdiction = None
        if not doc_type and "doc_type" in doc_info:
            doc_type = doc_info.get("doc_type")
        if "jurisdiction" in doc_info:
            jurisdiction = doc_info.get("jurisdiction")
        
        manifest_record = get_manifest_info(clean_text)
        party_roles = manifest_record.get("party_roles") or None
        governing_law = manifest_record.get("governing_law") or None
        effective_date = manifest_record.get("effective_date") or None
        industry = manifest_record.get("industry") or None

        doc = add_document(sha256=content_sha, title=title, source_path=None, doc_type=doc_type, jurisdiction=jurisdiction, industry=industry, party_roles=party_roles, governing_law=governing_law, effective_date=effective_date)
        doc_id_val = getattr(doc, "doc_id", 0)

        file_format = file.mimetype
        file.stream.seek(0, 2)
        file_size = file.tell()
        file.stream.seek(0)

        raw_filename = f"{str(doc_id_val)}" + "." + filename.rsplit('.', 1)[1].lower()
        raw_path = os.path.join(UPLOAD_FOLDER, raw_filename)

        txt_filename = f"{str(doc_id_val)}" + '.txt'
        clean_path = os.path.join(CLEAN_FOLDER, txt_filename)

        file.stream.seek(0)
        file.save(raw_path)
        logger.info("Saved raw file to %s", raw_path)

        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        logger.info("Saved clean text to %s", clean_path)

        doc = update_document(doc_id=doc_id_val, updates={"source_path": raw_path})

        upsert_manifest_record(
            sha256=content_sha,
            title=title,
            source_path=raw_path,
            clean_path=clean_path,
            doc_type=doc_type or "",
            jurisdiction=jurisdiction or "",
            party_roles=party_roles or "",
            governing_law=governing_law or "",
            effective_date=effective_date or "",
            industry=industry or "",
            doc_id=doc_id_val,
            text=clean_text,
            size=str(file_size),
            content_type=file_format,
        )

        logger.error("INFO: Added document id=%s, sha256=%s, title=%s, doc_type=%s, jurisdiction=%s", doc_id_val, content_sha, title, doc_type, jurisdiction)

        try:
            chunks = chunk_doc(text=clean_text, doc_id=doc_id_val)
            logger.info("Chunked document into %d chunks", len(chunks))
        except Exception as e:
            logger.exception("Chunking failed: %s", e)
            pass

    else:
        doc_id_val = getattr(doc, "doc_id", 0)
        title = getattr(doc, "title", None)
        jurisdiction = getattr(doc, "jurisdiction", None)
        try:
            doc = add_document(sha256=content_sha, doc_type=doc_type, jurisdiction=jurisdiction)
        except Exception:
            pass

    return jsonify({
        "message": "ok",
        "doc_id": doc_id_val,
        "sha256": content_sha,
        "title": title,
        "doc_type": getattr(doc, "doc_type", None),
        "jurisdiction": getattr(doc, "jurisdiction", None)
    }), 200


@api_bp.route("/chunks", methods=["GET"])
def get_chunks():
    """List chunks for a document: /chunks?doc_id=123"""
    try:
        doc_id = int(request.args.get("doc_id", "0"))
    except ValueError:
        return jsonify({"error": "invalid doc_id"}), 400
    if not doc_id:
        return jsonify({"error": "doc_id required"}), 400

    chunks = get_chunks_for_doc(doc_id)
    if not chunks:
        return jsonify({"doc_id": doc_id, "count": 0, "chunks": []}), 200

    # Populate embedding counts to align with global chunks view
    try:
        chunk_ids = [c.chunk_id for c in chunks]
        embedding_counts = get_embedding_counts_for_chunks(chunk_ids)
    except Exception:
        embedding_counts = {}

    out = []
    for c in chunks:
        out.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
            "tok_ver": c.tok_ver,
            "seg_ver": c.seg_ver,
            "text": str(getattr(c, 'text', '') or ''),
            "page_start": c.page_start,
            "page_end": c.page_end,
            "section_number": c.section_number,
            "section_title": c.section_title,
            "clause_type": c.clause_type,
            "path": c.path,
            "numbers_present": c.numbers_present,
            "definition_terms": c.definition_terms,
            "embedding_count": embedding_counts.get(c.chunk_id, 0),
        })

    return jsonify({"doc_id": doc_id, "count": len(out), "chunks": out}), 200

@api_bp.route("/documents", methods=["GET"])
def list_documents():
    """List documents with pagination and document metadata.

    Query params:
        limit (default 100, max 500)
        offset (default 0)
        
    Returns documents with keys: doc_id, sha_256, title, source_path, created_at,
    docu_type, jurisdiction, governing_law, party_roles, industry, effective_date, chunks.
    """
    try:
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"error": "invalid pagination"}), 400
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    
    docs = get_documents(limit=limit, offset=offset)
    
    if not docs:
        return jsonify({"documents": [], "count": 0, "limit": limit, "offset": offset}), 200
    
    # Get chunk counts for all documents
    ids = [int(getattr(d, 'doc_id')) for d in docs]
    counts = get_document_chunk_counts(ids)
    
    out = []
    for d in docs:
        effective_date = getattr(d, 'effective_date', None)
        if effective_date is not None and isinstance(effective_date, datetime):
            effective_date_str = str(effective_date.isoformat())
        else:
            effective_date_str = str(effective_date) if effective_date else None
        doc_data = {
            "doc_id": int(getattr(d, 'doc_id')),
            "sha256": d.sha256,
            "title": d.title,
            "source_path": d.source_path,
            "created_at": d.created_at.isoformat() if getattr(d, 'created_at', None) else None,
            "doc_type": getattr(d, "doc_type", None),
            "jurisdiction": getattr(d, "jurisdiction", None),
            "governing_law": getattr(d, "governing_law", None),
            "party_roles": getattr(d, "party_roles", None),
            "industry": getattr(d, "industry", None),
            "effective_date": effective_date_str or None,
            "chunk_count": counts.get(int(getattr(d, 'doc_id')), 0),
        }
        out.append(doc_data)
    
    return jsonify({"documents": out, "count": len(out), "limit": limit, "offset": offset}), 200


@api_bp.route("/chunks/all", methods=["GET"])
def list_all_chunks():
    """List global chunks slice (paginated) with specified fields and embedding counts.
    
    Query params: 
        limit (<=500, default 200)
        offset (default 0)
        
    Returns chunks with keys: chunk_id, doc_id, chunk_index, token_count, tok_ver, 
    seg_ver, text, page_start, page_end, section_number, section_title, clause_type, 
    path, numbers_present, definition_terms, embedding_count.
    """
    try:
        limit = int(request.args.get("limit", 200))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"error": "invalid pagination"}), 400
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    
    rows = get_all_chunks(limit=limit, offset=offset)
    
    if not rows:
        return jsonify({"chunks": [], "count": 0, "limit": limit, "offset": offset}), 200
    
    # Get embedding counts for all chunks
    chunk_ids = [c.chunk_id for c in rows]
    embedding_counts = get_embedding_counts_for_chunks(chunk_ids)
    
    out = []
    for c in rows:
        chunk_data = {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
            "tok_ver": c.tok_ver,
            "seg_ver": c.seg_ver,
            "text": str(getattr(c, 'text', '') or ''),
            "page_start": c.page_start,
            "page_end": c.page_end,
            "section_number": c.section_number,
            "section_title": c.section_title,
            "clause_type": c.clause_type,
            "path": c.path,
            "numbers_present": c.numbers_present,
            "definition_terms": c.definition_terms,
            "embedding_count": embedding_counts.get(c.chunk_id, 0),
        }
        out.append(chunk_data)
    
    return jsonify({"chunks": out, "count": len(out), "limit": limit, "offset": offset}), 200


@api_bp.route("/chunks/jsonl", methods=["GET"])
def list_chunks_from_jsonl():
    """Return all chunk records found in /data/chunks/chunks.jsonl for viewing.

    This endpoint parses the JSONL (or JSON-array) file produced by the chunker
    and normalizes records to the same shape used in the table UI. It also
    augments with embedding_count using the ChromaDB store when chunk IDs exist.
    """

    path = os.path.join(CHUNKS_DIR, "chunks.jsonl")
    rows: list[dict] = []

    if not os.path.exists(path):
        return jsonify({"chunks": [], "count": 0}), 200

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.error("Failed reading %s: %s", path, e)
        return jsonify({"error": "failed to read chunks.jsonl"}), 500

    content_stripped = (content or "").strip()
    # Try robust parsing:
    # 1) Whole-file JSON (array or object)
    # 2) Streaming JSON decoder to handle concatenated JSON values
    # 3) Line-by-line JSONL fallback
    parsed_ok = False
    if content_stripped:
        try:
            parsed = json.loads(content_stripped)
            if isinstance(parsed, list):
                rows = [r for r in parsed if isinstance(r, dict)]
            elif isinstance(parsed, dict):
                rows = [parsed]
            parsed_ok = True
        except Exception:
            parsed_ok = False

    if not parsed_ok and content_stripped:
        try:
            dec = json.JSONDecoder()
            s = content_stripped
            idx = 0
            out: list[dict] = []
            while idx < len(s):
                # Skip whitespace between values
                while idx < len(s) and s[idx].isspace():
                    idx += 1
                if idx >= len(s):
                    break
                obj, end = dec.raw_decode(s, idx)
                if isinstance(obj, list):
                    out.extend([r for r in obj if isinstance(r, dict)])
                elif isinstance(obj, dict):
                    out.append(obj)
                idx = end
            if out:
                rows = out
                parsed_ok = True
        except Exception:
            parsed_ok = False

    if not parsed_ok:
        # Fallback: line-by-line JSONL
        rows = []
        for line in (content.splitlines() if content else []):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    rows.extend([r for r in obj if isinstance(r, dict)])
                elif isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                # tolerate malformed lines
                continue

    if not rows:
        return jsonify({"chunks": [], "count": 0}), 200

    # Normalize shape and collect IDs for embedding counts
    normalized: list[dict] = []
    chunk_ids: list[int] = []
    for r in rows:
        cid = r.get("chunk_id") or r.get("id")
        try:
            cid_int = int(cid) if cid is not None else None
        except Exception:
            cid_int = None
        if cid_int is not None:
            chunk_ids.append(cid_int)
        normalized.append({
            "chunk_id": cid_int,
            "doc_id": r.get("doc_id"),
            "chunk_index": r.get("chunk_index") or r.get("index"),
            "token_count": r.get("token_count"),
            "tok_ver": r.get("tok_ver"),
            "seg_ver": r.get("seg_ver"),
            "text": str(r.get("text") or ""),
            "page_start": r.get("page_start"),
            "page_end": r.get("page_end"),
            "section_number": r.get("section_number"),
            "section_title": r.get("section_title"),
            "clause_type": r.get("clause_type"),
            "path": r.get("path"),
            "numbers_present": r.get("numbers_present"),
            "definition_terms": r.get("definition_terms"),
            # embedding_count filled below
        })

    # Populate embedding counts where possible
    embedding_counts: dict[int, int] = {}
    try:
        if chunk_ids:
            embedding_counts = get_embedding_counts_for_chunks(chunk_ids)
    except Exception as e:
        logger.error("Failed to get embedding counts for JSONL rows: %s", e)
        embedding_counts = {}

    for item in normalized:
        cid = item.get("chunk_id")
        item["embedding_count"] = int(embedding_counts.get(int(cid), 0)) if isinstance(cid, int) else 0

    # Sort by chunk_id asc when available
    normalized.sort(key=lambda x: (x["chunk_id"] is None, x.get("chunk_id") or 0))

    return jsonify({"chunks": normalized, "count": len(normalized)}), 200

@api_bp.route("/chunks", methods=["PATCH"])
def update_chunks():
    """Update a chunk's metadata and optionally regenerate embedding if text changed."""
    payload = request.get_json(force=True) or {}
    chunk_id = payload.get("chunk_id")
    updates = payload.get("updates", {})
    
    if not chunk_id:
        return jsonify({"error": "chunk_id required"}), 400
    
    try:
        chunk_id = int(chunk_id)
    except (ValueError, TypeError):
        return jsonify({"error": "invalid chunk_id"}), 400
    
    if not updates or not isinstance(updates, dict):
        return jsonify({"error": "updates dict required"}), 400
    
    logger.info("Updating chunk %d with updates: %s", chunk_id, updates)
    
    try:
        updated_chunk = update_chunk(chunk_id, updates)
        if not updated_chunk:
            return jsonify({"error": "chunk not found"}), 404
        
        # If text was updated, we should regenerate the embedding using ChromaDB
        if "text" in updates:
            from infrastructure.embeddings import get_chroma_collection
            
            try:
                new_text = updates["text"]
                if new_text and new_text.strip():
                    # Update text and regenerate embedding via ChromaDB
                    collection = get_chroma_collection()
                    collection.update(
                        ids=[str(chunk_id)],
                        documents=[new_text]
                    )
                    logger.info("Regenerated embedding for chunk %d after text update", chunk_id)
            except Exception as e:
                logger.error("Failed to regenerate embedding for chunk %d: %s", chunk_id, e)
                # Don't fail the request, just log the error
        
        return jsonify({
            "message": "chunk updated successfully", 
            "chunk_id": chunk_id,
            "updated_fields": list(updates.keys())
        }), 200
        
    except Exception as e:
        logger.error("Failed to update chunk %d: %s", chunk_id, e)
        return jsonify({"error": "internal server error"}), 500

@api_bp.route("/query", methods=["POST"])
def query():
    """Vector search against chunk embeddings using ChromaDB."""
    payload = request.get_json(force=True) or {}
    question = payload.get("query")
    top_k = int(payload.get("top_k", 5))
    if not question:
        return jsonify({"error": "query required"}), 400

    try:
        neighbors = find_nearest_neighbors(query=question, num_neighbors=top_k)

        if not neighbors:
            return jsonify({"results": [], "message": "No results found."}), 200

        results = []

        # Handle both dict-based neighbors (current) and tuple-based (legacy) formats
        if isinstance(neighbors[0], dict):
            # Current format from vector_search.find_nearest_neighbors
            chunk_ids: list[int] = []
            for n in neighbors:
                val = n.get("id") if isinstance(n, dict) else None
                if val is None:
                    continue
                try:
                    chunk_ids.append(int(val))
                except (ValueError, TypeError):
                    continue

            chunks = get_chunks_by_ids(chunk_ids)
            chunk_map = {int(c.chunk_id): c for c in chunks}

            for n in neighbors:
                val = n.get("id") if isinstance(n, dict) else None
                if val is None:
                    continue
                try:
                    cid = int(val)
                except (ValueError, TypeError):
                    continue
                chunk = chunk_map.get(cid)
                if not chunk:
                    continue
                doc_info = {}
                if getattr(chunk, "document", None):
                    doc_info = {
                        "title": chunk.document.title,
                        "doc_type": chunk.document.doc_type,
                        "jurisdiction": chunk.document.jurisdiction,
                    }
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "distance": n.get("distance"),
                    "document": doc_info,
                })
        else:
            # Legacy: list of (id, distance)
            try:
                chunk_ids = [int(nid) for nid, _ in neighbors]
            except Exception:
                chunk_ids = []
            chunks = get_chunks_by_ids(chunk_ids)
            chunk_map = {int(c.chunk_id): c for c in chunks}
            for neighbor_id, distance in neighbors:
                try:
                    cid = int(neighbor_id)
                except Exception:
                    continue
                chunk = chunk_map.get(cid)
                if not chunk:
                    continue
                doc_info = {}
                if getattr(chunk, "document", None):
                    doc_info = {
                        "title": chunk.document.title,
                        "doc_type": chunk.document.doc_type,
                        "jurisdiction": chunk.document.jurisdiction,
                    }
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "distance": distance,
                    "document": doc_info,
                })

        response = refine_query_response(question, results)
        logger.error(f"Refined response: {response}")
        logger.error(f"JSONified return: {jsonify({'results': results, 'response': response})}")

        return jsonify({"results": results, "response": response}), 200
    except Exception as e:
        logger.exception("Query failed: %s", e)
        return jsonify({"error": "Failed to execute query"}), 500
    



# Health
@api_bp.route("/healthz", methods=["GET"])
def healthz():
    try:
        init_db()  # idempotent
        return jsonify({"ok": True}), 200
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return jsonify({"status": "error", "reason": str(e)}), 500


@api_bp.route("/reset", methods=["POST"])
def reset_system():
    """Dangerous: wipes /data directory (recursively) and clears ChromaDB collections.

    Optional JSON body: {"confirm": true}
    Reject unless confirm flag present to reduce accidental invocation.
    """
    # If a RESET_PASSWORD env var is set, require a matching password
    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}
    if RESET_PASSWORD:
        supplied = payload.get("password") or request.headers.get("X-Reset-Password")
        if not supplied or supplied != RESET_PASSWORD:
            return jsonify({"error": "forbidden"}), 403

    data_root = "/data"
    erased_paths: list[str] = []
    try:
        for root, dirs, files in os.walk(data_root, topdown=False):
            for name in files:
                fp = os.path.join(root, name)
                try:
                    os.remove(fp)
                    erased_paths.append(fp)
                except Exception:
                    pass
            for name in dirs:
                dp = os.path.join(root, name)
                # Skip root itself; remove only if empty now
                try:
                    if os.path.isdir(dp):
                        os.rmdir(dp)
                        erased_paths.append(dp + "/")
                except Exception:
                    pass
    except Exception as e:
        logger.exception("Filesystem wipe encountered errors: %s", e)

    chroma_ok = False
    try:
        chroma_ok = full_reset_chroma()
    except Exception as e:
        logger.exception("Chroma reset failed: %s", e)
        chroma_ok = False

    return jsonify({
        "message": "reset completed",
        "files_deleted": len([p for p in erased_paths if not p.endswith('/')]),
        "dirs_deleted": len([p for p in erased_paths if p.endswith('/')]),
        "chroma_reset": chroma_ok,
    }), 200


# -------- Manifest JSONL helpers and endpoints --------
def _manifest_path() -> str:
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    return os.path.join(MANIFEST_DIR, "manifest.jsonl")

def _read_manifest() -> list[dict]:
    path = _manifest_path()
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(__import__("json").loads(line))
                except Exception:
                    continue
    except Exception as e:
        logger.error("Failed reading manifest: %s", e)
    return rows

def _write_manifest(rows: list[dict]) -> bool:
    path = _manifest_path()
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(__import__("json").dumps(r) + "\n")
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        logger.error("Failed writing manifest: %s", e)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


@api_bp.route("/manifest", methods=["GET"])
def manifest_get():
    rows = _read_manifest()
    # minimal projection for UI: include keys asked by user, but pass through extra fields too
    return jsonify(rows), 200


@api_bp.route("/manifest", methods=["PATCH"])
def manifest_patch():
    payload = request.get_json(force=True) or {}
    doc_id = payload.get("doc_id")
    updates = payload.get("updates") or {}
    if doc_id is None:
        return jsonify({"error": "doc_id required"}), 400
    try:
        did = int(doc_id)
    except Exception:
        return jsonify({"error": "invalid doc_id"}), 400

    # Fields not editable: doc_id, source_path
    updates.pop("doc_id", None)
    updates.pop("source_path", None)

    rows = _read_manifest()
    found = False
    for r in rows:
        try:
            rid = int(str(r.get("doc_id")))
        except Exception:
            continue
        if rid == did:
            for k, v in updates.items():
                r[k] = v
            found = True
            break
    if not found:
        return jsonify({"error": "doc not found"}), 404

    if not _write_manifest(rows):
        return jsonify({"error": "failed to persist manifest"}), 500

    # Update DB metadata too
    db_doc = update_document(did, updates)
    if db_doc is None:
        # Not fatal for manifest, but report
        logger.error("update_document failed for doc_id=%s", did)

    return jsonify({"ok": True}), 200


@api_bp.route("/manifest", methods=["DELETE"])
def manifest_delete():
    payload = request.get_json(force=True) or {}
    doc_id = payload.get("doc_id")
    if doc_id is None:
        return jsonify({"error": "doc_id required"}), 400
    try:
        did = int(doc_id)
    except Exception:
        return jsonify({"error": "invalid doc_id"}), 400

    rows = _read_manifest()
    new_rows: list[dict] = []
    record: dict | None = None
    for r in rows:
        try:
            rid = int(str(r.get("doc_id")))
        except Exception:
            new_rows.append(r)
            continue
        if rid == did:
            record = r
            continue
        new_rows.append(r)

    if record is None:
        return jsonify({"error": "doc not found"}), 404

    # Persist manifest removal
    if not _write_manifest(new_rows):
        return jsonify({"error": "failed to persist manifest"}), 500

    # Delete from DB
    try:
        delete_document(did)
    except Exception as e:
        logger.error("Failed deleting document %s from DB: %s", did, e)

    # Delete files from corpus_raw and corpus_clean
    for key in ("source_path", "clean_path"):
        p = record.get(key)
        if p and isinstance(p, str):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                logger.error("Failed removing %s: %s", p, e)

    return jsonify({"ok": True}), 200


@api_bp.route("/raw/<path:filename>", methods=["GET"])
def serve_raw(filename: str):
    """Serve files from /data/corpus_raw for clicking source_path links in UI.

    Construct links as: /api/raw/<relative-to-RAW_PREFIX>
    Example: if source_path is /data/corpus_raw/123.pdf, link to /api/raw/123.pdf
    """
    return send_from_directory(RAW_PREFIX, filename)


@api_bp.route("/dbviewer/documents", methods=["GET"])
def dbviewer_documents():
    """List documents for the database viewer with pagination.
    
    Query params:
        page (default 1)
        limit (default 25, max 100)
        
    Returns documents with pagination info.
    """
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 25))
    except ValueError:
        return jsonify({"error": "invalid pagination parameters"}), 400
    
    page = max(1, page)
    limit = max(1, min(limit, 100))
    offset = (page - 1) * limit
    
    docs = get_documents(limit=limit, offset=offset)
    
    if not docs:
        return jsonify({
            "documents": [], 
            "page": page, 
            "limit": limit, 
            "total": 0,
            "total_pages": 0
        }), 200
    
    # Get chunk counts for all documents
    ids = [int(getattr(d, 'doc_id')) for d in docs]
    counts = get_document_chunk_counts(ids)
    
    # Get total count for pagination (this is approximate)
    # We'll fetch one more page to see if there are more documents
    next_docs = get_documents(limit=1, offset=offset + limit)
    has_more = len(next_docs) > 0
    
    out = []
    for d in docs:
        effective_date = d.effective_date
        if effective_date is not None and isinstance(effective_date, datetime):
            effective_date = effective_date.isoformat()
        elif effective_date is not None and isinstance(effective_date, str):
            try:
                effective_date = datetime.fromisoformat(effective_date).isoformat()
            except Exception:
                effective_date = None
        doc_data = {
            "doc_id": int(getattr(d, 'doc_id')),
            "sha256": d.sha256,
            "title": d.title,
            "source_path": d.source_path,
            "created_at": d.created_at.isoformat() if getattr(d, 'created_at', None) else None,
            "doc_type": getattr(d, "doc_type", None),
            "jurisdiction": getattr(d, "jurisdiction", None),
            "governing_law": getattr(d, "governing_law", None),
            "party_roles": getattr(d, "party_roles", None),
            "industry": getattr(d, "industry", None),
            "effective_date": effective_date or None,
            "chunk_count": counts.get(int(getattr(d, 'doc_id')), 0),
        }
        out.append(doc_data)
    
    # Estimate total for pagination
    estimated_total = offset + len(out) + (100 if has_more else 0)
    estimated_pages = max(1, (estimated_total + limit - 1) // limit)
    
    return jsonify({
        "documents": out, 
        "page": page, 
        "limit": limit, 
        "count": len(out),
        "has_more": has_more,
        "estimated_total": estimated_total,
        "estimated_pages": estimated_pages
    }), 200


@api_bp.route("/dbviewer/chunks", methods=["GET"])
def dbviewer_chunks():
    """List chunks for the database viewer with pagination.
    
    Query params:
        page (default 1)
        limit (default 25, max 100)
        
    Returns chunks with pagination info.
    """
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 25))
    except ValueError:
        return jsonify({"error": "invalid pagination parameters"}), 400
    
    page = max(1, page)
    limit = max(1, min(limit, 100))
    offset = (page - 1) * limit
    
    rows = get_all_chunks(limit=limit, offset=offset)
    
    if not rows:
        return jsonify({
            "chunks": [], 
            "page": page, 
            "limit": limit, 
            "total": 0,
            "total_pages": 0
        }), 200
    
    # Get embedding counts for all chunks
    chunk_ids = [c.chunk_id for c in rows]
    embedding_counts = get_embedding_counts_for_chunks(chunk_ids)
    
    # Check if there are more chunks
    next_chunks = get_all_chunks(limit=1, offset=offset + limit)
    has_more = len(next_chunks) > 0
    
    out = []
    for c in rows:
        chunk_data = {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
            "tok_ver": c.tok_ver,
            "seg_ver": c.seg_ver,
            "text": str(getattr(c, 'text', '') or ''),
            "page_start": c.page_start,
            "page_end": c.page_end,
            "section_number": c.section_number,
            "section_title": c.section_title,
            "clause_type": c.clause_type,
            "path": c.path,
            "numbers_present": c.numbers_present,
            "definition_terms": c.definition_terms,
            "embedding_count": embedding_counts.get(c.chunk_id, 0),
        }
        out.append(chunk_data)
    
    # Estimate total for pagination
    estimated_total = offset + len(out) + (100 if has_more else 0)
    estimated_pages = max(1, (estimated_total + limit - 1) // limit)
    
    return jsonify({
        "chunks": out, 
        "page": page, 
        "limit": limit, 
        "count": len(out),
        "has_more": has_more,
        "estimated_total": estimated_total,
        "estimated_pages": estimated_pages
    }), 200
