# api.py
import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from infrastructure.logger import get_logger

from infrastructure.database import (
    init_db,
    add_document,
    get_document_by_sha,
    get_chunks_for_doc,
    get_documents,
    get_document_chunk_counts,
    get_all_chunks,
)
from infrastructure.document_processor import (
    extract_text,
    extract_title_type_jurisdiction,
    sha256_text,
)
from infrastructure.chunker import chunk_doc
from infrastructure.vector_search import find_nearest_neighbors
from infrastructure.database import get_chunks_by_ids

api_bp = Blueprint("api", __name__, url_prefix="/api")
logger = get_logger()

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/data/corpus_raw")
CLEAN_FOLDER = os.getenv("CLEAN_FOLDER", "/data/corpus_clean")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEAN_FOLDER, exist_ok=True)

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
    jurisdiction = request.form.get("jurisdiction") or None

    logger.info("Received upload: filename=%s, doc_type=%s, jurisdiction=%s",
                file.filename, doc_type, jurisdiction)

    clean_text = extract_text(file=file)
    content_sha = sha256_text(clean_text)

    doc = get_document_by_sha(content_sha)
    title = None
    doc_id_val = None

    if not doc:
        filename = secure_filename(file.filename)
        raw_path = os.path.join(UPLOAD_FOLDER, filename)
        txt_filename = filename.rsplit('.', 1)[0] + '.txt'
        clean_path = os.path.join(CLEAN_FOLDER, txt_filename)

        file.stream.seek(0)
        file.save(raw_path)
        logger.info("Saved raw file to %s", raw_path)

        with open(clean_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        logger.info("Saved clean text to %s", clean_path)

        doc_info = extract_title_type_jurisdiction(clean_text)
        title = doc_info.get("title") or file.filename.split('.')[0]
        if not doc_type and "doc_type" in doc_info:
            doc_type = doc_info.get("doc_type")
        if not jurisdiction and "jurisdiction" in doc_info:
            jurisdiction = doc_info.get("jurisdiction")
        
        doc = add_document(sha256=content_sha, title=title, source_path=raw_path, doc_type=doc_type, jurisdiction=jurisdiction)
        doc_id_val = getattr(doc, "doc_id", None)

    else:
        doc_id_val = getattr(doc, "doc_id", None)
        title = getattr(doc, "title", None)
        try:
            doc = add_document(sha256=content_sha, doc_type=doc_type, jurisdiction=jurisdiction)
        except Exception:
            pass

    try:
        chunks = chunk_doc(text=clean_text, doc_id=str(doc_id_val))
        logger.info("Chunked document into %d chunks", len(chunks))
    except Exception as e:
        logger.exception("Chunking failed: %s", e)
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
    out = [{
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "chunk_index": c.chunk_index,
        "token_count": c.token_count,
        "tok_ver": c.tok_ver,
        "seg_ver": c.seg_ver,
        "text": str(getattr(c, "text", "") or ""),
    } for c in chunks]
    return jsonify({"doc_id": doc_id, "count": len(out), "chunks": out}), 200

@api_bp.route("/documents", methods=["GET"])
def list_documents():
    """List documents with pagination and basic counts.

    Query params:
      limit (default 100, max 500)
      offset (default 0)
    """
    try:
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"error": "invalid pagination"}), 400
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    docs = get_documents(limit=limit, offset=offset)
    ids = [int(getattr(d, 'doc_id')) for d in docs]
    counts = get_document_chunk_counts(ids)
    out = []
    for d in docs:
        out.append({
            "doc_id": int(getattr(d, 'doc_id')),
            "title": d.title,
            "sha256": d.sha256,
            "doc_type": getattr(d, "doc_type", None),
            "jurisdiction": getattr(d, "jurisdiction", None),
            "created_at": d.created_at.isoformat() if getattr(d, 'created_at', None) else None,
            "chunk_count": counts.get(int(getattr(d, 'doc_id')), 0),
        })
    return jsonify({"documents": out, "count": len(out), "limit": limit, "offset": offset}), 200


@api_bp.route("/chunks/all", methods=["GET"])
def list_all_chunks():
    """List global chunks slice (paginated). Query params: limit (<=500), offset."""
    try:
        limit = int(request.args.get("limit", 200))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"error": "invalid pagination"}), 400
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    rows = get_all_chunks(limit=limit, offset=offset)
    out = []
    for c in rows:
        doc = getattr(c, 'document', None)
        out.append({
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "token_count": c.token_count,
            "tok_ver": c.tok_ver,
            "seg_ver": c.seg_ver,
            "text": str(getattr(c, 'text', '') or ''),
            "doc_type": getattr(doc, 'doc_type', None) if doc else None,
            "jurisdiction": getattr(doc, 'jurisdiction', None) if doc else None,
            "title": getattr(doc, 'title', None) if doc else None,
        })
    return jsonify({"chunks": out, "count": len(out), "limit": limit, "offset": offset}), 200

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

        chunk_ids = [int(nid) for nid, dist in neighbors]
        chunks = get_chunks_by_ids(chunk_ids)
        
        # Create a dictionary for quick lookup
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        
        results = []
        for neighbor_id, distance in neighbors:
            chunk = chunk_map.get(int(neighbor_id))
            if chunk:
                doc_info = {}
                if chunk.document:
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
                    "document": doc_info
                })
        
        return jsonify({"results": results}), 200
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
