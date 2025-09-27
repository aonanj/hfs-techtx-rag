from openai import OpenAI
import fitz
from .logger import get_logger
import os
import io
import re
from PIL import Image
import pytesseract
import hashlib
import docx
from typing import Optional, Dict, Any
import json
from werkzeug.datastructures import FileStorage
from datetime import datetime, timezone

logger = get_logger()

API_KEY = os.getenv("OPENAI_API_KEY")
MANIFEST_DIR = os.getenv("MANIFEST_DIR", "/data/manifest")
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "/data/chunks")

def ocr_page(page: fitz.Page, zoom: float = 2.0) -> str:
    logger.info(f"Performing OCR on page {page.number}")
    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=mat, alpha=False)  # type: ignore
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)

def blocks_text(tp: fitz.TextPage) -> str:
    blocks = tp.extractBLOCKS() 
    blocks.sort(key=lambda b: (round(b[1], 2), round(b[0], 2)))
    parts = [(b[4] or "").strip() for b in blocks if (b[4] or "").strip()]
    return "\n\n".join(parts)

def normalize(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[‘’]", "'", s)
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)     # de-hyphenate soft wraps
    s = re.sub(r"\n{3,}", "\n\n", s)           # collapse excess breaks
    return s

def get_manifest_info(text: str) -> Dict[str, Optional[str]]:

    try: 
        client = OpenAI(api_key=API_KEY)
        prompt = (
            f"""
            You are an expert in legal research and analysis, specializing in technology transactions. Examine the attached technology transactions document, and then identify respective values for each of the following keys: the role(s) of the party or parties (use only the roles, e.g., licensor, licensee; do not include the names of the parties), the governing law (use standard legal abbreviations, e.g., US-CA, US-FED, US-DE, etc.), the industry (use 1-3 words, e.g., biotechnology, software, SaaS), and the effective date of the agreement. Consider the substantive meaning of words (e.g., "Page 1 of 12" is not likely to be the governing law), placement in the document, surrounding text, applicable section, and any other factors that might inform your decision. Format the date in a string "YYYY-MM-DD". If you are unable to identify a value for the governing law, use the jurisdiction, if available.
            Return a response in following JSON format only: 
            {{
                "party_roles": party_roles, 
                "governing_law": governing_law, 
                "industry": industry, 
                "effective_date": effective_date
            }}. 
            Do not return anything else outside of the JSON object. If you cannot identify a value corresponding to one of the keys, respond with null for that key.
            --- PAGE TEXT START ---\n
            {text}\n
            --- PAGE TEXT END ---
            """
        )

        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt,
			text={"verbosity": "low"},
        )

        response = response.output_text if response else ""
        logger.info(f"AI manifest raw response: {response}")
        candidate = (response or [])
        logger.info(f"AI manifest response: {candidate}")
        expected_keys = [
            "party_roles",
            "governing_law",
            "industry",
            "effective_date",
        ]

        data = {}
        if isinstance(candidate, dict):
            data = candidate
        elif isinstance(candidate, str):
            try:
                data = json.loads(candidate)
            except Exception:
                logger.error("AI manifest response not valid JSON; returning defaults")
                data = {}
        else:
            logger.error("AI manifest response unexpected type %s", type(candidate))

        # Build normalized manifest dict
        manifest = {k: (data.get(k) if isinstance(data, dict) else None) for k in expected_keys}

        # Coerce obvious placeholder / unknown strings to None
        for k, v in manifest.items():
            if isinstance(v, str) and v.strip().lower() in {"unknown", "null", "n/a", "none", ""}:
                manifest[k] = None

        return manifest

    except Exception as e:
        logger.error(f"Error extracting title from text: {e}")
        return {}


def extract_pdf_text(file: FileStorage):
    file.stream.seek(0)
    text_parts = []
    logger.info("Extracting text from PDF")
    with fitz.open(stream=file.stream.read(), filetype="pdf") as doc:
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                text_parts.append(text)
            else:
                pix = page.get_pixmap()
                img = Image.frombytes(mode="RGB", size=(pix.width, pix.height), data=pix.samples)
                ocr_text = pytesseract.image_to_string(img)
                text_parts.append(ocr_text)
            text_parts.append("\n\n\f\n\n")  # Page break
    raw = "\n\n\f\n\n".join(text_parts).strip()
    return normalize(raw)

def extract_docx_text(file: FileStorage):
    file.stream.seek(0)
    logger.info("Extracting text from DOCX")
    doc = docx.Document(file.stream)
    text = [p.text for p in doc.paragraphs]
    raw = "\n".join(text)
    return normalize(raw)

def extract_text(file: FileStorage) -> str:
    filename = file.filename or ""
    file.stream.seek(0)
    _, ext = os.path.splitext(filename.lower())

    if ext == ".txt":
        return normalize(file.stream.read().decode('utf-8'))
    if ext == ".pdf":
        return extract_pdf_text(file)
    elif ext == ".docx":
        return extract_docx_text(file)
    else:
        raise ValueError("Unsupported file format")

def extract_title_type_jurisdiction(text: str) -> Dict[str, Optional[str]]:
    try:
        client = OpenAI(api_key=API_KEY)
        prompt = (
            f"""
            You are an expert in legal research and analysis, specializing in technology transactions. Examine the attached legal document, which is a technology transactions document, and then identify the title, type of technology transactions document (e.g., IP agreement, NDA, license, MSA, etc.), and jurisdiction. Avoid using proper names of the parties in the title. Try to use the type of legal document as the title -- e.g., IP Agreement, Master Services Agreement, Terms of Service, etc. Use an accepted legal abbreviation for the jurisdiction (e.g., US-CA, US-FED, FRG, etc.). Where appropriate, use an accepted legal abbreviation for the doc_type -- e.g., "MSA" for Master Services Agreement, "EULA" for End User License Agreement, etc. If you are unable to identify the jurisdiction, use the governing law, if available. Consider the substantive meaning of words (e.g., "Page 1 of 12" is not likely to be the title), placement in the document, case of the letters, length, punctuation, justification, and any other factors that might inform your decision. If you cannot confidently identify a corresponding value for a field, respond with null for that field. Return only a JSON object, with no other characters outside the JSON object. The JSON object should have the following keys: title, doc_type, jurisdiction. For example: 
            {{"title": title, "doc_type": doc_type, "jurisdiction": jurisdiction}}.
            --- PAGE TEXT START ---\n
            {text}\n
            --- PAGE TEXT END ---
            """
        )

        ai_model = "gpt-5-nano"

        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "Extract the document's title, doc_type, and jurisdiction based on the following text.\n"},
                {"role": "user", "content": prompt},
            ]
        )

        response_text = response.choices[0].message.content if response.choices else ""
        candidate = (response_text or "").strip()

        # Expected keys
        keys = ["title", "doc_type", "jurisdiction"]
        result: Dict[str, Optional[str]] = {k: None for k in keys}

        if not candidate:
            return result

        parsed: Dict[str, Optional[str]] = {}

        # Try direct JSON parse first
        try:
            parsed = json.loads(candidate)
        except Exception:
            # Attempt to salvage JSON substring if model added stray text
            try:
                import re as _re
                match = _re.search(r"\{.*\}", candidate, _re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                else:
                    logger.error("AI title/doc_type/jurisdiction response not valid JSON: %s", candidate[:200])
            except Exception:
                logger.error("Failed to parse AI JSON for title/doc_type/jurisdiction")

        if isinstance(parsed, dict):
            for k in keys:
                v = parsed.get(k)
                if isinstance(v, str):
                    vs = v.strip()
                    if vs.lower() in {"", "null", "none", "unknown", "n/a"}:
                        result[k] = None
                    else:
                        result[k] = vs
                elif v is None:
                    result[k] = None
                else:
                    # Unexpected type -> stringify conservatively
                    try:
                        result[k] = str(v)
                    except Exception:
                        result[k] = None

        return result
    except Exception as e:
        logger.error(f"Error extracting title from text: {e}")
        return {"title": None, "doc_type": None, "jurisdiction": None}

def sha256_file(file: FileStorage) -> str:
    file.stream.seek(0)
    return sha256_bytes(file.stream.read())

def sha256_bytes(bytes: bytes) -> str:
    return hashlib.sha256(bytes).hexdigest()

def sha256_text(text: str) -> str:
    """Compute the SHA-256 hash of a text string."""
    return sha256_bytes(text.encode('utf-8'))

def upsert_manifest_record(text: str, size: str, doc_id: int, sha256: str, source_path: str, clean_path: str, content_type: str | None, title: str | None, jurisdiction: str | None, doc_type: str | None, party_roles: str | None, governing_law: str | None, effective_date: str | None, industry: str | None) -> Dict[str, Optional[Any]]:
    """Upsert a manifest record in the database."""
    
    manifest_record = {
        "doc_id": str(doc_id),
        "title": title,
        "sha256": sha256,
        "source_path": source_path,
        "clean_path": clean_path,
        "bytes": size,
        "content_type": content_type,
        "doc_type": doc_type,
        "party_roles": party_roles,
        "jurisdiction": jurisdiction,
        "governing_law": governing_law,
        "industry": industry,
        "effective_date": effective_date,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    line = json.dumps(manifest_record) + "\n"
    manifest_path = MANIFEST_DIR + "/manifest.jsonl"
    
    try:
        
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(line)
        
        logger.info(f"Appended manifest record to {manifest_path}")
    except IOError as e:
        logger.error(f"Error writing manifest record to {manifest_path}: {e}")

    return manifest_record