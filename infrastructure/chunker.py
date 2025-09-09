"""Utilities for segmenting plain-text documents into database chunks.

Currently provides a single public helper `chunk_doc` that:

1. Reads a cleaned `.txt` file (already extracted & normalized elsewhere).
2. Splits it into pages (form-feed ``\f`` boundaries) and paragraphs.
3. Aggregates paragraphs into size‑bounded chunks with optional character overlap.
4. Persists each chunk using `infrastructure.datab    path = os.path.join(CHUNKS_DIR, "chunks.jsonl")

    try:
        with open(path, "a", encoding="utf-8") as f:
            for record in chunk_metadata_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"Appended {len(chunk_metadata_records)} chunk records to existing chunks file {path}")
    except FileNotFoundError:
        with open(path, "w", encoding="utf-8") as f:
            for record in chunk_metadata_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"Wrote {len(chunk_metadata_records)} chunk records to new chunks file {path}")_chunk`.

Design notes:
* Chunk sizing is character based (not tokens) to avoid extra deps.
* Overlap helps preserve context across boundaries (default 150 chars).
* Each chunk records page_start / page_end (1-based indices of the spanned pages).
* A very small heuristic attempts to derive a section label from the first
  non-empty line that looks like a heading (ALL CAPS or Title Case and short).
* Idempotency: `persist_chunk` REPLACE logic + hash-derived chunk_id prevents
  uncontrolled duplication if the same text is reprocessed.

Future improvements (not required now):
* Token-based sizing using a tokenizer (e.g., tiktoken) when available.
* Heading detection via regex / numbering patterns (e.g., 1., 1.1, Section 2).
* Optional semantic boundary preservation (sentence splitter).
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any, Iterable
import os
import re
import json
from openai import OpenAI
from .logger import get_logger
from .embeddings import get_chroma_collection
from .database import add_chunk, get_max_chunk_id


try:  # Avoid hard import failure if heavy deps (openai, fitz) not installed during chunk-only operations
	from .document_processor import normalize  # type: ignore
except Exception:  # pragma: no cover - fallback path
	def normalize(s: str) -> str:  # minimal fallback
		s = s.replace("\r\n", "\n").replace("\r", "\n")
		s = re.sub(r"\n{3,}", "\n\n", s)
		return s

logger = get_logger()

API_KEY = os.getenv("OPENAI_API_KEY", "")
CHUNKS_DIR = os.getenv("CHUNKS_DIR", "/data/chunks")




def _split_pages(text: str) -> List[str]:
	"""Split text into pages on form-feed markers inserted during PDF extraction.

	If no form feed exists, the entire text is a single page.
	"""
	if "\f" not in text:
		return [text]
	# Allow light whitespace around form-feed separator pattern used in extraction
	parts = re.split(r"\n*\f\n*", text)
	return [p.strip() for p in parts if p.strip()]


def _paragraphs(page_text: str) -> List[str]:
	"""Return logical paragraphs (double-newline or blank-line separated)."""
	paras = re.split(r"\n{2,}", page_text)
	return [p.strip() for p in paras if p.strip()]


def _chunk_paragraphs(paragraphs: List[Tuple[int, str]], max_chars: int, overlap: int) -> List[Tuple[str, int, int]]:
	"""Aggregate (page_index, paragraph_text) into sized chunks.

	Returns list of (chunk_text, page_start, page_end).
	Overlap is applied on character basis between consecutive chunks.
	"""
	chunks: List[Tuple[str, int, int]] = []
	buf: List[str] = []
	buf_pages: List[int] = []
	current_len = 0
	last_tail = ""

	for page_idx, para in paragraphs:
		para_len = len(para)
		if current_len and current_len + para_len + 2 > max_chars:
			# Flush current buffer
			chunk_text = "\n\n".join(buf).strip()
			if chunk_text:
				chunks.append((chunk_text, min(buf_pages), max(buf_pages)))
				# Prepare overlap tail
				last_tail = chunk_text[-overlap:] if overlap > 0 else ""
			buf, buf_pages = [], []
			current_len = 0
		# Add overlap to new buffer if starting fresh and have tail
		if not buf and last_tail:
			buf.append(last_tail)
			buf_pages.append(page_idx)  # associate with current page for simplicity
			current_len = len(last_tail)
		buf.append(para)
		buf_pages.append(page_idx)
		current_len += para_len + 2  # account for join newlines

	# Final flush
	if buf:
		chunk_text = "\n\n".join(buf).strip()
		if chunk_text:
			chunks.append((chunk_text, min(buf_pages), max(buf_pages)))
	return chunks


try:  # Optional tokenizer (tiktoken) for accurate token counts
	import tiktoken  # type: ignore
	_TT_ENC = tiktoken.get_encoding("cl100k_base")
	def _encode_tokens(text: str) -> List[int]:
		return _TT_ENC.encode(text)  # type: ignore[union-attr]
	def _decode_tokens(tokens: List[int]) -> str:
		return _TT_ENC.decode(tokens)  # type: ignore[union-attr]
	TOKENIZER_NAME = "cl100k_base"
except Exception:  # pragma: no cover
	_TT_ENC = None  # type: ignore
	_FALLBACK_VOCAB: Dict[str, int] = {}
	_FALLBACK_ID2WORD: Dict[int, str] = {}
	def _encode_tokens(text: str) -> List[int]:  # fallback naive whitespace word -> int mapping
		tokens: List[int] = []
		for w in re.findall(r"\S+", text):
			if w not in _FALLBACK_VOCAB:
				idx = len(_FALLBACK_VOCAB) + 1
				_FALLBACK_VOCAB[w] = idx
				_FALLBACK_ID2WORD[idx] = w
			tokens.append(_FALLBACK_VOCAB[w])
		return tokens
	def _decode_tokens(tokens: List[int]) -> str:
		return " ".join(_FALLBACK_ID2WORD.get(t, "") for t in tokens).strip()
	TOKENIZER_NAME = "fallback_simple_vocab"


def _split_large_paragraph_tokens(tokens: List[int], max_tokens: int) -> Iterable[List[int]]:
	for i in range(0, len(tokens), max_tokens):
		yield tokens[i:i+max_tokens]


def _chunk_paragraphs_tokens(paragraphs: List[Tuple[int, str]], max_tokens: int, overlap_tokens: int) -> List[Tuple[str, int, int]]:
	"""Token-based chunk aggregation.

	paragraphs: list of (page_index, text)
	Returns list of (chunk_text, page_start, page_end)
	"""
	# Pre-tokenize paragraphs (cache by object id / value) to avoid repeats
	tokenized: List[Tuple[int, List[int], str]] = []  # (page_idx, token_ids, original_text)
	for page_idx, para in paragraphs:
		toks = _encode_tokens(para)
		# If single para exceeds max_tokens, split hard
		if len(toks) > max_tokens:
			for slice_tokens in _split_large_paragraph_tokens(toks, max_tokens):
				tokenized.append((page_idx, list(slice_tokens), _decode_tokens(list(slice_tokens))))
		else:
			tokenized.append((page_idx, toks, para))

	chunks: List[Tuple[str, int, int]] = []
	buf_tokens: List[int] = []
	buf_pages: List[int] = []
	tail_tokens: List[int] = []

	for page_idx, toks, original in tokenized:
		needed = len(toks)
		if buf_tokens and (len(buf_tokens) + needed) > max_tokens:
			# Flush current
			chunk_text = _decode_tokens(buf_tokens).strip()
			if chunk_text:
				chunks.append((chunk_text, min(buf_pages), max(buf_pages)))
				tail_tokens = buf_tokens[-overlap_tokens:] if overlap_tokens > 0 else []
			buf_tokens, buf_pages = [], []
		if not buf_tokens and tail_tokens:
			buf_tokens.extend(tail_tokens)
			buf_pages.append(page_idx)
		buf_tokens.extend(toks)
		buf_pages.append(page_idx)

	if buf_tokens:
		chunk_text = _decode_tokens(buf_tokens).strip()
		if chunk_text:
			chunks.append((chunk_text, min(buf_pages), max(buf_pages)))
	return chunks

def get_chunk_info(text: str, chunk_text: str) -> Dict[str, Optional[str]]:

    try: 
        client = OpenAI(api_key=API_KEY)
        prompt = (
            f"""
            You are an expert in legal research and analysis, specializing in technology transactions. Examine the attached technology transactions document and the chunk taken from the document then identify respective values for each of the following keys: section_number (full section, heading, subheading, subsection, paragraph, etc. number(s), letter(s),level(s), etc., applicable to the chunk -- e.g., 9.2(b); VII.A.i; etc.; note the section_number applicable to this chunk may not necessarily be included in this chunk), section_title (heading, section, subheading, subsection, etc. name(s), label(s), etc. applicable to this chunk), clause_type (the type(s) of clause(s) captured in this chunk), path (the heading, section, subheading, subsection, paragraph, etc. level(s)/name(s)/number(s) that lead to the value of section_number for this chunk -- e.g., if section_number is 9.2(b) for the chunk, the path would be 9 → 9.2 → 9.2(b)), numbers_present (if this chunk includes a section_number in the chunk_text), and definition_terms (a list of all defined terms in this chunk, exclude company and party names; include only legal terms). Do not include section_number in section_title, and do not include section_title in section_number. Consider the substantive meaning of words (e.g., "Page 1 of 12" is not likely to be the clause_type), placement in the document, surrounding text, applicable section, and any other factors that might inform your decision. Chunks may include multiple values for a given key (e.g., multiple section titles or clause types); in such cases, return all applicable values as a comma-separated list. If you cannot identify a value corresponding to one of the keys, respond with null for that key.
            Return a response in following JSON format only: 
            {{
                "section_number": section_number, 
                "section_title": section_title, 
                "clause_type": clause_type, 
                "path": path, 
                "numbers_present": numbers_present, 
                "definition_terms": [definition_terms]
            }}. 
            Do not return anything else outside of the JSON object. 
            --- DOCUMENT TEXT START ---\n
            {text}\n
            --- DOCUMENT TEXT END ---\n\n
			--- CHUNK TEXT START ---\n
			{chunk_text}\n
			--- CHUNK TEXT END ---
            """
        )

        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt,
			text={"verbosity": "low"},
        )

        response = response.output_text if response else ""
        logger.info(f"AI chunk raw response: {response}")
        candidate = (response or [])
        logger.info(f"AI chunk response: {candidate}")
        expected_keys = [
            "section_number",
            "section_title",
            "clause_type",
            "path",
            "numbers_present",
            "definition_terms",
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
        logger.error(f"Error extracting values from text: {e}")
        return {}

def chunk_doc(text: str, doc_id: int, max_chars: int = 1200, overlap: int = 150, *, max_tokens: Optional[int] = None, token_overlap: Optional[int] = None) -> List[int]:
	"""Chunk a cleaned text file, persist chunks, metadata, embeddings, and index.

	Returns list of chunk IDs.
	"""
	# Validate modes
	use_tokens = max_tokens is not None
	if use_tokens:
		if max_tokens is None or max_tokens <= 0:
			raise ValueError("max_tokens must be positive when provided")
		if token_overlap is None:
			# default 12% of max_tokens capped at 200
			token_overlap = min(max(0, int(max_tokens * 0.12)), 200)
		if token_overlap < 0:
			raise ValueError("token_overlap cannot be negative")
		if token_overlap >= max_tokens:
			logger.error("token_overlap (%d) >= max_tokens (%d); reducing", token_overlap, max_tokens)
			token_overlap = max(0, max_tokens // 5)
	else:
		if max_chars <= 0:
			raise ValueError("max_chars must be positive")
		if overlap < 0:
			raise ValueError("overlap cannot be negative")
		if overlap >= max_chars:
			logger.error("overlap (%d) >= max_chars (%d); reducing overlap", overlap, max_chars)
			overlap = max(0, max_chars // 4)  # soften


	pages = _split_pages(text)
	logger.error("Chunking doc_id=%s pages=%d (max_chars=%d overlap=%d)", doc_id, len(pages), max_chars, overlap)
	# Flatten paragraphs with page indices
	para_with_pages: List[Tuple[int, str]] = []
	for p_idx, page_text in enumerate(pages, start=1):
		for para in _paragraphs(page_text):
			para_with_pages.append((p_idx, para))

	if not para_with_pages:
		logger.error("No paragraphs detected; using raw text as single chunk for %s", doc_id)
		para_with_pages = [(1, text)]

	if use_tokens:
		chunk_specs = _chunk_paragraphs_tokens(para_with_pages, max_tokens=max_tokens, overlap_tokens=token_overlap or 0)  # type: ignore[arg-type]
		logger.info("Token chunking mode tokenizer=%s max_tokens=%s overlap_tokens=%s", TOKENIZER_NAME, max_tokens, token_overlap)
	else:
		chunk_specs = _chunk_paragraphs(para_with_pages, max_chars=max_chars, overlap=overlap)

	chunk_ids: List[int] = []

	chunk_metadata_records: List[Dict[str, Any]] = []

	max_id = get_max_chunk_id()
	start_chunk_id = max_id + 1

	for idx, (chunk_text, page_s, page_e) in enumerate(chunk_specs):
		
		chunk_data = get_chunk_info(text, chunk_text)
		sec_number_raw = chunk_data.get("section_number")
		section_title_raw = chunk_data.get("section_title")
		clause_type_raw = chunk_data.get("clause_type")
		path_raw = chunk_data.get("path")
		numbers_present_raw = chunk_data.get("numbers_present")
		definition_terms_raw = chunk_data.get("definition_terms")

		# Convert numbers_present to boolean
		numbers_present = None
		if numbers_present_raw is not None:
			if isinstance(numbers_present_raw, bool):
				numbers_present = numbers_present_raw
			elif isinstance(numbers_present_raw, str):
				numbers_present = numbers_present_raw.strip().lower() in {"true", "yes", "1"}

		# Convert definition_terms to list if it's a string or ensure it's a list
		definition_terms = None
		if definition_terms_raw is not None:
			if isinstance(definition_terms_raw, list):
				definition_terms = ", ".join(str(term) for term in definition_terms_raw)
			else:
				definition_terms = str(definition_terms_raw)
		
		sec_number = None
		if sec_number_raw is not None:
			if isinstance(sec_number_raw, list):
				sec_number = ", ".join(str(num) for num in sec_number_raw)
			else:
				sec_number = str(sec_number_raw)
	
		section_title = None
		if section_title_raw is not None:
			if isinstance(section_title_raw, list):
				section_title = ", ".join(str(title) for title in section_title_raw)
			else:
				section_title = str(section_title_raw)
		
		clause_type = None
		if clause_type_raw is not None:
			if isinstance(clause_type_raw, list):
				clause_type = ", ".join(str(ctype) for ctype in clause_type_raw)
			else:
				clause_type = str(clause_type_raw)
		
		path = None
		if path_raw is not None:
			if isinstance(path_raw, list):
				path = ", ".join(str(p) for p in path_raw)
			else:
				path = str(path_raw)

		chunk_id = start_chunk_id + idx

		prev_idx = None
		next_idx = None
		if idx > 0: 
			prev_idx = idx - 1
		if idx < len(chunk_specs) - 1:
			next_idx = idx + 1
		new_chunk = add_chunk(
			doc_id=doc_id,
			chunk_id=chunk_id,
			chunk_index=idx,
			text=chunk_text,
			page_start=page_s,
			page_end=page_e,
			section_number=sec_number,
			section_title=section_title,
			clause_type=clause_type,
			path=path,
			numbers_present=numbers_present,
			definition_terms=definition_terms,
		)
		logger.error("Persisted chunk %s (pages %d-%d) for doc_id=%s", chunk_id, page_s, page_e, doc_id)

		chunk_ids.append(chunk_id)
		logger.error(f"Chunk {chunk_id} info: {new_chunk}")

		chunk_metadata_records.append({
			'id': chunk_id,
			'index': idx,
			'doc_id': doc_id,
			'section_number': sec_number,
			'section_title': section_title,
			'clause_type': clause_type,
			'path': path,
			'numbers_present': numbers_present,
			'definition_terms': definition_terms,
			'text': chunk_text,
			'metadata': None,
			'prev_id': prev_idx,  # fill later
			'next_id': next_idx,  # fill later
		})
		logger.info("Chunk metadata record: %s", chunk_metadata_records[-1])


	records = json.dumps(chunk_metadata_records, ensure_ascii=False, indent=0)
	path = os.path.join(CHUNKS_DIR, "chunks.jsonl")

	try:
		with open(path, "a", encoding="utf-8") as f:
			f.write(records)
		logger.info(f"Appended {len(chunk_metadata_records)} chunk records to existing chunks file {path}")
	except FileNotFoundError:
		with open(path, "w", encoding="utf-8") as f:
			f.write(records)
		logger.info(f"Created new chunks file with {len(chunk_metadata_records)} records as it did not exist")
	except Exception as e:
		logger.error(f"Failed to write chunk records to {path}: {e}")

	chunk_texts = [spec[0] for spec in chunk_specs]
	# Generate embeddings and upsert to ChromaDB
	try:
		collection = get_chroma_collection()
		# ChromaDB metadatas must be Dict[str, Bool|Int|Float|Str] with no None values.
		# Also drop internal keys and any unsupported types (coerce to str as last resort).
		def _chroma_safe_metadata(rec: dict) -> dict:
			out: dict = {}
			for k, v in rec.items():
				if k in {"id", "text", "prev_id", "next_id", "metadata"}:
					continue
				if v is None:
					continue
				if isinstance(v, (bool, int, float, str)):
					out[k] = v
				else:
					# Fallback: stringify lists/dicts or other types
					out[k] = str(v)
			return out

		collection.add(
			ids=[str(cid) for cid in chunk_ids],
			documents=chunk_texts,
			metadatas=[_chroma_safe_metadata(rec) for rec in chunk_metadata_records]
		)
		logger.info('Upserted %d embeddings to ChromaDB for doc_id=%s', len(chunk_ids), doc_id)

	except Exception as e:  # pragma: no cover
		logger.error('Embedding/indexing step failed for %s: %s', doc_id, e)


	logger.info("Persisted %d chunks for doc_id=%s", len(chunk_ids), doc_id)
	return chunk_ids


__all__ = ["chunk_doc"]

