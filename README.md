---
title: Technology Transactions RAG
emoji: 📑
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "0.0.1"
app_file: app.py
pinned: false
---

# Technology Transactions RAG

A Flask-based Retrieval-Augmented Generation (RAG) application for document processing, embedding generation, and semantic search specific to technology transactions use cases in the legal domain.

Available at: [tech-trans-rag](https://huggingface.co/spaces/phaethon-order/tech-trans-rag "Hugging Face Spaces: phaethon-order/tech-trans-rag")

## 💡 Features

- **Document Processing**: Upload and process PDF, DOCX, and text files
- **Smart Chunking**: Intelligent document segmentation for optimal retrieval
- **Vector Embeddings**: Generate embeddings using Hugging Face models
- **Semantic Search**: ChromaDB-powered vector similarity search
- **Dynamic Corpus**: Corpus can be updated to increase knowledge base and customize responses to user preferences, including best practices, jurisdictions, and governing laws. 
- **Web UI**: Clean, responsive interface for document management and search
- **REST API**: Full API access for programmatic integration
- **Docker Support**: Containerized deployment with persistent data storage

## 🏗️ Architecture

```
├── app.py                     # Main Flask application
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── infrastructure/            # Core processing modules
│   ├── chunker.py             # Document chunking logic
│   ├── database.py            # ChromaDB interface
│   ├── document_processor.py  # File processing
│   ├── embeddings.py          # Embedding generation
│   ├── logger.py              # Logging configuration
│   └── vector_search.py       # Search functionality
├── routes/                    # Flask route handlers
│   ├── api.py                 # REST API endpoints
│   └── web.py                 # Web interface routes
├── static/                    # Frontend HTML templates
│   ├── index.html             # Main search interface
│   ├── upload.html            # Document upload page
│   ├── chunks.html            # Document chunks viewer
│   ├── manifest.html          # Document manifest viewer
│   └── dbviewer.html          # Raw database viewer
└── services/                  # GPT services
    └── gpt_service.py         # Query response enrichment
```

## 🛠️ Quick Start

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hfs-techtx-rag
   ```

2. **Set up environment variables** (create `.env` file)
   ```bash
   # Required API Keys
   OPENAI_API_KEY=your_openai_api_key_here
   CLAUDE_KEY=your_claude_api_key_here

   # Optional Configuration
   EMBEDDING_MODEL=text-embedding-3-small
   CHAT_MODEL=gpt-5
   OPENAI_BASE_URL=https://api.openai.com/v1
   MAX_CONTEXT_LENGTH=4000
   TOP_K_RESULTS=5
   MAX_QUERY_LENGTH=1000
   REQUEST_TIMEOUT=30
   SECRET_KEY=your_secret_key_here

   # Security (optional)
   RESET_PASSWORD=change-me
   ```

### Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   python app.py
   ```

3. **Access the application**
   - Web Interface: http://localhost:5000
   - Upload Documents: http://localhost:5000/upload
   - API Documentation: http://localhost:5000/api

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t tech-trans-rag .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 \
     -e OPENAI_API_KEY=your_key_here \
     -v $(pwd)/data:/data \
     tech-trans-rag
   ```

## 📚 API Endpoints

### Document Management
- `POST /api/upload` — Upload and process documents
- `GET /api/manifest` — List processed documents (document manifest)
- `PATCH /api/manifest` — Update manifest fields
- `DELETE /api/manifest` — Delete from manifest
- `GET /api/documents` — List documents (DB viewer optimized)
- `GET /api/chunks?doc_id=<id>` — List chunks for a given document ID
- `GET /api/chunks/all` — List chunks across all documents (paginated in UI)
- `GET /api/chunks/jsonl` — Download chunks as JSONL
- `PATCH /api/chunks` — Update chunk metadata (e.g., flags or annotations)

### Search & Health
- `POST /api/query` — Perform semantic search and response generation
- `GET /api/healthz` — Health check endpoint

### Admin
- `POST /api/reset` — Reset ChromaDB and reinitialize collections
   - If `RESET_PASSWORD` is set, provide `{"password": "<value>"}` in the JSON body or pass header `X-Reset-Password: <value>`

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for chat models | Required |
| `CLAUDE_KEY` | Anthropic Claude API key | Optional |
| `EMBEDDING_MODEL` | Hugging Face embedding model | `text-embedding-3-small` |
| `CHAT_MODEL` | OpenAI chat model | `gpt-5` |
| `MAX_CONTEXT_LENGTH` | Maximum context length for chat | `4000` |
| `TOP_K_RESULTS` | Number of search results | `5` |
| `RESET_PASSWORD` | Password for reset operations | None |

### API Field Notes

- Chunk responses include rich metadata when available: `section_number`, `section_title`, `clause_type`, `page_start`, `page_end`, `path`, `numbers_present`, `definition_terms`, along with operational fields like `chunk_index`, `token_count`, `tok_ver`, `seg_ver`, and `embedding_count`.
- When querying by `doc_id` (`GET /api/chunks?doc_id=<id>`), the response shape matches the global chunks listing so UIs can render consistently.

### Data Persistence

The application uses the following directories for data storage:

- `/data/chroma_db` - ChromaDB vector database
- `/data/corpus_raw` - Original uploaded files
- `/data/corpus_clean` - Processed documents
- `/data/manifest` - Document metadata
- `/data/chunks` - Document chunks
- `/data/cache` - Model cache
- `/data/.huggingface` - Hugging Face model cache

## 🔒 Security Features

### Reset Protection
You can protect the dangerous Reset operation by setting the `RESET_PASSWORD` environment variable:

```bash
export RESET_PASSWORD="change-me"
python app.py
```

The Reset button in the web interface will prompt for this password before allowing database wipes.

### Try the API quickly

Optional cURL snippets you can run locally:

```bash
# Health
curl -s http://localhost:5000/api/healthz | jq

# Manifest
curl -s http://localhost:5000/api/manifest | jq

# Chunks for a document (replace 1 with a real ID from manifest)
curl -s "http://localhost:5000/api/chunks?doc_id=1" | jq '.chunks[0]'

# Query with semantic search
curl -s -X POST http://localhost:5000/api/query \
   -H 'Content-Type: application/json' \
   -d '{"query": "What does the indemnity clause cover?", "top_k": 5}' | jq

# Admin reset (if password required)

# Update a chunk's metadata (example: set clause_type)
curl -s -X PATCH http://localhost:5000/api/chunks \
   -H 'Content-Type: application/json' \
   -d '{"chunk_id": 123, "updates": {"clause_type": "Indemnification"}}' | jq

# Get all chunks (paginated)
curl -s "http://localhost:5000/api/chunks/all?limit=200&offset=0" | jq '.chunks | length'
curl -s -X POST http://localhost:5000/api/reset \
   -H 'Content-Type: application/json' \
   -d '{"password": "change-me"}' | jq
```

## 🛠️ Development

### Project Structure
- **Infrastructure Layer**: Core processing modules for documents, embeddings, and search
- **Route Layer**: Flask blueprints for web and API endpoints
- **Static Layer**: Frontend templates and assets
- **Configuration**: Environment-based configuration management

### Key Components
- **Document Processor**: Handles PDF, DOCX, and text file processing with OCR support
- **Chunker**: Intelligent text segmentation for optimal retrieval
- **Embeddings**: Hugging Face model integration for vector generation
- **Vector Search**: ChromaDB-based semantic search capabilities
- **Database**: Persistent storage with automatic initialization and reset capabilities

### Troubleshooting

- Health endpoint returns 404: Ensure you are calling `/api/healthz` (not `/api/health`).
- No chunks returned for `doc_id`: Verify the document ID from `/api/manifest`. Chunks are only created after successful processing.
- Reset not permitted: Provide the correct password in JSON body (`{"password": "..."}`) or `X-Reset-Password` header and ensure `RESET_PASSWORD` is set in the server environment.
- ChromaDB file errors on macOS/Linux: Confirm your Docker volume mount maps to a writable host path: `-v $(pwd)/data:/data`.
- Missing API keys: Set `OPENAI_API_KEY` (and optional `CLAUDE_KEY`) before starting the app.
- JSONL not found: `/api/chunks/jsonl` returns empty if `/data/chunks/chunks.jsonl` has not been generated yet by the chunker; use `/api/chunks/all` as a fallback.
- Large uploads: For PDFs with many pages, processing can take time; watch the server logs for progress and ensure model caches under `/data/cache` are writable.

## 🏴‍☠️ License

AGPLv3 License - see the [LICENSE](LICENSE) file for details.

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Contact

For issues and questions, please contact [phaethon@phaethon.llc](mailto:phaethon@phaethon.llc). 
