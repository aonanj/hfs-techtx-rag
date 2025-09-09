# Tech Transfer RAG

A Flask-based Retrieval-Augmented Generation (RAG) application for document processing, embedding generation, and semantic search with AI-powered chat capabilities.

## 🌟 Features

- **Document Processing**: Upload and process PDF, DOCX, and text files
- **Smart Chunking**: Intelligent document segmentation for optimal retrieval
- **Vector Embeddings**: Generate embeddings using Hugging Face models
- **Semantic Search**: ChromaDB-powered vector similarity search
- **AI Chat Interface**: Interactive chat with document context using OpenAI or Claude
- **Web UI**: Clean, responsive interface for document management and chat
- **REST API**: Full API access for programmatic integration
- **Docker Support**: Containerized deployment with persistent data storage

## 🏗️ Architecture

```
├── app.py                 # Main Flask application
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── infrastructure/      # Core processing modules
│   ├── chunker.py       # Document chunking logic
│   ├── database.py      # ChromaDB interface
│   ├── document_processor.py  # File processing
│   ├── embeddings.py    # Embedding generation
│   ├── logger.py        # Logging configuration
│   └── vector_search.py # Search functionality
├── routes/              # Flask route handlers
│   ├── api.py          # REST API endpoints
│   └── web.py          # Web interface routes
└── static/             # Frontend HTML templates
    ├── index.html      # Main chat interface
    ├── upload.html     # Document upload page
    ├── chunks.html     # Document chunks viewer
    └── manifest.html   # Document manifest viewer
```

## 🚀 Quick Start

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
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   CHAT_MODEL=gpt-3.5-turbo
   OPENAI_BASE_URL=https://api.openai.com/v1
   MAX_CONTEXT_LENGTH=4000
   TOP_K_RESULTS=10
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
- `POST /api/upload` - Upload and process documents
- `GET /api/manifest` - List all processed documents
- `GET /api/chunks` - View document chunks
- `DELETE /api/reset` - Reset database (requires password)

### Search & Chat
- `POST /api/search` - Perform semantic search
- `POST /api/chat` - Chat with document context
- `GET /api/health` - Health check endpoint

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for chat models | Required |
| `CLAUDE_KEY` | Anthropic Claude API key | Optional |
| `EMBEDDING_MODEL` | Hugging Face embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHAT_MODEL` | OpenAI chat model | `gpt-3.5-turbo` |
| `MAX_CONTEXT_LENGTH` | Maximum context length for chat | `4000` |
| `TOP_K_RESULTS` | Number of search results | `10` |
| `RESET_PASSWORD` | Password for reset operations | None |

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

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions, please use the GitHub issue tracker.
