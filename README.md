# RAG Search & Retrieval System

A comprehensive **Retrieval-Augmented Generation (RAG)** system with **advanced document processing**, **multi-agent orchestration**, and **observability** through Phoenix AI. Built with CrewAI, LlamaIndex, FastAPI, and PostgreSQL.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (for local development)
- Git

### 1. Clone & Setup
```bash
git clone <repository-url>
cd rag_search
```

### 2. Start All Services
```bash
# Start the complete stack
docker-compose up -d

# Or start services individually:
docker-compose up -d phoenix     # Observability dashboard
docker-compose up -d rag-api     # RAG API server
docker-compose up -d open-webui  # Chat interface
```

### 3. Access the Applications

| Service | URL | Description |
|---------|-----|-------------|
| **OpenWebUI** | http://localhost:3000 | Main chat interface |
| **Phoenix UI** | http://localhost:6006 | Observability & traces |
| **RAG API** | http://localhost:8000 | Direct API access |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |

---

## 📁 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenWebUI     │───▶│    RAG API      │───▶│   PostgreSQL    │
│ (Chat Interface)│    │  (FastAPI)      │    │  + PGVector     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         
                              ▼                         
                    ┌─────────────────┐                 
                    │   CrewAI Agents │                 
                    │ Document Retriev│                 
                    │ Insight Synthesi│                 
                    └─────────────────┘                 
                              │                         
                              ▼                         
                    ┌─────────────────┐                 
                    │   Phoenix AI    │                 
                    │  (Observability)│                 
                    └─────────────────┘                 
```

### Core Components

- **CrewAI Multi-Agent System**: Document researcher + insight synthesizer
- **LlamaIndex**: Vector storage and retrieval with PostgreSQL + PGVector
- **FastAPI**: OpenAI-compatible API server
- **Phoenix AI**: Complete observability and tracing
- **OpenWebUI**: User-friendly chat interface

---

## 🛠️ Development Workflow

### Local Development (Non-Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Test FastAPI locally
python rag_api_server.py
# Access: http://localhost:8000
```

### Docker Development

```bash
# 1. Make changes to rag_api_server.py

# 2. Rebuild (fast):
docker-compose build rag-api  # ~5-10 seconds

# 3. Restart service:
docker-compose up -d rag-api  # ~2-3 seconds

# 4. Check logs:
docker logs -f --tail 10 rag-api
```

### Stop & Clean Up

```bash
# Stop individual services
docker-compose down phoenix
docker-compose down rag-api
docker-compose down open-webui

# Or stop everything
docker-compose down

# Remove containers completely
docker rm -f phoenix rag-api open-webui
```

---

## 🧪 Testing & Evaluation

### A) RAG Functionality Testing

```bash
# Test semantic retrieval strategy
pipenv run python tests/test_enhanced_rag_query.py --strategy semantic

# Test hierarchical strategy  
pipenv run python tests/test_enhanced_rag_query.py --strategy hierarchical

```

### B) Integration Testing

```bash
# Full RAG integration test
python tests/test_rag.py

# API endpoint testing
pipenv run python tests/test_rag_api.py

# Document ingestion testing
python tests/test_enhanced_ingestion.py
```

#### Chunking Strategy Tests
The system supports multiple chunking strategies:
- **Basic chunking**: Simple sentence-based splitting
- **Structure-aware chunking**: Respects document hierarchy
- **Semantic chunking**: Groups by semantic similarity  
- **Hierarchical chunking**: Parent-child relationships
- **Contextual RAG chunking**: Advanced Q&A generation

### C) RAGAs Evaluation

```bash
# RAGAs evaluation with Phoenix logging
pipenv run python tests/test_runner.py --test-type evaluation

# Run hierarchical strategy evaluation
pipenv run python tests/test_runner.py --strategy hierarchical

# Run both strategies comparison
pipenv run python tests/test_runner.py --strategy both
```

---

## 📊 Observability & Monitoring

### Phoenix AI Integration

Phoenix provides complete observability for the RAG system:

- **Trace Monitoring**: Track all LLM calls and agent interactions
- **Performance Metrics**: Response times, token usage, retrieval scores
- **Project Isolation**: Traces sent to dedicated `rag_system` project
- **Real-time Debugging**: Live trace viewing and analysis

```bash
# Access Phoenix dashboard
http://localhost:6006

# Phoenix traces are automatically sent to project: "rag_system"
```

### Key Metrics Tracked

- Document retrieval performance
- Agent execution traces  
- LLM response quality
- Source citation accuracy
- End-to-end request latency

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```bash
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/rag_db

# Ollama Integration  
OLLAMA_BASE_URL=http://ollama:11434
EMBEDDING_MODEL=nomic-embed-text:latest
LLM_MODEL=gemma2:9b

# API Configuration
RAG_API_PORT=8000
RAG_API_HOST=0.0.0.0
RAG_MODEL_NAME=rag-search

# Phoenix Observability
PHOENIX_HOST=phoenix
PHOENIX_PORT=6006
```

### Service Ports

| Service | Internal Port | External Port | 
|---------|---------------|---------------|
| RAG API | 8000 | 8000 |
| Phoenix | 6006 | 6006 |
| OpenWebUI | 8080 | 3000 |
| PostgreSQL | 5432 | - |
| Ollama | 11434 | - |

---

## 📖 API Usage

### OpenAI-Compatible Endpoints

```bash
# List available models
curl http://localhost:8000/v1/models

# Chat completion (OpenWebUI compatible)
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-search",
    "messages": [
      {"role": "user", "content": "What are the procurement principles in Abu Dhabi?"}
    ]
  }'

# Health check
curl http://localhost:8000/health
```

### Direct Search API

```bash
# Direct document search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "government procurement guidelines",
    "max_results": 5
  }'
```

### Response Format

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion", 
  "model": "rag-search",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Based on the Abu Dhabi Procurement Standards document...\n\nSources\n\nAbu Dhabi Procurement Standards.PDF (score: 0.8234)\nHR Bylaws.pdf (score: 0.7156)\n\nRetrieval Strategy\n\nMethod: semantic\nChunks Used: 5"
    },
    "finish_reason": "stop"
  }]
}
```

---

## 🗂️ Document Management

### Supported Formats
- PDF documents
- Word documents (.docx)
- Plain text files
- Markdown files

### Document Processing Features
- **Structure-aware chunking** with context preservation
- **Table and image extraction** from PDFs
- **Metadata extraction** for citations
- **Multiple processing strategies** for different use cases
- **Automatic source tracking** and citation generation

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check logs
docker logs rag-api
docker logs phoenix
docker logs open-webui

# Rebuild if needed
docker-compose build rag-api --no-cache
```

#### 2. Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose ps

# Verify database connectivity
docker exec -it rag-api bash
# Inside container: python -c "import psycopg2; print('DB OK')"
```

#### 3. Ollama Model Issues
```bash
# Check Ollama status
docker logs ollama

# Verify models are available
curl http://localhost:11434/api/tags
```

#### 4. No Document Retrieval
```bash
# Check if documents are indexed
docker exec -it postgres psql -U postgres -d rag_db
# SQL: SELECT count(*) FROM data_llamaindex_enhanced_semantic;
```

### Debug Mode

```bash
# Enable detailed logging
docker-compose up rag-api
# Watch logs in real-time
```

---

## 📚 Additional Documentation

| Document | Description |
|----------|-------------|
| [Phoenix Integration](docs/PHOENIX_INTEGRATION_COMPLETE.md) | Phoenix AI setup and configuration |
| [Phoenix Prompts](docs/PHOENIX_PROMPTS_README.md) | Prompt management system |
| [FastAPI Details](docs/RAG_FASTAPI_README.md) | API server architecture |
| [URL References](docs/URL_REFERENCE.md) | Service endpoints and URLs |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality  
5. Run the test suite
6. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🆘 Support

For issues and questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review existing [GitHub Issues](issues)
3. Create a new issue with detailed reproduction steps
4. Include relevant logs and configuration details

---

**Built using CrewAI, LlamaIndex, FastAPI, and Phoenix AI**