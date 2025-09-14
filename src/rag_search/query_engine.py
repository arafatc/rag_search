"""
RAG Query Engine Module
Shared query engine setup for both tests and API usage
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:
    try:
        from llama_index.embeddings import OllamaEmbedding
    except ImportError:
        print("Warning: OllamaEmbedding not available. Install llama-index-embeddings-ollama")
        OllamaEmbedding = None

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:
    try:
        from llama_index.embeddings import OllamaEmbedding
    except ImportError:
        print("Warning: OllamaEmbedding not available. Install llama-index-embeddings-ollama")
        OllamaEmbedding = None

try:
    from llama_index.llms.ollama import Ollama
except ImportError:
    try:
        from llama_index.llms import Ollama
    except ImportError:
        print("Warning: Ollama not available. Install llama-index-llms-ollama")
        Ollama = None

from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore

# Configuration
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "rag_db")

# Provider configuration - using Ollama only
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")

# Ollama configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

# Embedding dimensions mapping
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "nomic-embed-text:latest": 768,
    "mxbai-embed-large:latest": 1024,
    "all-minilm:latest": 384
}

# Add externalized EMBEDDING_DIM
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

def get_embedding_dim():
    """Get embedding dimension based on provider and model"""
    # Check for explicit override first
    if os.getenv("EMBEDDING_DIM"):
        return EMBEDDING_DIM
    
    # Use Ollama model dimensions
    return EMBEDDING_DIMENSIONS.get(EMBEDDING_MODEL, 768)

def setup_enhanced_rag_query_engine(strategy="semantic"):
    """Setup the enhanced RAG query engine for a specific strategy"""
    
    # Initialize embedding model - using Ollama only
    if OllamaEmbedding is None:
        raise Exception("OllamaEmbedding not available")
    embedding = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    
    # Initialize LLM - using Ollama only
    if Ollama is None:
        raise Exception("Ollama LLM not available")
    llm = Ollama(
        model=LLM_MODEL,  # Uses Gemma 3:1b model
        base_url=OLLAMA_BASE_URL,
        request_timeout=180.0,  # Match agents.py timeout
        temperature=0.3,  # Match agents.py temperature
        num_ctx=4096,     # Match agents.py context window
    )
    
    # Strategy-specific table names (match tools.py naming convention)
    vector_table = f"data_llamaindex_enhanced_{strategy.lower()}"
    docstore_table = f"data_llamaindex_enhanced_docstore_{strategy.lower()}"
    
    # Get the correct embedding dimension
    embed_dim = get_embedding_dim()
    
    # Connect to enhanced vector store
    vector_store = PGVectorStore.from_params(
        database=PG_DATABASE,
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        table_name=vector_table,
        embed_dim=embed_dim,
        perform_setup=False
    )
    
    # Connect to enhanced document store
    doc_store = PostgresDocumentStore.from_params(
        database=PG_DATABASE,
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        table_name=docstore_table,
        perform_setup=False
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=doc_store
    )
    
    # Create index from existing storage
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embedding
    )
    
    # Create query engine with settings matching agents.py configuration
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,  # Match tools.py default
        response_mode="compact",
        streaming=False,  # Disable streaming for better response control
        temperature=0.3   # Match agents.py temperature
    )
    
    return query_engine, index
