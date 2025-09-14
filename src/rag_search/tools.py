import os
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse
import psycopg2
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from crewai.tools import tool
from typing import Dict, Union, Any, List
 # Load environment variables
load_dotenv()

# Tool call counter to prevent repeated usage
_tool_call_counter = 0

def reset_tool_call_counter():
    """Reset the tool call counter for a new session"""
    global _tool_call_counter
    _tool_call_counter = 0

# Configuration for retrieval strategy priority
# Use only semantic since hierarchical table doesn't exist
PRIMARY_STRATEGY = "semantic"
FALLBACK_STRATEGY = "custom"  # Use CustomPGVectorRetriever as fallback

def warm_up_ollama(base_url: str, model_name: str) -> bool:
    """Pre-warm Ollama model to avoid cold start delays"""
    try:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model_name, "prompt": "test"},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Warning: Could not warm up Ollama model: {e}")
        return False

class LlamaIndexRetriever:
    """LlamaIndex-based retriever for properly ingested documents"""
    
    def __init__(self, database_url: str, table_name: str, embed_model):
        self.database_url = database_url
        self.table_name = table_name
        self.embed_model = embed_model
        self._setup_storage()
        print(f"DEBUG: LlamaIndexRetriever initialized with table_name: '{table_name}'")
    
    def _setup_storage(self):
        """Setup LlamaIndex storage context"""
        # Parse database URL
        db_url_parts = urlparse(self.database_url)
        
        # Vector store
        self.vector_store = PGVectorStore.from_params(
            database=db_url_parts.path.lstrip('/'),
            host=db_url_parts.hostname,
            port=db_url_parts.port,
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name=self.table_name,
            embed_dim=int(os.getenv("EMBEDDING_DIM", "768")),  # Use externalized embedding dimension
        )
        
        # Document store
        doc_table_name = self.table_name.replace('_documents', '_docstore')
        self.doc_store = PostgresDocumentStore.from_params(
            database=db_url_parts.path.lstrip('/'),
            host=db_url_parts.hostname,
            port=db_url_parts.port,
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name=doc_table_name
        )
        
        # Storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.doc_store
        )
    
    def retrieve(self, query: str, similarity_top_k: int = 5) -> List[NodeWithScore]:
        """Retrieve documents using LlamaIndex"""
        try:
            # Create index from storage
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )
            
            # Create retriever
            retriever = index.as_retriever(similarity_top_k=similarity_top_k)
            
            # Retrieve
            nodes = retriever.retrieve(query)
            return nodes
            
        except Exception as e:
            print(f"LlamaIndex retrieval failed: {e}")
            return []

class CustomPGVectorRetriever:
    """Custom PostgreSQL vector retriever that works reliably"""
    
    def __init__(self, database_url: str, table_name: str, embed_model):
        self.database_url = database_url
        self.table_name = table_name
        self.embed_model = embed_model
        print(f"DEBUG: CustomPGVectorRetriever initialized with table_name: '{table_name}'")
    
    def retrieve(self, query: str, similarity_top_k: int = 5) -> List[NodeWithScore]:
        """Retrieve documents using custom SQL query"""
        try:
            # Generate query embedding
            query_embedding = self.embed_model.get_query_embedding(query)
            query_vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Connect to database
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Perform vector similarity search
            print(f"DEBUG: Executing query with table_name: '{self.table_name}'")
            query_sql = """
                SELECT id, text, metadata_, node_id, embedding <-> %s::vector as distance
                FROM {}
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
            """.format(self.table_name)
            
            print(f"DEBUG: Generated SQL: {query_sql[:100]}...")
            cursor.execute(query_sql, (query_vector_str, query_vector_str, similarity_top_k))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            print(f"DEBUG: CustomPGVectorRetriever found {len(results)} results")
            
            # Convert to NodeWithScore objects
            nodes = []
            for row in results:
                id_, text, metadata_, node_id, distance = row
                
                # Create TextNode
                node = TextNode(
                    text=text,
                    id_=node_id or str(id_),
                    metadata=metadata_ or {}
                )
                
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = 1.0 / (1.0 + distance)
                
                node_with_score = NodeWithScore(
                    node=node,
                    score=similarity_score
                )
                nodes.append(node_with_score)
            
            return nodes
            
        except Exception as e:
            print(f"ERROR: CustomPGVectorRetriever failed: {e}")
            return []

# Tool definition
import json

# ... existing code ...

@tool("Document Retrieval Tool")
def document_retrieval_tool(query: Union[str, Dict[str, Any]]) -> str:
    """
    Retrieves relevant context from a collection of policy and standards documents
    using LlamaIndex with fallback. This tool is designed to handle complex,
    nested, or stringified JSON inputs from agents.
    """
    global _tool_call_counter
    _tool_call_counter += 1
    
    # Prevent repeated calls in the same session
    if _tool_call_counter > 1:
        return (
            "ERROR: Tool already used once in this session. "
            "Each agent can only call document_retrieval_tool once per query. "
            "Please use the results from your previous tool call."
        )
    
    search_query = None
    
    def find_query_in_data(data):
        """Recursively search for a query string."""
        if isinstance(data, str):
            try:
                # If the string is JSON, parse it and search inside
                data = json.loads(data)
            except json.JSONDecodeError:
                # It's just a string, so it could be our query
                return data

        if isinstance(data, dict):
            # Prioritize common keys where the query might be found
            for key in ['query', 'description', 'q', 'search_term', 'input']:
                if key in data:
                    # Recurse on the value associated with the key
                    result = find_query_in_data(data[key])
                    if result and isinstance(result, str):
                        return result
            
            # If not in priority keys, search all values
            for value in data.values():
                result = find_query_in_data(value)
                if result and isinstance(result, str):
                    return result

        return None

    search_query = find_query_in_data(query)

    if not search_query or not isinstance(search_query, str):
        return f"Error: Could not extract a valid search query from the provided input: {query}"

    print(f"INFO: Executing document retrieval for extracted query: '{search_query}'")
    
    primary_retriever = get_retriever(PRIMARY_STRATEGY)
    fallback_retriever = get_retriever(FALLBACK_STRATEGY)
    
    retrieved_nodes = primary_retriever.retrieve(search_query)
    strategy_used = PRIMARY_STRATEGY
    
    if not retrieved_nodes:
        print(f"WARN: Primary strategy '{PRIMARY_STRATEGY}' failed. Using fallback '{FALLBACK_STRATEGY}'.")
        retrieved_nodes = fallback_retriever.retrieve(search_query)
        strategy_used = FALLBACK_STRATEGY
        
    if not retrieved_nodes:
        print("ERROR: Both primary and fallback strategies failed.")
        # Check if database connection works at all
        try:
            import psycopg2
            conn = psycopg2.connect(os.getenv("DATABASE_URL"))
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM data_llamaindex_enhanced_semantic;")
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            print(f"DEBUG: Database has {count} documents in semantic table")
            return f"Error: Document retrieval failed with all strategies. Database accessible with {count} documents."
        except Exception as e:
            print(f"ERROR: Database connection failed: {e}")
            return f"Error: Document retrieval failed and database connection error: {e}"
        
    return format_retrieval_output(retrieved_nodes, strategy_used)


def get_retriever(strategy: str) -> Union[LlamaIndexRetriever, CustomPGVectorRetriever]:
    """Get the appropriate retriever instance based on the strategy"""
    DATABASE_URL = os.getenv("DATABASE_URL")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    
    # Initialize embedding model
    warm_up_ollama(ollama_base_url, embedding_model)
    
    embed_model = OllamaEmbedding(
        model_name=embedding_model,
        base_url=ollama_base_url,
        request_timeout=60.0  # Reduced timeout from 120s to 60s
    )

    if strategy == "semantic":
        return LlamaIndexRetriever(
            database_url=DATABASE_URL,
            table_name=f"data_llamaindex_enhanced_{strategy}",
            embed_model=embed_model
        )
    elif strategy == "custom":
        # Use CustomPGVectorRetriever directly against semantic table
        return CustomPGVectorRetriever(
            database_url=DATABASE_URL,
            table_name="data_llamaindex_enhanced_semantic",
            embed_model=embed_model
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def format_retrieval_output(retrieved_nodes: List[NodeWithScore], strategy_used: str) -> str:
    """Format the retrieved nodes into a human-readable string"""
    formatted_chunks = []
    source_info = {"sources": [], "strategy": strategy_used, "chunks_used": len(retrieved_nodes)}
    total_chars = 0
    max_context_chars = 2000  # Limit context size like ultra-simple version
    
    for i, node_with_score in enumerate(retrieved_nodes, 1):
        node = node_with_score.node
        score = node_with_score.score
        
        # Extract source from metadata - try multiple possible keys
        source = node.metadata.get('file_name') or node.metadata.get('source') or 'Unknown source'
        source_info["sources"].append({"document": source, "score": round(float(score), 4)})
        
        chunk_text = f"""--- Chunk {i} (Score: {score:.4f}) ---
Source: {source}
Content: {node.text}
"""
        
        # Stop if we exceed context limit
        if total_chars + len(chunk_text) > max_context_chars:
            if i == 1:  # Include at least first chunk
                # Truncate first chunk to fit
                remaining_chars = max_context_chars - total_chars - 200  # Leave room for metadata
                if remaining_chars > 100:
                    truncated_content = node.text[:remaining_chars] + "..."
                    chunk_text = f"""--- Chunk {i} (Score: {score:.4f}) ---
Source: {source}
Content: {truncated_content}
"""
                    formatted_chunks.append(chunk_text)
            break
        
        formatted_chunks.append(chunk_text)
        total_chars += len(chunk_text)
    
    result = "\n".join(formatted_chunks)
    if total_chars > max_context_chars:
        result = result[:max_context_chars] + "\n...[TRUNCATED FOR SPEED]"
    
    # Add source information for agents to use
    import json
    result += f"\n\nSOURCES_INFO: {json.dumps(source_info)}"
    
    # Also save sources_info directly to the JSON file
    try:
        import os
        import glob
        
        # Find the most recent JSON file in the output directory
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
        json_files = glob.glob(os.path.join(output_dir, "*.json"))
        if json_files:
            # Get the most recently created file
            latest_json_file = max(json_files, key=os.path.getctime)
            
            # Read the current JSON file
            with open(latest_json_file, "r", encoding="utf-8") as f:
                output_json = json.load(f)
            
            # Update the sources_info field with the actual data
            output_json["sources_info"] = source_info
            
            # Write back to the JSON file
            with open(latest_json_file, "w", encoding="utf-8") as f:
                json.dump(output_json, f, ensure_ascii=False, indent=2)
            
            print(f"DEBUG: Updated JSON file {latest_json_file} with sources_info: {source_info}")
        else:
            print("DEBUG: No JSON file found to update with sources_info")
    except Exception as e:
        print(f"DEBUG: Error updating JSON file with sources_info: {e}")

    return result
