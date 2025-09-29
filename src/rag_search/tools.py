import os
import json
import glob
import requests
import psycopg2
import logging
from dotenv import load_dotenv
from urllib.parse import urlparse
from typing import Dict, Union, Any, List

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from crewai.tools import tool

# Configure logging for tools
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv()

# Configuration - now configurable via environment variables
PRIMARY_STRATEGY = os.getenv("PRIMARY_STRATEGY", "structure_aware")    # Configurable via docker-compose
FALLBACK_STRATEGY = os.getenv("FALLBACK_STRATEGY", "semantic")         # Configurable via docker-compose
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))        # Made configurable
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "3"))  # Configurable via docker-compose
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))      # Configurable via docker-compose

# LLM timeout configuration
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "300"))     # 5 minutes timeout
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))               # Retry attempts

# Global state
_tool_call_counter = 0
_current_strategy = None


def reset_tool_call_counter():
    """Reset the tool call counter for a new session"""
    global _tool_call_counter
    _tool_call_counter = 0


def set_retrieval_strategy(strategy: str):
    """Set the current retrieval strategy for the session"""
    global _current_strategy
    _current_strategy = strategy
    logger.info(f"Set retrieval strategy to '{strategy}'")


def get_current_strategy() -> str:
    """Get the current strategy, defaulting to PRIMARY_STRATEGY"""
    return _current_strategy or PRIMARY_STRATEGY


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
        logger.warning(f"Could not warm up Ollama model: {e}")
        return False


def get_embedding_model():
    """Create and return configured embedding model - optimized for speed"""
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    
    warm_up_ollama(ollama_base_url, embedding_model)
    
    return OllamaEmbedding(
        model_name=embedding_model,
        base_url=ollama_base_url,
        request_timeout=60.0  # Reduced from 60 to 30 seconds
    )


def get_table_name(strategy: str) -> str:
    """Get table name for given strategy"""
    table_mapping = {
        "structure_aware": "data_llamaindex_enhanced_structure_aware",
        "semantic": "data_llamaindex_enhanced_semantic",
        "contextual_rag": "data_llamaindex_enhanced_contextual_rag"
    }
    
    if strategy not in table_mapping:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(table_mapping.keys())}")
    
    return table_mapping[strategy]


class CustomHybridRetriever:
    """
    Hybrid retriever: semantic (pgvector) + lexical (BM25) with normalization,
    keyword boosting, candidate expansion, and exact-match fallback.
    """

    def __init__(
        self,
        database_url: str,
        table_name: str,
        embed_model,
        alpha: float = 0.5,
        keyword_boost: float = 0.1,
        candidate_multiplier: int = 3,
        max_candidates: int = 50,
        debug: bool = False,
    ):
        self.database_url = database_url
        self.table_name = table_name
        self.embed_model = embed_model
        self.alpha = alpha
        self.keyword_boost = keyword_boost
        self.candidate_multiplier = candidate_multiplier
        self.max_candidates = max_candidates
        self.debug = debug

    # ------------------------
    # Semantic retrieval
    # ------------------------
    def _semantic_retrieve(self, query: str, k: int):
        scores = {}
        try:
            query_embedding = self.embed_model.get_query_embedding(query)
            query_vector_str = "[" + ",".join(map(str, query_embedding)) + "]"

            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor() as cursor:
                    sql = f"""
                        SELECT id, text, metadata_, node_id,
                               embedding <-> %s::vector AS distance
                        FROM {self.table_name}
                        ORDER BY embedding <-> %s::vector
                        LIMIT %s;
                    """
                    cursor.execute(sql, (query_vector_str, query_vector_str, k))
                    rows = cursor.fetchall()

            for id_, text, metadata_, node_id, distance in rows:
                nid = node_id or str(id_)
                sim = 1.0 / (1.0 + (distance if distance is not None else float("inf")))
                scores[nid] = (TextNode(text=text or "", id_=nid, metadata=metadata_ or {}), sim)
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}", exc_info=True)
        return scores

    # ------------------------
    # Lexical retrieval with BM25 + fallback exact match
    # ------------------------
    def _lexical_retrieve(self, query: str, k: int):
        scores = {}
        try:
            with psycopg2.connect(self.database_url) as conn:
                with conn.cursor() as cursor:
                    # BM25 with node_id weighting
                    sql = f"""
                        SELECT id, text, metadata_, node_id,
                               ts_rank(
                                   setweight(to_tsvector('english', coalesce(text, '')), 'A') ||
                                   setweight(to_tsvector('english', coalesce(node_id, '')), 'B'),
                                   websearch_to_tsquery('english', %s),
                                   1
                               ) AS rank
                        FROM {self.table_name}
                        WHERE (to_tsvector('english', coalesce(text, '')) ||
                               to_tsvector('english', coalesce(node_id, '')))
                               @@ websearch_to_tsquery('english', %s)
                        ORDER BY rank DESC
                        LIMIT %s;
                    """
                    cursor.execute(sql, (query, query, k))
                    rows = cursor.fetchall()

            # Fallback to exact substring match if BM25 fails
            if not rows:
                with psycopg2.connect(self.database_url) as conn:
                    with conn.cursor() as cursor:
                        fallback_sql = f"""
                            SELECT id, text, metadata_, node_id, 1.0 as rank
                            FROM {self.table_name}
                            WHERE LOWER(text) LIKE LOWER(%s)
                            LIMIT %s;
                        """
                        cursor.execute(fallback_sql, (f"%{query}%", k))
                        rows = cursor.fetchall()

            if not rows:
                return scores

            max_rank = max([row[4] for row in rows], default=1.0)

            for id_, text, metadata_, node_id, rank in rows:
                nid = node_id or str(id_)
                norm_rank = rank / max_rank
                # keyword boost if query terms appear
                keyword_boost = self.keyword_boost if query.lower() in (text or "").lower() else 0.0
                final_score = norm_rank + keyword_boost
                scores[nid] = (TextNode(text=text or "", id_=nid, metadata=metadata_ or {}), final_score)

        except Exception as e:
            logger.error(f"Lexical retrieval failed: {e}", exc_info=True)

        return scores

    # ------------------------
    # Normalize scores to 0-1
    # ------------------------
    def _min_max_normalize(self, raw):
        if not raw:
            return {}
        values = [v[1] for v in raw.values()]
        min_v, max_v = min(values), max(values)
        rng = max_v - min_v if max_v > min_v else 0.0
        out = {}
        for nid, (node, val) in raw.items():
            norm = (val - min_v) / rng if rng > 0 else (1.0 if val > 0 else 0.0)
            out[nid] = (node, norm)
        return out

    # ------------------------
    # Hybrid retrieve: merge semantic + lexical
    # ------------------------
    def retrieve(self, query: str, similarity_top_k: int = SIMILARITY_TOP_K):
        candidate_k = min(max(similarity_top_k * self.candidate_multiplier, 10), self.max_candidates)

        semantic_raw = self._semantic_retrieve(query, candidate_k)
        lexical_raw = self._lexical_retrieve(query, candidate_k)

        if not semantic_raw and not lexical_raw:
            return []

        sem_norm = self._min_max_normalize(semantic_raw)
        lex_norm = self._min_max_normalize(lexical_raw)

        all_ids = set(sem_norm.keys()) | set(lex_norm.keys())
        merged = {}

        for nid in all_ids:
            node_obj = sem_norm.get(nid, lex_norm.get(nid))[0]
            sem_score = sem_norm.get(nid, (None, 0.0))[1]
            lex_score = lex_norm.get(nid, (None, 0.0))[1]

            final_score = self.alpha * sem_score + (1.0 - self.alpha) * lex_score
            merged[nid] = NodeWithScore(node=node_obj, score=min(1.0, final_score))

        results = sorted(merged.values(), key=lambda x: x.score or 0.0, reverse=True)[:similarity_top_k]

        if self.debug:
            logger.info("Hybrid retrieval debug results:")
            for r in results:
                logger.info(f"Node={r.node.id_} score={r.score:.4f} text_preview={r.node.text[:80]!r}")

        return results


class LlamaIndexRetriever:
    """LlamaIndex-based retriever for fallback scenarios"""
    
    def __init__(self, database_url: str, table_name: str, embed_model):
        self.database_url = database_url
        self.table_name = table_name
        self.embed_model = embed_model
        self._setup_storage()
    
    def _setup_storage(self):
        """Setup LlamaIndex storage context"""
        db_url_parts = urlparse(self.database_url)
        
        # Vector store
        self.vector_store = PGVectorStore.from_params(
            database=db_url_parts.path.lstrip('/'),
            host=db_url_parts.hostname,
            port=db_url_parts.port,
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name=self.table_name,
            embed_dim=int(os.getenv("EMBEDDING_DIM", "768")),
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
    
    def retrieve(self, query: str, similarity_top_k: int = SIMILARITY_TOP_K) -> List[NodeWithScore]:
        """Retrieve documents using LlamaIndex"""
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )
            
            retriever = index.as_retriever(similarity_top_k=similarity_top_k)
            return retriever.retrieve(query)
            
        except Exception as e:
            logger.error(f"LlamaIndex retrieval failed: {e}")
            return []


def get_retriever(strategy: str, use_primary: bool = True, alpha: float = 0.5):
    """Get retriever instance based on strategy and preference"""
    DATABASE_URL = os.getenv("DATABASE_URL")
    table_name = get_table_name(strategy)
    embed_model = get_embedding_model()
    
    if use_primary:
        return CustomHybridRetriever(DATABASE_URL, table_name, embed_model, alpha)
    else:
        return LlamaIndexRetriever(DATABASE_URL, table_name, embed_model)


def try_retrieval(strategy: str, use_primary: bool, query: str, alpha: float = 0.5) -> tuple[List[NodeWithScore], str]:
    """Try retrieval with given configuration, return (nodes, description)"""
    retriever_type = "CustomHybridRetriever" if use_primary else "LlamaIndexRetriever"
    priority = "primary" if use_primary else "fallback"
    
    try:
        retriever = get_retriever(strategy, use_primary, alpha)
        nodes = retriever.retrieve(query, SIMILARITY_TOP_K)
        
        if nodes:
            logger.info(f"{retriever_type} with '{strategy}' strategy succeeded with {len(nodes)} nodes")
            return nodes, f"{strategy} ({priority})"
            
    except Exception as e:
        logger.warning(f"{retriever_type} with '{strategy}' strategy failed: {e}")
    
    return [], ""


def expand_section_query(query: str) -> list:
    """Expand section-level queries to include subsections"""
    queries = [query]  # Always include original query
    
    # Detect section-level queries and add subsection queries
    import re
    if re.match(r'^Section \d+:', query, re.IGNORECASE):
        # Extract section number (e.g., "Section 5:" -> "5")
        section_match = re.match(r'^Section (\d+):', query, re.IGNORECASE)
        if section_match:
            section_num = section_match.group(1)
            # Add subsection queries
            for subsection in ['.1', '.2', '.3', '.4', '.5']:
                queries.append(f"{section_num}{subsection}")
    
    return queries


def retrieve_with_strategy(query: str, strategy: str, alpha: float = None) -> str:
    """Single retrieval attempt - optimized for speed with hybrid retrieval"""
    if alpha is None:
        alpha = float(os.getenv("HYBRID_ALPHA", HYBRID_ALPHA))
    
    # Single retrieval attempt using hybrid retriever
    retriever = get_retriever(strategy, use_primary=True, alpha=alpha)
    nodes = retriever.retrieve(query)  # Use correct method name
    
    if not nodes:
        return "RETRIEVAL TASK COMPLETED: No relevant documents found for your query."
    
    # Filter meaningful content and format results - optimized for speed
    meaningful_nodes = [n for n in nodes if len(n.node.text.strip()) > 50]
    
    if not meaningful_nodes:
        return "RETRIEVAL TASK COMPLETED: No relevant documents found for your query."
    
    # Sort by score and take top results (already sorted by hybrid retriever)
    selected_nodes = meaningful_nodes[:3]  # Use top 3 hybrid-scored documents
    
    # Format results - with balanced approach for quality vs performance
    formatted_results = []
    total_chars = 0
    max_total_chars = int(os.getenv("MAX_CONTEXT_CHARS", "4000"))
    doc_char_limit = 800  # Fixed document limit

    for i, node_with_score in enumerate(selected_nodes, 1):
        text = node_with_score.node.text.strip()
        # Tighter text length limit for faster LLM processing
        if len(text) > doc_char_limit:
            text = text[:doc_char_limit] + "... [truncated for performance]"
        score = node_with_score.score or 0
        doc_text = f"Document {i} (score: {score:.3f}):\n{text}"
        
        # Stop if adding this would exceed our limit
        if total_chars + len(doc_text) > max_total_chars:
            break
            
        formatted_results.append(doc_text)
        total_chars += len(doc_text)
    
    result_text = "\n\n".join(formatted_results)
    
    # Save source info for debugging
    sources_info = {
        'sources': [{'document': n.node.metadata.get('source_document', n.node.metadata.get('file_name', 'Unknown')), 'score': n.score or 0} 
                   for n in selected_nodes],
        'strategy': f"hybrid_{strategy}",
        'chunks_used': len(formatted_results),  # Use actual formatted results count
        'alpha': alpha
    }
    save_sources_info(sources_info)
    
    # Return result with single prefix
    return f"RETRIEVAL TASK COMPLETED:\n\n{result_text}"


def extract_query_from_input(data) -> str:
    """Extract query string from input - improved approach to handle agent context"""
    if isinstance(data, str):
        return data
    if isinstance(data, (int, float)):
        return str(data)
    if isinstance(data, dict):
        # Check common keys for query content
        for key in ['description', 'query', 'q', 'question', 'search_query', 'text', 'content']:
            if key in data:
                value = data[key]
                return str(value) if value is not None else ""
        
        # If no direct query key found, check if there's any string value that looks like a query
        for key, value in data.items():
            if isinstance(value, str) and len(value.strip()) > 10:  # Reasonable query length
                return value.strip()
        
        # Last resort: convert the entire dict to string for debugging
        return str(data)
    
    return ""


def save_sources_info(source_info: dict):
    """Save source information to the latest JSON output file"""
    try:
        output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
        json_files = glob.glob(os.path.join(output_dir, "*.json"))
        
        if json_files:
            latest_json_file = max(json_files, key=os.path.getctime)
            
            with open(latest_json_file, "r", encoding="utf-8") as f:
                output_json = json.load(f)
            
            output_json["sources_info"] = source_info
            
            with open(latest_json_file, "w", encoding="utf-8") as f:
                json.dump(output_json, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        pass  # Silent fail for source info saving


def format_retrieval_output(retrieved_nodes: List[NodeWithScore], strategy_used: str) -> str:
    """Format retrieved nodes into human-readable string"""
    formatted_chunks = []
    source_info = {"sources": [], "strategy": strategy_used, "chunks_used": len(retrieved_nodes)}
    total_chars = 0
    
    for i, node_with_score in enumerate(retrieved_nodes, 1):
        node = node_with_score.node
        score = node_with_score.score
        
        # Extract source from metadata
        source = (node.metadata.get('source_document') or 
                 node.metadata.get('file_name') or 
                 node.metadata.get('source') or 
                 'Unknown source')
        
        source_info["sources"].append({
            "document": source, 
            "score": round(float(score), 4)
        })
        
        chunk_text = f"""--- Chunk {i} (Score: {score:.4f}) ---
Source: {source}
Content: {node.text}

"""
        
        # Check context limit
        if total_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
            if i == 1:  # Always include first chunk
                remaining_chars = MAX_CONTEXT_CHARS - total_chars - 200
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
    
    result = "".join(formatted_chunks)
    if total_chars > MAX_CONTEXT_CHARS:
        result = result[:MAX_CONTEXT_CHARS] + "\n...[TRUNCATED FOR SPEED]"
    
    result += f"\n\nSOURCES_INFO: {json.dumps(source_info)}"
    
    # Save to JSON file
    save_sources_info(source_info)
    
    return result


@tool("Document Retrieval Tool")
def document_retrieval_tool(query: Union[str, Dict[str, Any]]) -> str:
    """
    Retrieve relevant documents for a given query.
    """
    global _tool_call_counter
    _tool_call_counter += 1
    
    # Extract query string from input
    search_query = extract_query_from_input(query)
    logger.info(f"TOOL CALL #{_tool_call_counter}: Document Retrieval Tool called with query: '{search_query[:100]}...'")
    
    if not search_query:
        logger.error(f"Could not extract a valid search query from: {query}")
        return f"Error: Could not extract a valid search query from: {query}"

    current_strategy = get_current_strategy()
    logger.info(f"Using retrieval strategy: {current_strategy}")
    
    result = retrieve_with_strategy(search_query, current_strategy)
    logger.info(f"TOOL CALL #{_tool_call_counter}: Document retrieval completed, returned {len(result)} characters")
    return result