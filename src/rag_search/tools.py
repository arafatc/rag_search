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
# Use enhanced custom retriever as primary for better performance and accuracy
PRIMARY_STRATEGY = "custom"  # Use enhanced CustomPGVectorRetriever as primary
FALLBACK_STRATEGY = "semantic"  # LlamaIndex as backup only

# Global strategy override for multi-strategy support
_current_strategy = None

def set_retrieval_strategy(strategy: str):
    """Set the current retrieval strategy for the session"""
    global _current_strategy
    _current_strategy = strategy
    print(f"INFO: Set retrieval strategy to '{strategy}'")

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
        try:
            self.vector_store = PGVectorStore.from_params(
                database=db_url_parts.path.lstrip('/'),
                host=db_url_parts.hostname,
                port=db_url_parts.port,
                user=db_url_parts.username,
                password=db_url_parts.password,
                table_name=self.table_name,
                embed_dim=int(os.getenv("EMBEDDING_DIM", "768")),  # Use externalized embedding dimension
            )
            print(f"DEBUG: PGVectorStore created successfully for table: {self.table_name}")
        except Exception as e:
            print(f"ERROR: Failed to create PGVectorStore: {e}")
            raise
        
        # Try to create document store, but don't fail if it doesn't work
        self.doc_store = None
        try:
            # For LlamaIndex, docstore table is typically table_name + "_docstore" 
            # but since we don't have separate docstore tables, skip it
            doc_table_name = self.table_name + "_docstore"
            # First check if the docstore table exists
            import psycopg2
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename = %s
                );
            """, (doc_table_name,))
            table_exists = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            if table_exists:
                self.doc_store = PostgresDocumentStore.from_params(
                    database=db_url_parts.path.lstrip('/'),
                    host=db_url_parts.hostname,
                    port=db_url_parts.port,
                    user=db_url_parts.username,
                    password=db_url_parts.password,
                    table_name=doc_table_name
                )
                print(f"DEBUG: Using existing docstore table: {doc_table_name}")
            else:
                print(f"DEBUG: Docstore table '{doc_table_name}' doesn't exist, proceeding without docstore")
                
        except Exception as e:
            print(f"DEBUG: Could not initialize docstore: {e}")
        
        # Storage context - only include docstore if it exists
        if self.doc_store:
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                docstore=self.doc_store
            )
        else:
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
    
    def retrieve(self, query: str, similarity_top_k: int = 3) -> List[NodeWithScore]:
        """Retrieve documents using LlamaIndex"""
        try:
            print(f"DEBUG: LlamaIndex retrieving for query: '{query[:50]}...'")
            
            # Create index from storage
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )
            
            print(f"DEBUG: Created VectorStoreIndex successfully")
            
            # Create retriever
            retriever = index.as_retriever(similarity_top_k=similarity_top_k)
            
            print(f"DEBUG: Created retriever, attempting to retrieve...")
            
            # Retrieve
            nodes = retriever.retrieve(query)
            
            print(f"DEBUG: LlamaIndex retrieved {len(nodes)} nodes")
            return nodes
            
        except Exception as e:
            print(f"ERROR: LlamaIndex retrieval failed: {e}")
            import traceback
            traceback.print_exc()
            return []

class CustomPGVectorRetriever:
    """Enhanced PostgreSQL vector retriever with hybrid keyword + vector search"""
    
    def __init__(self, database_url: str, table_name: str, embed_model):
        self.database_url = database_url
        self.table_name = table_name
        self.embed_model = embed_model
        print(f"DEBUG: CustomPGVectorRetriever initialized with table_name: '{table_name}'")
    
    def extract_article_number(self, query: str) -> str:
        """Extract article number from query"""
        import re
        match = re.search(r'article\s*\(?\s*(\d+)\s*\)?', query.lower())
        return match.group(1) if match else None
    
    def extract_topic_keywords(self, query: str) -> List[str]:
        """Extract topic-specific keywords that might map to articles"""
        query_lower = query.lower()
        
        # Map topic keywords to article-related terms for boosting
        topic_mappings = {
            'absence by permission': ['absence by permission', 'permission hours', 'study leave'],
            'delegation of power': ['delegation of power', 'delegation of powers', 'delegated authorities'],
            'compliance': ['compliance with applicable', 'compliance with legislations'],
            'modification of status': ['modification of status', 'employee status modification']
        }
        
        found_topics = []
        for topic, keywords in topic_mappings.items():
            if any(keyword in query_lower for keyword in keywords):
                found_topics.extend(keywords)
        
        return found_topics
    
    def keyword_filter_results(self, query: str, article_num: str = None, topic_keywords: List[str] = None, 
                              all_results: List = None, vec_weight: float = 0.7, kw_weight: float = 0.3) -> List:
        """Filter and boost results using binary keyword scoring with weighted combination"""
        if not article_num and not topic_keywords:
            return all_results
            
        # Build search patterns
        search_patterns = []
        
        # Article number patterns (if available)
        if article_num:
            search_patterns.extend([
                f'article ({article_num}):',
                f'article ({article_num}) ',
                f'article {article_num}:',
                f'article {article_num} ',
            ])
        
        # Topic keyword patterns (if available)
        if topic_keywords:
            search_patterns.extend([kw.lower() for kw in topic_keywords])
        
        # Calculate hybrid scores for all results
        scored_results = []
        for result in all_results:
            text_lower = result[1].lower()  # text is at index 1
            distance = result[4]  # distance is at index 4
            
            # Calculate vector similarity (lower distance = higher similarity)
            vec_sim = 1.0 / (1.0 + distance)  # Convert distance to similarity
            
            # Calculate keyword score (binary: 1 if match, 0 if no match)
            kw_score = 1.0 if any(pattern in text_lower for pattern in search_patterns) else 0.0
            
            # Combine scores with weights
            final_score = (vec_weight * vec_sim) + (kw_weight * kw_score)
            
            # Create new result tuple with final score
            scored_result = list(result)
            scored_result.append(final_score)  # Add final_score as last element
            scored_results.append(tuple(scored_result))
            
            # Debug info for keyword matches
            if kw_score > 0:
                match_type = "Article" if article_num and any(f'article' in pattern for pattern in search_patterns if pattern in text_lower) else "Topic"
                source = result[2].get('file_name', 'Unknown') if result[2] else 'Unknown'
                print(f"DEBUG: {match_type} match - Vec: {vec_sim:.3f}, KW: {kw_score:.1f}, Final: {final_score:.3f} - {source}")
        
        # Sort by final score (descending - higher is better)
        scored_results.sort(key=lambda x: x[-1], reverse=True)
        
        # Remove the final_score from results to maintain original format
        return [result[:-1] for result in scored_results]
    
    def retrieve(self, query: str, similarity_top_k: int = 3) -> List[NodeWithScore]:
        """Enhanced retrieval with keyword boosting for article and topic queries"""
        try:
            # Extract article number and topic keywords
            article_num = self.extract_article_number(query)
            topic_keywords = self.extract_topic_keywords(query)
            
            if article_num:
                print(f"DEBUG: Detected Article {article_num} query, will boost matching results")
            if topic_keywords:
                print(f"DEBUG: Detected topic keywords: {topic_keywords}, will boost matching results")
            
            # Generate query embedding with timeout handling
            print(f"DEBUG: Generating embedding for query length: {len(query)} characters")
            try:
                query_embedding = self.embed_model.get_query_embedding(query)
                print(f"DEBUG: Successfully generated embedding with {len(query_embedding)} dimensions")
            except Exception as e:
                print(f"ERROR: Embedding generation failed: {e}")
                raise
            
            query_vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Connect to database
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Perform vector similarity search (get fewer results for better performance)
            search_limit = similarity_top_k * 2 if (article_num or topic_keywords) else similarity_top_k
            
            print(f"DEBUG: Executing optimized query with search_limit: {search_limit}")
            query_sql = """
                SELECT id, text, metadata_, node_id, embedding <-> %s::vector as distance
                FROM {}
                ORDER BY embedding <-> %s::vector
                LIMIT %s;
            """.format(self.table_name)
            
            print(f"DEBUG: Generated SQL: {query_sql[:100]}...")
            cursor.execute(query_sql, (query_vector_str, query_vector_str, search_limit))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            print(f"DEBUG: CustomPGVectorRetriever found {len(results)} initial results")
            
            # Apply keyword filtering and boosting
            if article_num or topic_keywords:
                results = self.keyword_filter_results(
                    query=query, 
                    article_num=article_num, 
                    topic_keywords=topic_keywords, 
                    all_results=results,
                    vec_weight=0.7,  # 70% vector similarity
                    kw_weight=0.3    # 30% keyword match
                )
                print(f"DEBUG: After keyword boosting: {len(results)} results")
            
            # Convert to NodeWithScore objects (take only top similarity_top_k)
            nodes = []
            for row in results[:similarity_top_k]:
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

def retrieve_with_strategy(query: str, strategy: str) -> str:
    """
    Retrieve documents using a specific strategy with strategy-specific fallback
    """
    print(f"INFO: Executing document retrieval for extracted query: '{query}'")
    
    # Try primary strategy first (now enhanced custom retriever)
    try:
        print(f"INFO: Using PRIMARY strategy: '{strategy}'")
        primary_retriever = get_retriever(strategy)
        retrieved_nodes = primary_retriever.retrieve(query)
        strategy_used = strategy
        
        if retrieved_nodes:
            print(f"INFO: Primary strategy '{strategy}' succeeded with {len(retrieved_nodes)} nodes")
            # Debug source attribution - try multiple metadata fields
            sources = [
                node.node.metadata.get('source') or 
                node.node.metadata.get('source_document') or 
                node.node.metadata.get('file_name') or 
                'Unknown' 
                for node in retrieved_nodes
            ]
            print(f"INFO: Retrieved sources: {list(set(sources))}")
            return format_retrieval_output(retrieved_nodes, strategy_used)
        else:
            print(f"WARN: Primary strategy '{strategy}' returned no results")
    except Exception as e:
        print(f"ERROR: Primary strategy '{strategy}' failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback to alternative strategy
    fallback_strategy = f"custom_{strategy}" if strategy != "custom" else FALLBACK_STRATEGY
    print(f"INFO: Primary failed. Using FALLBACK strategy: '{fallback_strategy}'")
    
    try:
        fallback_retriever = get_retriever(fallback_strategy)
        retrieved_nodes = fallback_retriever.retrieve(query)
        strategy_used = fallback_strategy
        
        if retrieved_nodes:
            print(f"INFO: Fallback strategy '{fallback_strategy}' succeeded with {len(retrieved_nodes)} nodes")
            # Debug source attribution - try multiple metadata fields
            sources = [
                node.node.metadata.get('source') or 
                node.node.metadata.get('source_document') or 
                node.node.metadata.get('file_name') or 
                'Unknown' 
                for node in retrieved_nodes
            ]
            print(f"INFO: Retrieved sources: {list(set(sources))}")
            return format_retrieval_output(retrieved_nodes, strategy_used)
        else:
            print(f"WARN: Fallback strategy '{fallback_strategy}' returned no results")
    except Exception as e:
        print(f"ERROR: Fallback strategy '{fallback_strategy}' failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    # Last resort - try semantic custom as universal fallback
    if fallback_strategy != "custom":
        print("INFO: Trying semantic custom as last resort.")
        try:
            last_resort_retriever = get_retriever("custom")
            retrieved_nodes = last_resort_retriever.retrieve(query)
            strategy_used = "custom"
            
            if retrieved_nodes:
                return format_retrieval_output(retrieved_nodes, strategy_used)
        except Exception as e:
            print(f"ERROR: Last resort strategy also failed: {e}")
    
    return "Error: All retrieval strategies failed."


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
    
    # Use current strategy with proper fallback logic
    current_strategy = get_current_strategy()
    print(f"DEBUG: Using retrieval strategy: '{current_strategy}'")
    
    # Use the same strategy-aware retrieval logic as retrieve_with_strategy
    return retrieve_with_strategy(search_query, current_strategy)


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

    if strategy in ["semantic", "hierarchical", "contextual_rag"]:
        return LlamaIndexRetriever(
            database_url=DATABASE_URL,
            table_name=f"data_llamaindex_enhanced_{strategy}",
            embed_model=embed_model
        )
    elif strategy == "custom":
        # Use enhanced CustomPGVectorRetriever with hybrid capabilities
        return CustomPGVectorRetriever(
            database_url=DATABASE_URL,
            table_name="data_llamaindex_enhanced_semantic",
            embed_model=embed_model
        )
    elif strategy.startswith("custom_"):
        # Support custom with specific strategy: custom_hierarchical, custom_contextual_rag
        base_strategy = strategy.replace("custom_", "")
        return CustomPGVectorRetriever(
            database_url=DATABASE_URL,
            table_name=f"data_llamaindex_enhanced_{base_strategy}",
            embed_model=embed_model
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Available strategies: semantic, hierarchical, contextual_rag, custom, custom_hierarchical, custom_contextual_rag")

def format_retrieval_output(retrieved_nodes: List[NodeWithScore], strategy_used: str) -> str:
    """Format the retrieved nodes into a human-readable string"""
    formatted_chunks = []
    source_info = {"sources": [], "strategy": strategy_used, "chunks_used": len(retrieved_nodes)}
    total_chars = 0
    max_context_chars = 1800  # Further optimized for Gemma 3:1b performance and timeout prevention
    
    for i, node_with_score in enumerate(retrieved_nodes, 1):
        node = node_with_score.node
        score = node_with_score.score
        
        # Extract source from metadata - try multiple possible keys in order of preference
        source = (node.metadata.get('source') or 
                 node.metadata.get('source_document') or 
                 node.metadata.get('file_name') or 
                 node.metadata.get('source_citation') or 
                 'Unknown source')
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
