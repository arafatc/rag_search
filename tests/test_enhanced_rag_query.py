#!/usr/bin/env python3
"""
Enhanced RAG Query Script - Test semantic retrieval from the enhanced ingestion
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv(override=True)  # Added override=True to force reload

# Initialize Phoenix tracing for this test session
try:
    from phoenix.otel import register
    from opentelemetry import trace
    
    # Initialize Phoenix WITHOUT auto-instrumentation to avoid conflicts
    tracer_provider = register(
        project_name="rag_system",
        endpoint="http://localhost:6006/v1/traces",
        auto_instrument=False  # Changed to False to avoid embedding conflicts
    )
    
    # Get tracer for creating spans
    tracer = trace.get_tracer(__name__)
    
    def log_interaction(query: str, response: str, metadata=None):
        """Create actual Phoenix spans for tracing"""
        with tracer.start_as_current_span("rag_interaction") as span:
            span.set_attribute("query.text", query[:200])
            span.set_attribute("response.length", len(response) if response else 0)
            span.set_attribute("interaction.type", "test")
            
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"metadata.{key}", str(value))
    
    PHOENIX_AVAILABLE = True
    print("INFO: Phoenix tracing initialized for RAG testing")
except ImportError:
    PHOENIX_AVAILABLE = False
    def log_interaction(query: str, response: str, metadata=None):
        pass
    print("WARNING: Phoenix not available - continuing without observability")

# Import the shared query engine
from src.rag_search.query_engine import setup_enhanced_rag_query_engine

# Database connection for statistics
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuration from environment
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "rag_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def setup_enhanced_rag_query_engine_with_logging(strategy="semantic"):
    """Setup the enhanced RAG query engine with detailed logging for tests"""
    print(f"CHECKING: Setting up Enhanced RAG Query Engine for: {strategy}")
    
    try:
        # Use the shared module but add test-specific logging
        query_engine, index = setup_enhanced_rag_query_engine(strategy)
        
        # Add test-specific logging
        from src.rag_search.query_engine import (
            LLM_PROVIDER, EMBEDDING_PROVIDER, LLM_MODEL, EMBEDDING_MODEL,
            OPENAI_MODEL, OPENAI_EMBEDDING_MODEL, get_embedding_dim
        )
        
        # Log provider information for tests
        if EMBEDDING_PROVIDER.lower() == "ollama":
            print(f"SUCCESS: Using Ollama embedding: {EMBEDDING_MODEL}")
        else:
            print(f"SUCCESS: Using OpenAI embedding: {OPENAI_EMBEDDING_MODEL}")
        
        if LLM_PROVIDER.lower() == "ollama":
            print(f"SUCCESS: Using Ollama LLM: {LLM_MODEL}")
        else:
            print(f"SUCCESS: Using OpenAI LLM: {OPENAI_MODEL}")
        
        # Log table information
        vector_table = f"llamaindex_enhanced_{strategy.lower()}"
        docstore_table = f"llamaindex_enhanced_docstore_{strategy.lower()}"
        embed_dim = get_embedding_dim()
        
        print(f"INFO: Connecting to strategy-specific tables:")
        print(f"   Vector: {vector_table}")
        print(f"   DocStore: {docstore_table}")
        print(f"   Embedding Dimension: {embed_dim}")
        
        print(f"SUCCESS: Enhanced RAG Query Engine ready for {strategy} strategy!")
        return query_engine, index
        
    except Exception as e:
        print(f"ERROR: Failed to setup query engine: {e}")
        return None, None

def run_enhanced_rag_queries(show_full_answers=True):
    """Run sample queries to test the enhanced RAG system"""
    print("STARTING: Enhanced RAG Strategy Comparison Test")
    print("=" * 60)
    
    # Test strategies available
    strategies = ["semantic", "hierarchical"]
    
    # Test query
    test_query = "What are the main HR policies and employee bylaws mentioned?"
    
    print(f"\nCHECKING: Testing Query: {test_query}")
    print("=" * 60)
    
    results = {}
    
    for strategy in strategies:
        print(f"\nFOCUS: Testing {strategy.upper()} Strategy:")
        print("-" * 40)
        
        try:
            # Setup query engine for this strategy
            query_engine, index = setup_enhanced_rag_query_engine(strategy)
            
            # Get chunk count from database
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, user=PG_USER, 
                password=PG_PASSWORD, database=PG_DATABASE
            )
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            table_name = f"data_llamaindex_enhanced_{strategy}"
            try:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                chunk_count = cursor.fetchone()['count']
            except:
                chunk_count = 0
            
            cursor.close()
            conn.close()
            
            print(f"INFO: Strategy Statistics:")
            print(f"   Chunks Available: {chunk_count}")
            
            if chunk_count > 0:
                # Run query
                start_time = time.time()
                response = query_engine.query(test_query)
                query_time = time.time() - start_time
                
                # Log interaction with Phoenix if available
                if PHOENIX_AVAILABLE:
                    log_interaction(
                        query=test_query,
                        response=str(response),
                        metadata={
                            'strategy': strategy,
                            'chunks_available': chunk_count,
                            'query_time': query_time,
                            'sources_used': len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
                        }
                    )
                
                print(f"TIME: Response Time: {query_time:.2f}s")
                print(f"NOTE: Answer Length: {len(str(response))} chars")
                
                # Show source information
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    print(f"CITATION: Sources Found: {len(response.source_nodes)}")
                    
                    # Show first few sources with simple formatting
                    for i, node in enumerate(response.source_nodes[:3], 1):
                        if hasattr(node, 'metadata') and node.metadata:
                            doc_name = node.metadata.get('file_name', 'Unknown')
                            doc_type = node.metadata.get('file_extension', 'Document')
                            print(f"   {i}. {doc_name} ({doc_type})")
                    
                    # Show detailed metadata from first source
                    first_source = response.source_nodes[0]
                    if hasattr(first_source, 'metadata') and first_source.metadata:
                        print(f"CITATION: Detailed Source Info:")
                        print(f"   Document: {first_source.metadata.get('file_name', 'Unknown')}")
                        print(f"   File Path: {first_source.metadata.get('file_path', 'Unknown')}")
                        print(f"   File Size: {first_source.metadata.get('file_size', 'Unknown')} bytes")
                        print(f"   Strategy: {first_source.metadata.get('chunking_strategy', 'Unknown')}")
                        print(f"   Processed: {first_source.metadata.get('processed_at', 'Unknown')}")
                        print(f"   Content Length: {first_source.metadata.get('content_length', 'Unknown')} chars")
                
                if show_full_answers:
                    print(f" Full Answer:")
                    print("-" * 40)
                    print(str(response))
                    print("-" * 40)
                else:
                    print(f"CHECKING: Answer Preview: {str(response)[:200]}...")
                
                # Store results for comparison
                results[strategy] = {
                    'chunks': chunk_count,
                    'time': query_time,
                    'response': str(response),  # Store full response
                    'sources': len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
                }
            else:
                print(f"WARNING: No chunks found for {strategy} strategy")
                results[strategy] = {'chunks': 0, 'error': 'No data'}
                
        except Exception as e:
            print(f"ERROR: Error testing {strategy}: {e}")
            results[strategy] = {'error': str(e)}
    
    # Print comparison summary
    print(f"\nINFO: STRATEGY COMPARISON SUMMARY")
    print("=" * 60)
    
    for strategy, data in results.items():
        if 'error' not in data:
            print(f"\nFOCUS: {strategy.upper()}:")
            print(f"   Chunks: {data['chunks']:,}")
            print(f"   Response Time: {data['time']:.2f}s") 
            print(f"   Sources Used: {data['sources']}")
            print(f"   Answer Quality: {'High' if len(data['response']) > 200 else 'Low'}")
        else:
            print(f"\nERROR: {strategy.upper()}: {data['error']}")

def run_single_strategy_query(strategy, query, show_full_answers=True):
    """Run a single query on a specific strategy"""
    print(f"STARTING: Enhanced RAG Query Test - {strategy.upper()} Strategy")
    print("=" * 60)
    
    print(f"\nCHECKING: Testing Query: {query}")
    print("-" * 40)
    
    try:
        # Setup query engine for this strategy
        query_engine, index = setup_enhanced_rag_query_engine(strategy)
        
        # Get chunk count from database
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, user=PG_USER, 
            password=PG_PASSWORD, database=PG_DATABASE
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        table_name = f"data_llamaindex_enhanced_{strategy}"
        try:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            chunk_count = cursor.fetchone()['count']
        except:
            chunk_count = 0
        
        cursor.close()
        conn.close()
        
        print(f"INFO: Strategy Statistics:")
        print(f"   Chunks Available: {chunk_count:,}")
        
        if chunk_count > 0:
            # Run query
            start_time = time.time()
            response = query_engine.query(query)
            query_time = time.time() - start_time
            
            # Log interaction with Phoenix if available
            if PHOENIX_AVAILABLE:
                log_interaction(
                    query=query,
                    response=str(response),
                    metadata={
                        'strategy': strategy,
                        'chunks_available': chunk_count,
                        'query_time': query_time,
                        'sources_used': len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
                    }
                )
            
            print(f"TIME: Response Time: {query_time:.2f}s")
            print(f"NOTE: Answer Length: {len(str(response))} chars")
            print(f" Sources Used: {len(response.source_nodes) if hasattr(response, 'source_nodes') else 0}")
            
            if show_full_answers:
                print(f"\nDOCUMENT: Full Answer:")
                print("=" * 60)
                print(str(response))
                print("=" * 60)
            else:
                print(f"CHECKING: Answer Preview: {str(response)[:200]}...")
                
        else:
            print(f"WARNING: No chunks found for {strategy} strategy")
            
    except Exception as e:
        print(f"ERROR: Error testing {strategy}: {e}")
        import traceback
        print(traceback.format_exc())

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Enhanced RAG Query System')
    parser.add_argument('--strategy', type=str, choices=['semantic', 'hierarchical', 'both'], 
                       default='both', help='Strategy to test')
    parser.add_argument('--query', type=str, 
                       default="What are the main HR policies and employee bylaws mentioned?",
                       help='Query to test')
    parser.add_argument('--full', action='store_true', default=True,
                       help='Show full answers (default: True)')
    parser.add_argument('--preview', action='store_true', default=False,
                       help='Show only answer previews')
    
    args = parser.parse_args()
    
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found. Please set your OpenAI API key in .env file.")
        return False
    
    try:
        if args.strategy == 'both':
            run_enhanced_rag_queries(show_full_answers=not args.preview)
        else:
            # Single strategy test
            run_single_strategy_query(args.strategy, args.query, show_full_answers=not args.preview)
        return True
    except Exception as e:
        print(f"ERROR: Enhanced RAG query test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
