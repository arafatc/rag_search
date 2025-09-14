#!/usr/bin/env python3
"""
Enhanced RAG API Test Script
Tests RAG functionality using FastAPI /v1/chat/completions endpoint
with different strategies and comprehensive logging
"""

import os
import sys
import time
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)  # Added override=True to force reload

# Initialize Phoenix tracing for this test session
try:
    from phoenix.otel import register
    from opentelemetry import trace
    
    # Initialize Phoenix with proper tracing
    tracer_provider = register(
        project_name="rag_api_test",
        endpoint="http://localhost:6006/v1/traces",
        auto_instrument=False  # Changed to False to avoid embedding conflicts
    )
    
    # Get tracer for creating spans
    tracer = trace.get_tracer(__name__)
    
    def log_api_interaction(query: str, response: str, metadata=None):
        """Create actual Phoenix spans for tracing API calls"""
        with tracer.start_as_current_span("api_rag_interaction") as span:
            span.set_attribute("query.text", query[:200])
            span.set_attribute("response.length", len(response) if response else 0)
            span.set_attribute("interaction.type", "api_test")
            span.set_attribute("api.endpoint", "/v1/chat/completions")
            
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"metadata.{key}", str(value))
    
    PHOENIX_AVAILABLE = True
    print("INFO: Phoenix tracing initialized for RAG API testing")
except ImportError:
    PHOENIX_AVAILABLE = False
    def log_api_interaction(query: str, response: str, metadata=None):
        pass
    print("WARNING: Phoenix not available - continuing without observability")

# Database connection for statistics
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuration from environment
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "rag_db")

def get_chunk_statistics(strategy=None):
    """Get chunk statistics from the database for a given strategy"""
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, user=PG_USER, 
            password=PG_PASSWORD, database=PG_DATABASE
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if strategy:
            table_name = f"data_llamaindex_enhanced_{strategy}"
            try:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                chunk_count = cursor.fetchone()['count']
                cursor.close()
                conn.close()
                return chunk_count
            except:
                cursor.close()
                conn.close()
                return 0
        else:
            # Get statistics for all strategies
            stats = {}
            strategies = ["semantic", "hierarchical"]
            for strat in strategies:
                table_name = f"data_llamaindex_enhanced_{strat}"
                try:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                    stats[strat] = cursor.fetchone()['count']
                except:
                    stats[strat] = 0
            
            cursor.close()
            conn.close()
            return stats
            
    except Exception as e:
        print(f"WARNING: Could not get database statistics: {e}")
        return 0 if strategy else {}

def test_basic_api_endpoints():
    """Test basic API endpoints for functionality"""
    base_url = "http://localhost:8001"
    
    print("STARTING: Basic API Endpoint Tests")
    print("=" * 60)
    
    # Test models endpoint (OpenWebUI compatibility)
    try:
        response = requests.get(f"{base_url}/v1/models")
        models = response.json()
        print("SUCCESS: Models Endpoint (OpenWebUI):")
        print(f"   Available models: {len(models['data'])}")
        for model in models['data']:
            print(f"   - {model['id']} (max_tokens: {model['max_tokens']})")
        return True
    except Exception as e:
        print(f"ERROR: Models endpoint failed: {e}")
        print("CRITICAL: API Server may not be running on port 8001")
        return False

def test_enhanced_rag_via_api(show_full_answers=True):
    """Test enhanced RAG functionality via /v1/chat/completions endpoint"""
    base_url = "http://localhost:8001"
    
    if not test_basic_api_endpoints():
        return False
    
    print("\nSTARTING: Enhanced RAG Strategy Testing via API")
    print("=" * 60)
    
    # Get database statistics
    chunk_stats = get_chunk_statistics()
    print(f"\nINFO: Database Statistics:")
    for strategy, count in chunk_stats.items():
        print(f"   {strategy.upper()}: {count:,} chunks")
    
    # Test queries - similar to test_enhanced_rag_query.py
    test_queries = [
        "If Im a non-citizen, what are the rules for me to get hired in a government job?",
        "How long is the probation period for new employees and what happens if I dont pass it?",
        "What basic principles guide government procurement in Abu Dhabi?",
        "What ethical rules do procurement officers need to follow?",
        "How are conflicts of interest managed for employees and also for procurement staff?",
        "What do the HR bylaws say about hiring employees and what do the procurement standards say about terminating suppliers?"
    ]
    
    results = {}
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nFOCUS: Testing Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Prepare chat completion request
            chat_data = {
                "model": "rag-search",
                "messages": [
                    {"role": "user", "content": query}
                ]
            }
            
            # Make API request
            start_time = time.time()
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=chat_data,
                timeout=180  # 3 minute timeout for RAG processing
            )
            query_time = time.time() - start_time
            
            if response.status_code == 200:
                chat_result = response.json()
                response_content = chat_result['choices'][0]['message']['content']
                
                # Parse the JSON response to extract structured data
                try:
                    import json
                    enhanced_data = json.loads(response_content)
                    answer = enhanced_data.get('answer', response_content)
                    strategy_used = enhanced_data.get('strategy_used', 'unknown')
                    source_info = enhanced_data.get('source_info', [])
                    raw_context_length = enhanced_data.get('raw_context_length', 0)
                except (json.JSONDecodeError, TypeError):
                    # Fallback for non-JSON responses
                    answer = response_content
                    strategy_used = 'unknown'
                    source_info = []
                    raw_context_length = 0
                
                # Log interaction with Phoenix if available
                if PHOENIX_AVAILABLE:
                    log_api_interaction(
                        query=query,
                        response=answer,
                        metadata={
                            'query_number': i,
                            'query_time': query_time,
                            'status_code': response.status_code,
                            'answer_length': len(answer),
                            'strategy_used': strategy_used,
                            'sources_count': len(source_info)
                        }
                    )
                
                print(f"FOCUS: Testing {strategy_used.upper()} Strategy:")
                print(f"SUCCESS: API Response received")
                print(f"TIME: Response Time: {query_time:.2f}s")
                print(f"NOTE: Answer Length: {len(answer)} chars")
                
                # Display citation information
                if source_info:
                    print(f"CITATION: Sources Found: {len(source_info)}")
                    for i, source in enumerate(source_info, 1):
                        doc_name = source.get('document', 'Unknown')
                        print(f"   {i}. {doc_name} (.pdf)")
                    
                    print("CITATION: Detailed Source Info:")
                    for source in source_info:
                        print(f"   Document: {source.get('document', 'Unknown')}")
                        print(f"   File Path: {source.get('file_path', 'Unknown')}")
                        print(f"   File Size: {source.get('file_size', 'Unknown')} bytes")
                        print(f"   Strategy: {source.get('strategy', 'unknown')}")
                        print(f"   Processed: {source.get('processed', 'Unknown')}")
                        print(f"   Content Length: {source.get('content_length', 0)} chars")
                
                if show_full_answers:
                    print(f"DOCUMENT: Full Answer:")
                    print("=" * 60)
                    print(answer)
                    print("=" * 60)
                else:
                    print(f"CHECKING: Answer Preview: {answer[:200]}...")
                
                # Store results for analysis
                results[f"query_{i}"] = {
                    'query': query,
                    'time': query_time,
                    'success': True,
                    'answer_length': len(answer),
                    'answer': answer
                }
                
            else:
                print(f"ERROR: API request failed with status {response.status_code}")
                print(f"ERROR: Response: {response.text}")
                results[f"query_{i}"] = {
                    'query': query,
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.Timeout:
            print(f"ERROR: API request timed out after 180 seconds")
            results[f"query_{i}"] = {
                'query': query,
                'success': False,
                'error': "Timeout after 180 seconds"
            }
        except Exception as e:
            print(f"ERROR: API request failed: {e}")
            results[f"query_{i}"] = {
                'query': query,
                'success': False,
                'error': str(e)
            }
    
    # Print summary results
    print(f"\nINFO: RAG API TEST SUMMARY")
    print("=" * 60)
    
    successful_queries = sum(1 for result in results.values() if result.get('success', False))
    total_queries = len(results)
    
    print(f"METRICS: Success Rate: {successful_queries}/{total_queries}")
    
    if successful_queries > 0:
        successful_results = [r for r in results.values() if r.get('success', False)]
        avg_time = sum(r['time'] for r in successful_results) / len(successful_results)
        avg_length = sum(r['answer_length'] for r in successful_results) / len(successful_results)
        
        print(f"METRICS: Average Response Time: {avg_time:.2f}s")
        print(f"METRICS: Average Answer Length: {avg_length:.0f} chars")
    
    # Show individual results
    for key, result in results.items():
        if result.get('success', False):
            print(f"\nSUCCESS: {key}:")
            print(f"   Query: {result['query'][:60]}...")
            print(f"   Time: {result['time']:.2f}s")
            print(f"   Length: {result['answer_length']} chars")
        else:
            print(f"\nERROR: {key}:")
            print(f"   Query: {result['query'][:60]}...")
            print(f"   Error: {result['error']}")
    
    return successful_queries > 0

def test_single_query_via_api(query, show_full_answer=True):
    """Test a single query via the API endpoint"""
    base_url = "http://localhost:8001"
    
    print(f"STARTING: Single RAG Query Test via API")
    print("=" * 60)
    print(f"\nCHECKING: Testing Query: {query}")
    
    if not test_basic_api_endpoints():
        return False
    
    try:
        # Get database statistics
        chunk_stats = get_chunk_statistics()
        print(f"\nINFO: Database Statistics:")
        for strategy, count in chunk_stats.items():
            print(f"   {strategy.upper()}: {count:,} chunks")
        
        # Prepare chat completion request
        chat_data = {
            "model": "rag-search",
            "messages": [
                {"role": "user", "content": query}
            ]
        }
        
        # Make API request
        print(f"\nINFO: Sending request to {base_url}/v1/chat/completions")
        start_time = time.time()
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=chat_data,
            timeout=180  # 3 minute timeout for RAG processing
        )
        query_time = time.time() - start_time
        
        if response.status_code == 200:
            chat_result = response.json()
            response_content = chat_result['choices'][0]['message']['content']
            
            # Parse the JSON response to extract structured data
            try:
                import json
                enhanced_data = json.loads(response_content)
                answer = enhanced_data.get('answer', response_content)
                strategy_used = enhanced_data.get('strategy_used', 'unknown')
                source_info = enhanced_data.get('source_info', [])
                raw_context_length = enhanced_data.get('raw_context_length', 0)
            except (json.JSONDecodeError, TypeError):
                # Fallback for non-JSON responses
                answer = response_content
                strategy_used = 'unknown'
                source_info = []
                raw_context_length = 0
            
            # Log interaction with Phoenix if available
            if PHOENIX_AVAILABLE:
                log_api_interaction(
                    query=query,
                    response=answer,
                    metadata={
                        'query_time': query_time,
                        'status_code': response.status_code,
                        'answer_length': len(answer),
                        'strategy_used': strategy_used,
                        'sources_count': len(source_info)
                    }
                )
            
            print(f"FOCUS: Testing {strategy_used.upper()} Strategy:")
            print(f"SUCCESS: API Response received")
            print(f"TIME: Response Time: {query_time:.2f}s")
            print(f"NOTE: Answer Length: {len(answer)} chars")
            
            # Display citation information
            if source_info:
                print(f"CITATION: Sources Found: {len(source_info)}")
                for i, source in enumerate(source_info, 1):
                    doc_name = source.get('document', 'Unknown')
                    print(f"   {i}. {doc_name} (.pdf)")
                
                print("CITATION: Detailed Source Info:")
                for source in source_info:
                    print(f"   Document: {source.get('document', 'Unknown')}")
                    print(f"   File Path: {source.get('file_path', 'Unknown')}")
                    print(f"   File Size: {source.get('file_size', 'Unknown')} bytes")
                    print(f"   Strategy: {source.get('strategy', 'unknown')}")
                    print(f"   Processed: {source.get('processed', 'Unknown')}")
                    print(f"   Content Length: {source.get('content_length', 0)} chars")
            
            if show_full_answer:
                print(f"DOCUMENT: Full Answer:")
                print("=" * 60)
                print(answer)
                print("=" * 60)
            else:
                print(f"CHECKING: Answer Preview: {answer[:200]}...")
            
            return True
            
        else:
            print(f"ERROR: API request failed with status {response.status_code}")
            print(f"ERROR: Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR: API request failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Main function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Enhanced RAG via FastAPI')
    parser.add_argument('--mode', type=str, choices=['single', 'multiple', 'basic'], 
                       default='multiple', help='Test mode')
    parser.add_argument('--query', type=str, 
                       default="What are the main HR policies and employee bylaws mentioned?",
                       help='Query to test (for single mode)')
    parser.add_argument('--full', action='store_true', default=True,
                       help='Show full answers (default: True)')
    parser.add_argument('--preview', action='store_true', default=False,
                       help='Show only answer previews')
    
    args = parser.parse_args()
    
    print("INFO: Using Ollama for LLM and embeddings.")
    
    try:
        if args.mode == 'basic':
            success = test_basic_api_endpoints()
        elif args.mode == 'single':
            success = test_single_query_via_api(args.query, show_full_answer=not args.preview)
        else:  # multiple
            success = test_enhanced_rag_via_api(show_full_answers=not args.preview)
        
        if success:
            print("\nSUCCESS: RAG API testing completed successfully!")
        else:
            print("\nERROR: RAG API testing encountered errors!")
            
        return success
        
    except Exception as e:
        print(f"ERROR: Enhanced RAG API test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Wait a moment for server to be ready
    time.sleep(1)
    success = main()
    exit(0 if success else 1)
