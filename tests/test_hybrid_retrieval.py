#!/usr/bin/env python3
"""Test our new hybrid retrieval system with both LlamaIndex and custom retrievers"""

import os
from dotenv import load_dotenv
from llama_index.embeddings.ollama import OllamaEmbedding

# Load environment
load_dotenv()

# Import our tools
import sys
sys.path.append('src/rag_search')
from tools import LlamaIndexRetriever, CustomPGVectorRetriever

def main():
    print("=== TESTING HYBRID RETRIEVAL SYSTEM ===\n")
    
    # Setup embedding model
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text:latest",
        base_url="http://localhost:11434",
        request_timeout=120.0
    )
    
    database_url = os.getenv("DATABASE_URL")
    query = "HR policies employee conduct"
    
    print("1. Testing LlamaIndex Retriever (new ingestion)...")
    try:
        llamaindex_retriever = LlamaIndexRetriever(
            database_url=database_url,
            table_name="llamaindex_documents",
            embed_model=embed_model
        )
        
        results = llamaindex_retriever.retrieve(query, similarity_top_k=3)
        print(f"   SUCCESS: LlamaIndex retriever: {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"     Result {i+1}: Score {getattr(result, 'score', 'N/A'):.4f}")
            print(f"       Text: {result.text[:100]}...")
            print(f"       Source: {result.metadata.get('source', 'Unknown')}")
            if hasattr(result, 'relationships') and result.relationships:
                print(f"       Relationships: {list(result.relationships.keys())}")
    
    except Exception as e:
        print(f"   ERROR: LlamaIndex retriever failed: {e}")
    
    print("\n2. Testing Custom Retriever (fallback)...")
    try:
        custom_retriever = CustomPGVectorRetriever(
            database_url=database_url,
            table_name="data_document_embeddings",
            embed_model=embed_model
        )
        
        results = custom_retriever.retrieve(query, similarity_top_k=3)
        print(f"   SUCCESS: Custom retriever: {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"     Result {i+1}: Score {result.score:.4f}")
            print(f"       Text: {result.node.text[:100]}...")
            print(f"       Source: {result.node.metadata.get('source', 'Unknown')}")
    
    except Exception as e:
        print(f"   ERROR: Custom retriever failed: {e}")
    
    print("\n3. Testing document_retrieval_tool (hybrid approach)...")
    try:
        from tools import document_retrieval_tool
        result = document_retrieval_tool(query)
        
        if "Error" in result:
            print(f"   ERROR: Tool failed: {result}")
        else:
            print(f"   SUCCESS: Tool succeeded: {len(result)} characters")
            print(f"     Preview: {result[:200]}...")
    
    except Exception as e:
        print(f"   ERROR: Tool failed: {e}")

if __name__ == "__main__":
    main()
