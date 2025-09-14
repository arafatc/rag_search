"""
RAG Search module for document retrieval and question answering.

This module provides a minimalistic implementation of a Retrieval-Augmented Generation (RAG) system
using CrewAI agents, LlamaIndex for document retrieval, and Ollama for language models and embeddings.

Main components:
- agents.py: Document researcher and insight synthesizer agents
- tools.py: Document retrieval tool with hybrid search capabilities
- crew.py: Crew configuration for coordinating the RAG workflow
"""

# Lazy imports to avoid heavy dependencies when not needed
def get_rag_components():
    """Lazy loading of RAG components"""
    from .agents import document_researcher, insight_synthesizer
    from .tools import document_retrieval_tool
    from .crew import create_rag_crew
    
    return {
        'document_researcher': document_researcher,
        'insight_synthesizer': insight_synthesizer,
        'document_retrieval_tool': document_retrieval_tool,
        'create_rag_crew': create_rag_crew
    }

# Export configuration without heavy imports
# Configuration moved to rag_api_server.py (consolidated)

__all__ = [
    'get_rag_components'
]
