"""
Simple RAG Evaluation Framework
Lean evaluation system for RAG implementations using RAGAs
"""

__version__ = "1.0.0"
__all__ = ["evaluate_strategy", "compare_strategies", "load_evaluation_dataset"]

# Import main functions from simple_rag_evaluator
try:
    from .simple_rag_evaluator import evaluate_strategy, compare_strategies, load_evaluation_dataset
except ImportError:
    # Fallback for direct script execution
    pass
