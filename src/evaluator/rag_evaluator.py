#!/usr/bin/env python3
"""
Simple RAG Evaluator using RAGAs
Lean evaluation system focused on core functionality with Phoenix observability
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from dotenv import load_dotenv

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
tests_dir = os.path.join(project_root, 'tests')

sys.path.insert(0, os.path.normpath(project_root))
sys.path.insert(0, os.path.normpath(tests_dir))

# Phoenix integration for evaluation observability
try:
    from phoenix.otel import register
    from opentelemetry import trace
    
    # Initialize tracer for evaluation
    tracer = trace.get_tracer(__name__)
    
    def log_interaction(query, response, metadata=None):
        """Create Phoenix spans for evaluation interactions"""
        with tracer.start_as_current_span("evaluation_interaction") as span:
            span.set_attribute("query.text", str(query)[:200])
            span.set_attribute("response.length", len(str(response)) if response else 0)
            span.set_attribute("interaction.source", "evaluator")
            
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"metadata.{key}", str(value))
    
    PHOENIX_AVAILABLE = True
    print("INFO: Phoenix integration available for evaluation logging")
except ImportError:
    PHOENIX_AVAILABLE = False
    def log_interaction(query, response, metadata=None):
        """Fallback function when Phoenix is not available"""
        pass
    print("WARNING: Phoenix integration not available - continuing without observability")

# Load environment variables
load_dotenv()

# Phoenix integration - using the already imported function above
PHOENIX_LOGGING = PHOENIX_AVAILABLE

# Try to import the enhanced query function
try:
    # Import using modern importlib approach
    import importlib.util
    import sys
    import os
    tests_path = os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_enhanced_rag_query.py')
    spec = importlib.util.spec_from_file_location("test_enhanced_rag_query", tests_path)
    test_module = importlib.util.module_from_spec(spec)
    sys.modules["test_enhanced_rag_query"] = test_module
    spec.loader.exec_module(test_module)
    setup_enhanced_rag_query_engine = test_module.setup_enhanced_rag_query_engine
    ENHANCED_QUERY_AVAILABLE = True
    print("SUCCESS: Enhanced query engine loaded")
except Exception as e:
    print(f"WARNING: Enhanced query engine not available: {e}")
    ENHANCED_QUERY_AVAILABLE = False

def load_evaluation_dataset(dataset_path: str = None):
    """Load evaluation dataset from JSONL file"""
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), 'evaluation_dataset.jsonl')
    
    print(f"DOCUMENTATION: Loading evaluation dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Error: Evaluation dataset not found at {dataset_path}")
        return None
    
    try:
        dataset = Dataset.from_json(dataset_path)
        print(f"SUCCESS: Loaded {len(dataset)} questions")
        return dataset
    except Exception as e:
        print(f"ERROR: Error loading dataset: {e}")
        return None

def run_rag_pipeline(query: str, strategy: str = "semantic"):
    """Run the RAG pipeline and return answer with contexts"""
    try:
        if not ENHANCED_QUERY_AVAILABLE:
            return {"answer": "Enhanced query engine not available", "contexts": []}
        
        # Setup query engine for the specified strategy
        query_engine, index = setup_enhanced_rag_query_engine(strategy)
        
        if query_engine is None:
            return {"answer": f"Strategy '{strategy}' not available", "contexts": []}
        
        # Query the system
        response = query_engine.query(query)
        
        # Log interaction with Phoenix for observability
        if PHOENIX_AVAILABLE:
            log_interaction(
                query=query,
                response=str(response),
                metadata={
                    'evaluation_strategy': strategy,
                    'evaluation_mode': True,
                    'source_nodes_count': len(response.source_nodes) if hasattr(response, 'source_nodes') and response.source_nodes else 0
                }
            )
        
        # Extract answer and contexts
        answer = str(response)
        contexts = []
        
        if hasattr(response, 'source_nodes') and response.source_nodes:
            contexts = [node.text for node in response.source_nodes[:5]]
        
        # If no contexts, use answer as context for RAGAs evaluation
        if not contexts:
            contexts = [answer]
        
        # Log interaction for Phoenix observability
        if PHOENIX_LOGGING:
            log_interaction(
                query=query,
                response=answer,
                metadata={
                    "strategy": strategy,
                    "contexts_count": len(contexts),
                    "evaluation_mode": True
                }
            )
        
        return {"answer": answer, "contexts": contexts}
        
    except Exception as e:
        print(f"ERROR: Error running RAG pipeline for query '{query[:50]}...': {e}")
        return {"answer": f"Error: {str(e)}", "contexts": []}

def evaluate_strategy(strategy: str = "semantic", dataset_path: str = None):
    """Evaluate a specific RAG strategy"""
    print(f"\nCHECKING: Evaluating {strategy.upper()} strategy")
    print("-" * 50)
    
    # Load dataset
    dataset = load_evaluation_dataset(dataset_path)
    if dataset is None:
        return None
    
    # Run RAG pipeline for each question
    print(f"\nSTARTING: Running RAG pipeline on {len(dataset)} questions...")
    
    questions = []
    ground_truths = []
    answers = []
    contexts_list = []
    
    for i, entry in enumerate(dataset, 1):
        question = entry['question']
        ground_truth = entry.get('ground_truth', 'No ground truth available')
        
        print(f"  {i}/{len(dataset)}: {question[:80]}...")
        
        # Skip empty questions
        if not question or not question.strip():
            print("    WARNING:  Skipping empty question")
            continue
        
        # Run the pipeline
        result = run_rag_pipeline(question, strategy)
        
        questions.append(question)
        ground_truths.append(ground_truth)
        answers.append(result["answer"])
        contexts_list.append(result["contexts"])
        
        print(f"    SUCCESS: Answer length: {len(result['answer'])} chars, Contexts: {len(result['contexts'])}")
    
    if not questions:
        print("ERROR: No valid questions to evaluate")
        return None
    
    # Prepare dataset for RAGAs evaluation
    print(f"\nINFO: Evaluating {len(questions)} results with RAGAs...")
    
    evaluation_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }
    
    eval_dataset = Dataset.from_dict(evaluation_data)
    
    # Define metrics
    metrics = [
        faithfulness,       # How factually accurate is the answer?
        answer_relevancy,   # How relevant is the answer to the question?
        context_recall,     # Did we find all relevant context?
        context_precision,  # Was the context precise?
    ]
    
    # Run evaluation
    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
        )
        
        # Log evaluation results with Phoenix
        if PHOENIX_AVAILABLE and result is not None:
            # Extract metric scores from result
            metric_scores = {}
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                if not df.empty:
                    metric_scores = {col: df[col].mean() for col in df.columns if col in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']}
            
            log_interaction(
                query=f"EVALUATION_{strategy.upper()}",
                response=f"Evaluation completed with {len(eval_dataset)} questions",
                metadata={
                    'evaluation_type': 'ragas_metrics',
                    'strategy': strategy,
                    'dataset_size': len(eval_dataset),
                    'metrics': metric_scores,
                    'timestamp': datetime.now().isoformat()
                }
            )
        
        print(f"\n Evaluation Complete for {strategy.upper()}!")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"ERROR: RAGAs evaluation failed: {e}")
        return None

def compare_strategies(strategies: list = ["semantic", "hierarchical"], dataset_path: str = None):
    """Compare multiple strategies"""
    print(f"\n Comparing strategies: {', '.join(strategies)}")
    print("=" * 80)
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*20} EVALUATING {strategy.upper()} {'='*20}")
        result = evaluate_strategy(strategy, dataset_path)
        if result is not None:
            results[strategy] = result
    
    # Print comparison summary
    if len(results) > 1:
        print(f"\n COMPARISON SUMMARY")
        print("=" * 50)
        
        for strategy, result in results.items():
            print(f"\n{strategy.upper()}:")
            for metric, value in result.items():
                print(f"  {metric}: {value:.4f}")
    
    return results

async def main():
    """Main evaluation function"""
    print("STARTING: Simple RAG Evaluator")
    print("=" * 40)
    
    # Check if enhanced query is available
    if not ENHANCED_QUERY_AVAILABLE:
        print("ERROR: Enhanced query engine not available. Cannot run evaluation.")
        return
    
    # Default: evaluate semantic strategy only
    strategy = os.getenv('EVALUATION_STRATEGY', 'semantic')
    
    # Run evaluation
    result = evaluate_strategy(strategy)
    
    if result:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_results_{strategy}_{timestamp}.json"
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dict(result), f, indent=2, ensure_ascii=False)
            print(f"\nSAVING: Results saved to: {output_path}")
        except Exception as e:
            print(f"WARNING:  Could not save results: {e}")

if __name__ == "__main__":
    asyncio.run(main())
