"""
Minimal RAG Strategy Evaluator using RAGAs
Loads questions from evaluation_dataset.jsonl and evaluates different RAG strategies
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, os.path.normpath(project_root))

class SimpleRAGEvaluator:
    def __init__(self):
        self.strategies = ['semantic', 'hierarchical', 'contextual_rag']
        self.test_questions = self._load_evaluation_dataset()
    
    def _load_evaluation_dataset(self) -> List[Dict[str, Any]]:
        """Load test questions from evaluation_dataset.jsonl"""
        dataset_path = os.path.join(os.path.dirname(__file__), 'evaluation_dataset.jsonl')
        
        if not os.path.exists(dataset_path):
            print(f"WARNING: Evaluation dataset not found at {dataset_path}")
            return []
        
        questions = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        questions.append({
                            "question": data.get("question", ""),
                            "ground_truth": data.get("ground_truth", ""),
                            "category": data.get("category", "unknown"),
                            "difficulty": data.get("difficulty", "medium")
                        })
            
            print(f"📊 Loaded {len(questions)} evaluation questions from dataset")
            return questions
            
        except Exception as e:
            print(f"❌ ERROR: Failed to load evaluation dataset: {e}")
            return []
    
    def get_strategy_response(self, strategy: str, question: str) -> Dict[str, Any]:
        """Get response from specific strategy using direct retrieval"""
        try:
            print(f"    Querying {strategy} strategy directly...")
            
            # Use strategy-specific retrieval to get real contexts
            from src.rag_search.tools import retrieve_with_strategy, reset_tool_call_counter
            from src.rag_search.crew import create_rag_crew
            
            # Reset tool counter and use the strategy-specific retrieval function
            reset_tool_call_counter()
            
            # Get retrieval results using strategy-specific logic
            retrieval_result = retrieve_with_strategy(question, strategy)
            
            # Parse the formatted result to extract contexts
            contexts = []
            if "--- Chunk" in retrieval_result:
                chunks = retrieval_result.split("--- Chunk")
                for chunk in chunks[1:]:  # Skip first empty part
                    if "Content:" in chunk:
                        content_start = chunk.find("Content:") + 8
                        content_end = chunk.find("---", content_start)
                        if content_end == -1:
                            content = chunk[content_start:].strip()
                        else:
                            content = chunk[content_start:content_end].strip()
                        if content:
                            contexts.append(content)
            
            # Fallback: if no contexts parsed, use the whole result as one context
            if not contexts and retrieval_result:
                contexts = [retrieval_result[:1000]]  # Limit to 1000 chars
            
            # Get answer using crew
            crew_output = create_rag_crew(question, "")
            answer = str(crew_output.raw) if hasattr(crew_output, 'raw') else str(crew_output)
            
            # Clean up answer formatting
            if answer.startswith('**'):
                answer = answer[2:].strip()
            
            return {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "retrieved_chunks": len(contexts),
                "success": True
            }
                
        except Exception as e:
            print(f"    ERROR: Failed to get {strategy} response: {e}")
            import traceback
            traceback.print_exc()
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "contexts": [],
                "retrieved_chunks": 0,
                "success": False,
                "error": str(e)
            }
    
    def run_ragas_evaluation(self, strategy: str, test_data: List[Dict]) -> Optional[Dict[str, float]]:
        """Run RAGAs evaluation on test data"""
        try:
            import pandas as pd
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            
            print(f"    🔬 Running RAGAs metrics for {len(test_data)} questions...")
            
            # Debug: Print first item structure
            if test_data:
                print(f"    🔍 Debug - First item keys: {list(test_data[0].keys())}")
                print(f"    🔍 Debug - Contexts type: {type(test_data[0]['contexts'])}")
                print(f"    🔍 Debug - Sample contexts: {test_data[0]['contexts'][:1] if test_data[0]['contexts'] else 'None'}")
            
            # Ensure contexts are properly formatted as lists of strings
            formatted_data = []
            for item in test_data:
                formatted_item = {
                    'question': str(item['question']),
                    'answer': str(item['answer']),
                    'contexts': item['contexts'] if isinstance(item['contexts'], list) else [str(item['contexts'])],
                    'ground_truth': str(item['ground_truth'])
                }
                formatted_data.append(formatted_item)
            
            # Convert to Dataset
            df = pd.DataFrame(formatted_data)
            dataset = Dataset.from_pandas(df)
            
            # Run evaluation
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
            
            print(f"    🔍 Debug - RAGAs result type: {type(result)}")
            print(f"    🔍 Debug - RAGAs result keys: {list(result.keys()) if hasattr(result, 'keys') else 'No keys'}")
            
            # Extract scores - handle both scalar and list results
            def extract_score(value):
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    return float(value[0])  # Take first value if it's a list
                elif isinstance(value, (int, float)):
                    return float(value)
                else:
                    return 0.0  # Fallback
            
            scores = {
                'faithfulness': extract_score(result['faithfulness']),
                'answer_relevancy': extract_score(result['answer_relevancy']),
                'context_precision': extract_score(result['context_precision']),
                'context_recall': extract_score(result['context_recall']),
            }
            scores['overall_score'] = sum(scores.values()) / len(scores)
            
            return scores
            
        except ImportError as e:
            print(f"    ⚠️  RAGAs not available: {e}")
            print(f"    📦 Install with: pipenv install ragas datasets")
            return None
            
        except Exception as e:
            print(f"    ❌ RAGAs evaluation failed: {e}")
            return None
    
    def evaluate_strategy(self, strategy: str, max_questions: int = None) -> Optional[Dict[str, Any]]:
        """Evaluate a single strategy"""
        print(f"\n🔬 EVALUATING STRATEGY: {strategy.upper()}")
        print("-" * 50)
        
        if not self.test_questions:
            print("❌ No evaluation questions available!")
            return None
        
        # Limit questions for testing
        questions_to_test = self.test_questions[:max_questions] if max_questions else self.test_questions
        print(f"📝 Testing {len(questions_to_test)} questions")
        
        # Get responses for all test questions
        test_data = []
        successful_queries = 0
        
        for i, test_item in enumerate(questions_to_test, 1):
            print(f"  Question {i}/{len(questions_to_test)}: {test_item['question'][:60]}...")
            
            response_data = self.get_strategy_response(strategy, test_item['question'])
            
            if response_data['success']:
                test_data.append({
                    "question": test_item['question'],
                    "answer": response_data['answer'],
                    "contexts": response_data['contexts'],
                    "ground_truth": test_item['ground_truth']
                })
                successful_queries += 1
                print(f"    ✅ Retrieved {response_data['retrieved_chunks']} chunks")
            else:
                print(f"    ❌ Failed: {response_data.get('error', 'Unknown error')}")
        
        if not test_data:
            print(f"❌ No successful queries for {strategy}")
            return None
        
        print(f"📊 Successfully processed {successful_queries}/{len(questions_to_test)} queries")
        
        # Run RAGAs evaluation
        ragas_scores = self.run_ragas_evaluation(strategy, test_data)
        
        # Create result summary
        result = {
            'strategy': strategy,
            'total_questions': len(questions_to_test),
            'successful_queries': successful_queries,
            'success_rate': successful_queries / len(questions_to_test),
            'average_chunks': sum(len(item['contexts']) for item in test_data) / len(test_data),
            'average_response_length': sum(len(item['answer']) for item in test_data) / len(test_data),
        }
        
        if ragas_scores:
            result['ragas_scores'] = ragas_scores
            result['overall_score'] = ragas_scores['overall_score']
            print(f"  📈 RAGAs Scores:")
            print(f"     Faithfulness: {ragas_scores['faithfulness']:.3f}")
            print(f"     Answer Relevancy: {ragas_scores['answer_relevancy']:.3f}")
            print(f"     Context Precision: {ragas_scores['context_precision']:.3f}")
            print(f"     Context Recall: {ragas_scores['context_recall']:.3f}")
            print(f"     Overall Score: {ragas_scores['overall_score']:.3f}")
        else:
            result['overall_score'] = result['success_rate']  # Use success rate as fallback
            print(f"  ⚠️  RAGAs evaluation not available - using success rate as score")
        
        print(f"✅ {strategy.upper()} evaluation complete!")
        return result


# Global instance for easy importing
evaluator = SimpleRAGEvaluator()

def evaluate_strategy(strategy: str, max_questions: int = None) -> Optional[Dict[str, Any]]:
    """Main function called by test_runner.py"""
    return evaluator.evaluate_strategy(strategy, max_questions)

def evaluate_all_strategies(max_questions: int = None) -> Dict[str, Any]:
    """Evaluate all available strategies"""
    results = {}
    
    print(f"🎯 EVALUATING ALL STRATEGIES")
    print("=" * 60)
    
    for strategy in evaluator.strategies:
        try:
            result = evaluate_strategy(strategy, max_questions)
            if result:
                results[strategy] = result
            else:
                results[strategy] = {'error': 'Evaluation failed'}
        except Exception as e:
            results[strategy] = {'error': str(e)}
    
    # Compare strategies
    print(f"\n🏆 STRATEGY COMPARISON:")
    print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        # Sort by overall score
        sorted_strategies = sorted(valid_results.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        
        for i, (strategy, result) in enumerate(sorted_strategies, 1):
            print(f"{i}. {strategy.upper()}: {result['overall_score']:.3f}")
            print(f"   Success Rate: {result['success_rate']:.1%}")
            print(f"   Avg Chunks: {result['average_chunks']:.1f}")
            if 'ragas_scores' in result:
                ragas = result['ragas_scores']
                print(f"   RAGAs - F:{ragas['faithfulness']:.2f} AR:{ragas['answer_relevancy']:.2f} CP:{ragas['context_precision']:.2f} CR:{ragas['context_recall']:.2f}")
            print()
    
    return results

if __name__ == "__main__":
    # Can be run directly
    import sys
    
    if len(sys.argv) > 1:
        strategy = sys.argv[1]
        max_q = int(sys.argv[2]) if len(sys.argv) > 2 else None
        
        if strategy == "all":
            evaluate_all_strategies(max_q)
        elif strategy in evaluator.strategies:
            evaluate_strategy(strategy, max_q)
        else:
            print(f"❌ Invalid strategy. Choose from: {', '.join(evaluator.strategies)} or 'all'")
    else:
        print("Usage: python simple_rag_evaluator.py <strategy|all> [max_questions]")
        print(f"Available strategies: {', '.join(evaluator.strategies)}")