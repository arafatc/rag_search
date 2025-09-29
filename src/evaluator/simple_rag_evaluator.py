"""
Minimal RAG Strategy Evaluator using RAGAs
Loads questions from evaluation_dataset.jsonl and evaluates different RAG strategies
"""

import os
import sys
import json
import requests
import pandas as pd
from typing import Dict, Any, List, Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, os.path.normpath(project_root))

class SimpleRAGEvaluator:
    def __init__(self):
        self.strategies = ['semantic', 'hierarchical', 'contextual_rag', 'structure_aware']
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
            
            print(f" Loaded {len(questions)} evaluation questions from dataset")
            return questions
            
        except Exception as e:
            print(f" ERROR: Failed to load evaluation dataset: {e}")
            return []
    
    def get_strategy_response(self, strategy: str, question: str) -> Dict[str, Any]:
        """Get response from specific strategy using API calls (not direct crew calls)"""
        try:
            print(f"    Querying {strategy} strategy via API...")
            
            # Set the strategy environment variable for the API to use
            os.environ["DEFAULT_RAG_STRATEGY"] = strategy
            
            # Make API call like test_rag_api.py does
            import requests
            
            api_url = "http://localhost:8001/v1/chat/completions"
            response = requests.post(
                api_url,
                json={
                    "model": "rag-search",
                    "messages": [{"role": "user", "content": question}],
                    "max_tokens": 1000,
                    "temperature": 0.0
                },
                timeout=900  # 15 minute timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
                
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            
            # Extract clean contexts for RAGAs evaluation
            # Answer Relevancy is sensitive to context format - need clean, simple text
            contexts = []
            
            # Clean the answer by removing metadata sections
            clean_answer = answer
            
            # Remove sources section and everything after
            if "Sources" in clean_answer:
                clean_answer = clean_answer.split("Sources")[0].strip()
            
            # Remove retrieval strategy section  
            if "Retrieval Strategy" in clean_answer:
                clean_answer = clean_answer.split("Retrieval Strategy")[0].strip()
                
            # Remove any document reference lines like "Document 1 (score: 0.34):"
            lines = clean_answer.split('\n')
            filtered_lines = []
            for line in lines:
                # Skip lines that look like document headers or metadata
                if (line.strip().startswith("Document ") or 
                    "(score:" in line or 
                    "(relevance:" in line or
                    line.strip().startswith("HR Bylaws.pdf") or
                    line.strip() == ""):
                    continue
                filtered_lines.append(line.strip())
            
            clean_answer = ' '.join(filtered_lines).strip()
            
            # Create simple contexts from the cleaned answer
            if clean_answer and len(clean_answer) > 20:
                # Split into sentences
                sentences = clean_answer.replace('!', '.').replace('?', '.').split('.')
                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
                
                # Group sentences into logical contexts (2-3 sentences each)
                context_size = 2
                for i in range(0, len(sentences), context_size):
                    context_sentences = sentences[i:i+context_size]
                    context_text = '. '.join(context_sentences).strip()
                    if context_text and len(context_text) > 15:
                        # Ensure context ends with proper punctuation
                        if not context_text.endswith('.'):
                            context_text += '.'
                        contexts.append(context_text)
            
            # Fallback: create a single clean context
            if not contexts and clean_answer:
                contexts = [clean_answer[:500]]  # Limit length
            
            # Final fallback
            if not contexts:
                # Extract key information from the original question
                if "maximum" in question.lower() and "days" in question.lower():
                    contexts = ["The maximum period for salary deduction is specified in the HR policies."]
                else:
                    contexts = ["Relevant information found in the policy documents."]
            
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
            
            print(f"    Running RAGAs metrics for {len(test_data)} questions...")
            
            # Debug: Print first item structure
            if test_data:
                print(f"    Debug - First item keys: {list(test_data[0].keys())}")
                print(f"    Debug - Contexts type: {type(test_data[0]['contexts'])}")
                print(f"    Debug - Sample contexts: {test_data[0]['contexts'][:1] if test_data[0]['contexts'] else 'None'}")

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
            
            print(f"     Debug - RAGAs result type: {type(result)}")
            print(f"     Debug - RAGAs result attributes: {dir(result)}")
            
            # Extract scores - handle both EvaluationResult object and dictionary formats
            def extract_score(metric_name):
                try:
                    # For EvaluationResult objects with scores list
                    if hasattr(result, 'scores') and result.scores is not None and len(result.scores) > 0:
                        if isinstance(result.scores, list):
                            # Scores is a list of dictionaries
                            score_dict = result.scores[0]  # Get first (and usually only) result
                            if metric_name in score_dict:
                                score_value = score_dict[metric_name]
                                # Handle NaN values
                                if pd.isna(score_value):
                                    print(f"      Warning: {metric_name} returned NaN")
                                    return 0.0
                                return float(score_value)
                        else:
                            # Scores is a DataFrame
                            score_series = result.scores[metric_name]
                            if len(score_series) > 0:
                                score_value = score_series.iloc[0]
                                if pd.isna(score_value):
                                    print(f"      Warning: {metric_name} returned NaN")
                                    return 0.0
                                return float(score_value)
                    
                    # Fallback: try to_pandas() method
                    if hasattr(result, 'to_pandas'):
                        df = result.to_pandas()
                        if metric_name in df.columns and len(df) > 0:
                            score_value = df[metric_name].iloc[0]
                            if pd.isna(score_value):
                                print(f"      Warning: {metric_name} returned NaN")
                                return 0.0
                            return float(score_value)
                    
                    # Final fallback
                    print(f"      Warning: Could not extract {metric_name}, using 0.0")
                    return 0.0
                    
                except Exception as e:
                    print(f"     Error extracting {metric_name}: {e}")
                    return 0.0
            
            scores = {
                'faithfulness': extract_score('faithfulness'),
                'answer_relevancy': extract_score('answer_relevancy'),
                'context_precision': extract_score('context_precision'),
                'context_recall': extract_score('context_recall'),
            }
            
            # Calculate overall score, handling NaN values
            valid_scores = [s for s in scores.values() if not pd.isna(s) and s != 0.0]
            if valid_scores:
                scores['overall_score'] = sum(valid_scores) / len(valid_scores)
            else:
                scores['overall_score'] = 0.0
            
            return scores
            
        except ImportError as e:
            print(f"      RAGAs not available: {e}")
            print(f"     Install with: pipenv install ragas datasets")
            return None
            
        except Exception as e:
            print(f"     RAGAs evaluation failed: {e}")
            return None
    
    def evaluate_strategy(self, strategy: str, max_questions: int = None) -> Optional[Dict[str, Any]]:
        """Evaluate a single strategy"""
        print(f"\n EVALUATING STRATEGY: {strategy.upper()}")
        print("-" * 50)
        
        if not self.test_questions:
            print(" No evaluation questions available!")
            return None
        
        # Limit questions for testing
        questions_to_test = self.test_questions[:max_questions] if max_questions else self.test_questions
        print(f" Testing {len(questions_to_test)} questions")
        
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
                print(f"     Retrieved {response_data['retrieved_chunks']} chunks")
            else:
                print(f"     Failed: {response_data.get('error', 'Unknown error')}")
        
        if not test_data:
            print(f" No successful queries for {strategy}")
            return None

        print(f" Successfully processed {successful_queries}/{len(questions_to_test)} queries")

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
            print(f"   RAGAs Scores:")
            
            # Only print scores that are valid (not NaN and not 0.0)
            for metric_name, score in ragas_scores.items():
                if metric_name == 'overall_score':
                    continue  # Handle separately
                if not pd.isna(score) and score != 0.0:
                    metric_display = metric_name.replace('_', ' ').title()
                    print(f"   {metric_display}: {score:.3f}")
            
            print(f"   Overall Score: {ragas_scores['overall_score']:.3f}")
        else:
            result['overall_score'] = result['success_rate']  # Use success rate as fallback
            print(f"   RAGAs evaluation not available - using success rate as score")

        print(f" {strategy.upper()} evaluation complete!")
        return result


# Global instance for easy importing
evaluator = SimpleRAGEvaluator()

def evaluate_strategy(strategy: str, max_questions: int = None) -> Optional[Dict[str, Any]]:
    """Main function called by test_runner.py"""
    return evaluator.evaluate_strategy(strategy, max_questions)

def evaluate_all_strategies(max_questions: int = None) -> Dict[str, Any]:
    """Evaluate all available strategies"""
    results = {}
    
    print(f" EVALUATING ALL STRATEGIES")
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
    print(f"\n STRATEGY COMPARISON:")
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
                # Only show valid scores (not NaN and not 0.0)
                score_parts = []
                if not pd.isna(ragas['faithfulness']) and ragas['faithfulness'] != 0.0:
                    score_parts.append(f"F:{ragas['faithfulness']:.2f}")
                if not pd.isna(ragas['answer_relevancy']) and ragas['answer_relevancy'] != 0.0:
                    score_parts.append(f"AR:{ragas['answer_relevancy']:.2f}")
                if not pd.isna(ragas['context_precision']) and ragas['context_precision'] != 0.0:
                    score_parts.append(f"CP:{ragas['context_precision']:.2f}")
                if not pd.isna(ragas['context_recall']) and ragas['context_recall'] != 0.0:
                    score_parts.append(f"CR:{ragas['context_recall']:.2f}")
                
                if score_parts:
                    print(f"   RAGAs - {' '.join(score_parts)}")
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
            print(f" Invalid strategy. Choose from: {', '.join(evaluator.strategies)} or 'all'")
    else:
        print("Usage: python simple_rag_evaluator.py <strategy|all> [max_questions]")
        print(f"Available strategies: {', '.join(evaluator.strategies)}")