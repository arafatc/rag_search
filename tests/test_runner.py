#!/usr/bin/env python3
"""
RAGAs Evaluation Test Runner
Focused evaluation testing with Phoenix observability
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Any

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.insert(0, os.path.normpath(project_root))

# Phoenix integration
try:
    import phoenix as px
    from phoenix.otel import register
    from opentelemetry import trace
    
    # Initialize Phoenix
    register(project_name="rag_system", endpoint="http://localhost:6006/v1/traces")
    tracer = trace.get_tracer(__name__)
    PHOENIX_AVAILABLE = True
    print("SUCCESS: Phoenix tracing initialized for RAGAs evaluation session")
    
    def log_interaction(query: str, response: str, metadata: Dict[str, Any] = None):
        """Log interaction to Phoenix with OpenTelemetry span"""
        with tracer.start_as_current_span("test_runner_interaction") as span:
            span.set_attribute("query.text", str(query))
            span.set_attribute("response.length", len(str(response)))
            span.set_attribute("interaction.source", "test_runner")
            
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"metadata.{key}", str(value))
                    
except ImportError as e:
    PHOENIX_AVAILABLE = False
    print(f"WARNING:  Phoenix integration not available: {e}")
    
    def log_interaction(query: str, response: str, metadata: Dict[str, Any] = None):
        """Fallback logging when Phoenix not available"""
        pass



def run_evaluation_tests(strategy: str = "semantic"):
    """Run RAGAs evaluation tests"""
    print(f"\nSTARTING: STARTING: Evaluation Tests - {strategy.upper()} (Phoenix: {'SUCCESS:' if PHOENIX_AVAILABLE else 'ERROR:'})")
    print("=" * 80)
    
    try:
        from src.evaluator.simple_rag_evaluator import evaluate_strategy
        
        # Log evaluation start
        if PHOENIX_AVAILABLE:
            log_interaction(
                query=f"TEST_SESSION_START_EVALUATION",
                response=f"Starting RAGAs evaluation for {strategy} strategy",
                metadata={
                    "test_type": "evaluation",
                    "strategy": strategy,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Run evaluation
        result = evaluate_strategy(strategy)
        success = result is not None
        
        print(f"\nINFO: EVALUATION RESULTS for {strategy.upper()}:")
        print("=" * 60)
        if result:
            print(result)
        else:
            print("ERROR: Evaluation failed or returned no results")
        print("=" * 60)
        
        # Log evaluation completion  
        if PHOENIX_AVAILABLE:
            log_interaction(
                query=f"TEST_SESSION_END_EVALUATION",
                response=f"Evaluation for {strategy} {'completed' if success else 'failed'}",
                metadata={
                    "test_type": "evaluation",
                    "strategy": strategy,
                    "success": success,
                    "result_available": result is not None,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        return success
        
    except Exception as e:
        print(f"ERROR: ERROR: Evaluation tests failed: {e}")
        if PHOENIX_AVAILABLE:
            log_interaction(
                query="TEST_ERROR_EVALUATION",
                response=f"Evaluation test execution failed: {str(e)}",
                metadata={"test_type": "evaluation", "error": True}
            )
        return False

def main():
    """Main RAGAs evaluation test runner with Phoenix observability"""
    parser = argparse.ArgumentParser(description='RAGAs Evaluation Test Runner with Phoenix Integration')
    parser.add_argument('--strategy', choices=['semantic', 'hierarchical', 'both'], 
                       default='semantic', help='RAG strategy to evaluate')
    
    args = parser.parse_args()
    
    # Start test session
    session_start = time.time()
    print(f" PHOENIX RAGAS EVALUATION RUNNER")
    print(f"Phoenix Available: {'SUCCESS: YES' if PHOENIX_AVAILABLE else 'ERROR: NO'}")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    results = {}
    
    try:
        # Run evaluation tests
        if args.strategy == 'both':
            results['evaluation_semantic'] = run_evaluation_tests('semantic')
            results['evaluation_hierarchical'] = run_evaluation_tests('hierarchical')
        else:
            results[f'evaluation_{args.strategy}'] = run_evaluation_tests(args.strategy)
        
        # Summary
        session_time = time.time() - session_start
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        
        print(f"\n EVALUATION SESSION SUMMARY")
        print("=" * 80)
        print(f"Total Evaluations: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Session Time: {session_time:.2f}s")
        print(f"Phoenix Tracing: {'SUCCESS: Active' if PHOENIX_AVAILABLE else 'ERROR: Inactive'}")
        
        # Log final session summary
        if PHOENIX_AVAILABLE:
            log_interaction(
                query="EVALUATION_SESSION_SUMMARY",
                response=f"Evaluation session completed: {passed_tests}/{total_tests} passed",
                metadata={
                    "session_summary": True,
                    "total_evaluations": total_tests,
                    "passed_evaluations": passed_tests,
                    "session_time": session_time,
                    "evaluation_results": results,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Exit with appropriate code
        exit_code = 0 if passed_tests == total_tests else 1
        print(f"\n{'SUCCESS: ALL EVALUATIONS PASSED!' if exit_code == 0 else 'ERROR: SOME EVALUATIONS FAILED!'}")
        
        return exit_code == 0
        
    except Exception as e:
        print(f"ERROR: FATAL ERROR: Evaluation runner failed: {e}")
        if PHOENIX_AVAILABLE:
            log_interaction(
                query="EVALUATION_RUNNER_ERROR",
                response=f"Evaluation runner failed: {str(e)}",
                metadata={"fatal_error": True}
            )
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
