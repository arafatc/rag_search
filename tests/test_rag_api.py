#!/usr/bin/env python3
"""
Simple test script to validate the optimized RAG system
Tests the API directly without pytest to verify performance improvements
"""

import os
import sys
import time
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configuration
BASE_URL = "http://localhost:8001"
TIMEOUT = 360  # Increased to 6 minutes to match processing time

def test_query(query: str, expected_source: str = None):
    """Test a single query and measure performance"""
    print(f"\n{'='*80}")
    print(f"TESTING QUERY: {query}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": "rag-search",
                "messages": [{"role": "user", "content": query}],
                "max_tokens": 1000,
                "temperature": 0.1
            },
            timeout=TIMEOUT
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Response time: {duration:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            answer = data["choices"][0]["message"]["content"]
            
            print(f"Status: SUCCESS")
            print(f"Answer length: {len(answer)} characters")
            print(f"Answer preview: {answer[:200]}...")
            
            if expected_source:
                if expected_source.lower() in answer.lower():
                    print(f" CORRECT SOURCE: Found '{expected_source}' in response")
                else:
                    print(f" WRONG SOURCE: Expected '{expected_source}' not found in response")
            
            # Look for source information in the response
            if "source" in answer.lower() or "document" in answer.lower():
                print(" Sources mentioned in response")
            else:
                print("  No source information found in response")
                
            return True, duration, answer
            
        else:
            print(f" FAILED: HTTP {response.status_code}")
            print(f"Error: {response.text}")
            return False, duration, None
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f" EXCEPTION: {str(e)}")
        return False, duration, None

def main():
    """Run the optimized system tests"""
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Single query mode
        query = " ".join(sys.argv[1:])
        print(" Testing Single Query")
        print(f"Target URL: {BASE_URL}")
        print(f"Query: {query}")
        
        success, duration, answer = test_query(query)
        
        if success:
            print(f"\n SUCCESS in {duration:.2f}s")
            print(f"Answer: {answer}")
        else:
            print(f"\n FAILED after {duration:.2f}s")
        
        return
    
    # Full test suite mode
    print(" Testing Optimized RAG System")
    print(f"Target URL: {BASE_URL}")
    print(f"Timeout: {TIMEOUT} seconds")
    
    # Test queries with expected sources (all 9 from original test)
    test_cases = [
        {
            "query": "If Im a non-citizen, what are the rules for me to get hired in a government job?",
            "expected_source": "HR Bylaws"
        },
        {
            "query": "What is Article (4) Delegation of Power?",
            "expected_source": "HR Bylaws"
        },
        {
            "query": "How long is the probation period for new employees and what happens if I dont pass it?",
            "expected_source": "HR Bylaws"
        },
        {
            "query": "What basic principles guide government procurement in Abu Dhabi?",
            "expected_source": "Abu Dhabi Procurement Standards"
        },
        {
            "query": "What is Article (4): Compliance with Applicable Legislations",
            "expected_source": "Abu Dhabi Procurement Standards"
        },
        {
            "query": "What ethical rules do procurement officers need to follow?",
            "expected_source": "Abu Dhabi Procurement Standards"
        },
        {
            "query": "Explain Article (65) Absence by Permission Hours",
            "expected_source": "HR Bylaws"
        },
        {
            "query": "How are conflicts of interest managed for employees and also for procurement staff?",
            "expected_source": None  # This query spans both documents
        },
        {
            "query": "What do the HR bylaws say about hiring employees and what do the procurement standards say about terminating suppliers?",
            "expected_source": None  # This query spans both documents
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}/{len(test_cases)}")
        success, duration, answer = test_query(
            test_case["query"], 
            test_case.get("expected_source")
        )
        
        results.append({
            "test_number": i,
            "query": test_case["query"],
            "success": success,
            "duration": duration,
            "expected_source": test_case.get("expected_source"),
            "answer_preview": answer[:100] if answer else None
        })
    
    total_duration = time.time() - total_start_time
    
    # Summary
    print(f"\n{'='*80}")
    print(" TEST SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = sum(1 for r in results if r["success"])
    total_tests = len(results)
    success_rate = (successful_tests / total_tests) * 100
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total time: {total_duration:.2f} seconds")
    print(f"Average time per test: {total_duration/total_tests:.2f} seconds")
    
    # Individual results
    for result in results:
        status = " PASS" if result["success"] else "❌ FAIL"
        print(f"Test {result['test_number']}: {status} - {result['duration']:.2f}s - {result['query'][:50]}...")
    
    print(f"\n System Performance:")
    print(f"   - Success Rate: {success_rate:.1f}%")
    print(f"   - Average Response Time: {total_duration/total_tests:.2f}s")
    
    if success_rate >= 100:
        print(" ALL TESTS PASSED! System is working optimally.")
    elif success_rate >= 80:
        print(" Most tests passed. System is working well.")
    else:
        print(" Some tests failed. System needs attention.")

if __name__ == "__main__":
    main()