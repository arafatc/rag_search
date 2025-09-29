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
TIMEOUT = 960  # 16 minutes to accommodate 15-minute LLM timeout + buffer

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
            print(f"Full answer:\n{'-'*60}")
            print(answer)
            print(f"{'-'*60}")
            
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
    
    # Test queries with expected sources - updated comprehensive test set
    test_cases = [
        # Abu Dhabi Procurement Standards
        {
            "query": "How do the \"Delivery Terms\" and \"Payment Terms\" relate to a \"Purchase Order\" within the procurement process described in this document?",
            "expected_source": "Abu Dhabi Procurement Standards"
        },
        {
            "query": "Describe the relationship between \"Intellectual Property created during the tenure and framework of the contract\" (Foreground Intellectual Property) and the \"relevant Intellectual Property supplied by the contracting parties at the beginning of the engagement\" (Background Intellectual Property), as defined in the standards.",
            "expected_source": "Abu Dhabi Procurement Standards"
        },
        {
            "query": "Based on the introduction and scope, what is the primary purpose of the Abu Dhabi Procurement Standards, and how does it aim to achieve it?",
            "expected_source": "Abu Dhabi Procurement Standards"
        },
        {
            "query": "Explain the significance of a \"Valid Exception Code\" in the context of issuing \"post factum Purchase Orders\" as outlined in the document.",
            "expected_source": "Abu Dhabi Procurement Standards"
        },
        
        # HR Bylaws
        {
            "query": "Outline the disciplinary actions and their corresponding financial rules violations as presented in the penalty clauses.",
            "expected_source": "HR Bylaws"
        },
        {
            "query": "Based on the penalties section, what are the different levels of disciplinary actions for various infringements, and how do they relate to preserving pension or bonus rights?",
            "expected_source": "HR Bylaws"
        },
        {
            "query": "Summarize the overall objective of Decision No. (10) of 2020 as it relates to human resources in the Emirate of Abu Dhabi.",
            "expected_source": "HR Bylaws"
        },
        {
            "query": "Infer the hierarchical relationship between Law No. (6) of 2016 and Decision No. (10) of 2020, based on their descriptions in the document.",
            "expected_source": "HR Bylaws"
        },
        
        # Procurement Manual (Ariba Aligned)
        {
            "query": "Explain the roles of \"Team Grader\" and \"User Enablement Team\" within the context of \"SAP Ariba\" and their interdependencies.",
            "expected_source": "Procurement Manual (Ariba Aligned)"
        },
        {
            "query": "How does the concept of \"Supplier Manager System Groups\" relate to \"Supplier\" registration and qualification within the SAP Ariba framework?",
            "expected_source": "Procurement Manual (Ariba Aligned)"
        },
        {
            "query": "Based on the introduction and scope, what is the primary objective of aligning the Procurement Manual with SAP Ariba?",
            "expected_source": "Procurement Manual (Ariba Aligned)"
        },
        {
            "query": "Discuss how the manual's definitions of \"Technical Envelope\" and \"Technical and Commercial Envelope\" facilitate different aspects of the sourcing event process.",
            "expected_source": "Procurement Manual (Ariba Aligned)"
        },
        
        # Procurement Manual (Business Process)
        {
            "query": "Differentiate between \"Request for Proposal (RFP)\" and \"Request for Quotation (RFQ)\" based on their stated purposes and the type of information sought from suppliers.",
            "expected_source": "Procurement Manual (Business Process)"
        },
        {
            "query": "Illustrate the relationship between a \"Sole/Single Source\" tendering method and the concept of \"Supplier\" qualification within the procurement process.",
            "expected_source": "Procurement Manual (Business Process)"
        },
        {
            "query": "Based on the overall content, what is the main purpose of this Procurement Manual in relation to business processes?",
            "expected_source": "Procurement Manual (Business Process)"
        },
        {
            "query": "Explain how the definition of \"Reverse Auction (eAuction)\" reflects its role in achieving competitive bids for specific types of goods, services, and projects.",
            "expected_source": "Procurement Manual (Business Process)"
        },
        
        # Information Security
        {
            "query": "Explain the relationship between the risk-based approach and the \"Always Applicable\" controls as described in the UAE IA Standards. How should an implementing entity apply both concepts when determining their security controls?",
            "expected_source": "Information Security"
        },
        {
            "query": "What are the specific sub-controls listed under \"T5.2.3 USER SECURITY CREDENTIALS MANAGEMENT\"?",
            "expected_source": "Information Security"
        },
        {
            "query": "According to Annex C, what is the corresponding NIST SP 800-53 control for the UAE IA Standard \"M4.2.1 Screening\"?",
            "expected_source": "Information Security"
        },
        {
            "query": "What is the definition of \"Critical Entity\" as provided in Annex F of the UAE IA Standards?",
            "expected_source": "Information Security"
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