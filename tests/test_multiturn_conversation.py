#!/usr/bin/env python3
"""
Patient Multi-Turn Test
Test with longer timeouts and progress monitoring
"""

import requests
import json
import time
import os
import glob

def patient_test():
    """Test with longer timeout and progress monitoring"""
    
    url = "http://localhost:8001/v1/chat/completions"
    
    print(" Patient Multi-Turn Conversation Test")
    print("=" * 50)
    print("TIME:  Using 5-minute timeout per request...")
    
    # Turn 1: HR Policies
    print("\nNOTE: Turn 1: HR Policies")
    messages = [{"role": "user", "content": "What are basic HR hiring policies?"}]
    
    response1 = patient_api_call(url, messages, turn=1)
    if not response1:
        print("ERROR: Turn 1 failed")
        return False
        
    assistant_msg1 = response1['choices'][0]['message']['content']
    messages.append({"role": "assistant", "content": assistant_msg1})
    print(f"SUCCESS: Turn 1 Response: {assistant_msg1[:150]}...")
    
    # Turn 2: Procurement Policies  
    print("\nNOTE: Turn 2: Procurement Policies (with HR context)")
    messages.append({"role": "user", "content": "What about basic procurement policies?"})
    
    response2 = patient_api_call(url, messages, turn=2)
    if not response2:
        print("ERROR: Turn 2 failed")
        return False
        
    assistant_msg2 = response2['choices'][0]['message']['content']
    messages.append({"role": "assistant", "content": assistant_msg2})
    print(f"SUCCESS: Turn 2 Response: {assistant_msg2[:150]}...")
    
    # Turn 3: Context Test - Ask about a specific topic that should reference previous context
    print("\nNOTE: Turn 3: Testing Context Awareness")
    messages.append({"role": "user", "content": "Are there any rules about suppliers in these policies I asked about?"})
    
    response3 = patient_api_call(url, messages, turn=3)
    if not response3:
        print("ERROR: Turn 3 failed")
        return False
        
    assistant_msg3 = response3['choices'][0]['message']['content']
    print(f"SUCCESS: Turn 3 Response: {assistant_msg3}")
    
    # Check for context awareness
    context_indicators = ['supplier', 'policies', 'these', 'hr', 'procurement', 'mentioned', 'discussed', 'asked about']
    response_lower = assistant_msg3.lower()
    found_indicators = [ind for ind in context_indicators if ind in response_lower]
    
    print(f"\nCHECKING: Context Analysis:")
    print(f"INFO: Context indicators found: {found_indicators}")
    print(f" Total messages in conversation: {len(messages)}")
    
    # Check if response shows awareness of previous conversation
    context_aware = (len(found_indicators) >= 2 and 
                    ('these' in response_lower or 'mentioned' in response_lower or 'discussed' in response_lower))
    
    if context_aware:
        print(" SUCCESS: Multi-turn conversation with context awareness!")
        return True
    else:
        print("WARNING: Limited context awareness detected")
        print("INFO: For context awareness, response should reference previous conversation")
        return False

def patient_api_call(url, messages, turn):
    """API call with progress monitoring"""
    
    payload = {"model": "rag-search", "messages": messages}
    
    print(f" Sending Turn {turn} request with {len(messages)} messages...")
    print("INFO: Monitoring progress...")
    
    try:
        start_time = time.time()
        
        # Make the request with 5 minute timeout
        response = requests.post(
            url, 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minutes
        )
        
        elapsed = time.time() - start_time
        print(f"TIME:  Request completed in {elapsed:.1f} seconds")
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"ERROR: HTTP Error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print(" Request timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"ERROR: Request failed: {e}")
        return None

def monitor_progress():
    """Monitor JSON files in output directory"""
    output_dir = "output"
    if os.path.exists(output_dir):
        json_files = glob.glob(os.path.join(output_dir, "*.json"))
        if json_files:
            latest_file = max(json_files, key=os.path.getctime)
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                print(f"FOLDER: Latest file: {os.path.basename(latest_file)}")
                print(f"INFO: Status: {data.get('status', 'unknown')}")
            except:
                pass

if __name__ == "__main__":
    print("STARTING: Starting Patient Multi-Turn Test")
    print("This test uses 5-minute timeouts per request")
    print()
    
    # Quick health check first
    try:
        health_response = requests.get("http://localhost:8001/health", timeout=10)
        if health_response.status_code == 200:
            print("SUCCESS: Server health check passed")
        else:
            print("ERROR: Server not responding properly")
            exit(1)
    except:
        print("ERROR: Cannot reach server")
        exit(1)
    
    # Run the patient test
    success = patient_test()
    
    print("\n" + "="*50)
    print(f" FINAL RESULT: {'PASS' if success else 'FAIL'}")
    if success:
        print(" Multi-turn conversation is working with context preservation!")
    else:
        print("ERROR: Multi-turn conversation needs improvement")
