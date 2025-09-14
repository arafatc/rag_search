#!/usr/bin/env python3
"""Simple Phoenix prompt initialization"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def main():
    print(" Initializing Phoenix Prompts...")
    
    try:
        from rag_search.phoenix_prompts import initialize_phoenix_prompts, get_prompt
        
        # Test prompt loading first
        print(" Testing prompt loading from JSON...")
        test_prompt = get_prompt("research_task")
        if test_prompt:
            print(f"SUCCESS: Successfully loaded prompts from JSON (sample: {test_prompt[:50]}...)")
        else:
            print("ERROR: Failed to load prompts from JSON")
            return
        
        # Try Phoenix initialization
        print(" Attempting to initialize prompts in Phoenix...")
        success = initialize_phoenix_prompts()
        if success:
            print("SUCCESS: All prompts initialized in Phoenix!")
            print(" Open http://localhost:6006 to view prompts in Phoenix UI")
        else:
            print("WARNING:  Phoenix not available - application will use JSON/environment fallback")
            print("SUCCESS: Application will work normally with JSON prompts")
            
    except Exception as e:
        print(f"ERROR: Error: {e}")
        print("WARNING:  Application will fall back to environment variables if available")

if __name__ == "__main__":
    main()
