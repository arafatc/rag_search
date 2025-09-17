#!/usr/bin/env python3
"""
Minimalistic RAG FastAPI Server - OpenWebUI Compatible
Single file solution for RAG API endpoints
"""

import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from dotenv import load_dotenv

# --- Arize Phoenix Tracing Setup ---
# This block configures the tracer to send data to your local Phoenix instance.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use host.docker.internal for Docker container to access host services
phoenix_host = os.getenv("PHOENIX_HOST", "phoenix")  # Changed to use Docker service name
phoenix_endpoint = f"http://{phoenix_host}:6006"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

try:
    import uuid
    import os
    # Prepare output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    # Generate unique trace id for this query
    trace_id = str(uuid.uuid4())
    from phoenix.otel import register
    tracer_provider = register(
        project_name="default",
        endpoint=f"{phoenix_endpoint}/v1/traces",
        auto_instrument=True  # This automatically instruments CrewAI and other libraries
    )
    logging.info(f"SUCCESS: Arize Phoenix tracing successfully initialized for API server at {phoenix_endpoint}")
except ImportError as e:
    logging.warning(f"WARNING:  Phoenix module not found: {e}. Install with: pip install arize-phoenix")
except Exception as e:
    logging.warning(f"WARNING:  Could not initialize Arize Phoenix tracing: {e}")
# --- End of Tracing Setup ---

# Ensure the project root is in the Python path
import sys
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import your existing crew creation function
from src.rag_search.crew import create_rag_crew

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI(
    title="RAG Search API",
    description="RAG API for OpenWebUI integration",
    version="1.0.0",
)

# Add CORS middleware to allow requests from OpenWebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define the request model to be compatible with OpenAI's format
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API server is running
    """
    try:
        # Check if RAG components are available
        from src.rag_search.crew import create_rag_crew
        rag_available = True
    except ImportError:
        rag_available = False
    
    return {
        "status": "healthy",
        "rag_available": rag_available,
        "model": "Gemma 3:1b",
        "version": "2.1.0",
        "context_length": 4096
    }

@app.get("/v1/models")
def list_models():
    """
    OpenAI-compatible endpoint to list available models.
    This is required for OpenWebUI to discover available models.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-search",
                "object": "model", 
                "created": 1677652288,
                "owned_by": "rag-search",
                "permission": [],
                "root": "rag-search",
                "parent": None,
                "max_tokens": 4096,         # Production-optimized context for Gemma 3:1b
                "context_length": 4096      # Production-optimized context for Gemma 3:1b
            }
        ]
    }

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible endpoint with simple multi-turn conversation support.
    """
    # Extract the last user message as the query
    user_message = next((msg["content"] for msg in reversed(request.messages) if msg["role"] == "user"), None)

    if not user_message:
        return {"error": "No user message found"}

    print(f"Received query for API: {user_message}")
    
    # Filter out auto-generated followup questions that should be ignored
    auto_followup_patterns = [
        "### Task:",
        "Generate a concise",
        "word title with an emoji",
        "summarizing the chat history",
        "JSON format:",
        "chat_history>"
    ]
    
    # Check if this looks like an auto-followup question
    if any(pattern in user_message for pattern in auto_followup_patterns):
        print(f"DEBUG: Ignoring auto-followup question: {user_message[:100]}...")
        # Return a minimal response indicating this type of request is not supported
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion", 
            "created": 1677652288,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This request type is not supported by the RAG system. Please ask questions about documents or policies."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    # Simple conversation context: just format previous messages as context
    conversation_context = ""
    if len(request.messages) > 1:
        # Get all messages except the last one (current query) for context
        previous_messages = request.messages[:-1]
        context_parts = []
        for msg in previous_messages[-4:]:  # Keep only last 4 messages for context
            context_parts.append(f"{msg['role'].title()}: {msg['content']}")
        conversation_context = "\n".join(context_parts)
    
    print(f"Conversation context: {conversation_context[:200]}..." if conversation_context else "No conversation context")

    # Generate trace id and create output JSON file immediately after receiving query
    import uuid
    import os
    import json
    trace_id = str(uuid.uuid4())
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{trace_id}.json")
    # Step 1: Create JSON file with status 'Running', trace_id, and query
    output_json = {
        "status": "Running",
        "trace_id": trace_id,
        "query": user_message,
        "final_answer": "",
        "sources_info": ""
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    try:
        print("DEBUG: Starting crew execution...")
        # Import the required modules
        from src.rag_search.crew import create_rag_crew
        import json
        import re
        
        # Strategy detection from query or use environment variable default
        strategy = os.getenv("DEFAULT_RAG_STRATEGY", "semantic")  # Environment-controlled default
        if "hierarchical" in user_message.lower():
            strategy = "hierarchical"
        elif "contextual" in user_message.lower():
            strategy = "contextual_rag"
        
        print(f"INFO: Using strategy '{strategy}' for query: {user_message[:50]}...")
        
        # Use CrewAI for processing with conversation context and strategy
        # create_rag_crew already executes the crew and returns CrewOutput
        crew_output = create_rag_crew(user_message, conversation_context, strategy)
        
        # Debug: Print the full crew output structure
        print(f"DEBUG: crew_output type: {type(crew_output)}")
        print(f"DEBUG: crew_output raw: {crew_output.raw if crew_output else 'None'}")
        print(f"DEBUG: crew_output full structure: {str(crew_output)[:1000] if crew_output else 'None'}")
        
        # The final result is in the 'raw' attribute of the CrewOutput object
        final_result = crew_output.raw if crew_output else ""

        # The raw output from the last task is also available (same as final_result in this case)
        raw_output = crew_output.raw if crew_output else ""
        
        # The final answer for the user is the 'result' - ensure it's a string
        if hasattr(final_result, 'result'):
            # Handle nested CrewOutput objects
            clean_final_answer = str(final_result.result).strip() if final_result.result else "Unable to generate response."
        else:
            clean_final_answer = str(final_result).strip() if final_result else "Unable to generate response."
        
        print(f"DEBUG: clean_final_answer: {clean_final_answer[:200]}...")
        
        # Update JSON file with completed status and final answer
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                output_json = json.load(f)
        except Exception:
            output_json = {
                "status": "Running",
                "trace_id": trace_id,
                "query": user_message,
                "final_answer": "",
                "sources_info": ""
            }
        
        # Update the status and final answer
        output_json["status"] = "Completed"
        output_json["final_answer"] = clean_final_answer
        
        # Save the updated JSON (tool will update sources_info separately)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        # Format the structured API response with sources and strategy info
        formatted_response = str(clean_final_answer) if clean_final_answer else "Unable to generate response."
        
        # Read the sources_info directly from the updated JSON file
        # The tool should have updated it by now
        print(f"DEBUG: Reading final sources_info from JSON file: {output_path}")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                final_output_json = json.load(f)
            display_sources_data = final_output_json.get("sources_info")
            print(f"DEBUG: Retrieved sources_info from JSON: {display_sources_data}")
        except Exception as e:
            print(f"DEBUG: Error reading sources_info from JSON file: {e}")
            display_sources_data = None  # No fallback
        
        # Add Sources section if we have source data
        if display_sources_data and display_sources_data.get("sources"):
            formatted_response += "\n\nSources\n"
            for source_item in display_sources_data["sources"]:
                document_name = source_item.get("document", "Unknown")
                score = source_item.get("score", 0.0)
                formatted_response += f"\n{document_name} (score: {score:.4f})\n"
        
        # Get strategy and chunks info
        display_strategy = os.getenv("DEFAULT_RAG_STRATEGY", "semantic")  # Environment-controlled default
        display_chunks = 0  # Default
        if display_sources_data:
            display_strategy = display_sources_data.get("strategy", os.getenv("DEFAULT_RAG_STRATEGY", "semantic"))
            display_chunks = display_sources_data.get("chunks_used", 0)
        
        # Add Retrieval Strategy section
        formatted_response += "\n\nRetrieval Strategy\n"
        formatted_response += f"\nMethod: {display_strategy}\n"
        formatted_response += f"\nChunks Used: {display_chunks}"
        
        # Format the response to be compatible with the OpenAI API standard
        # Return final_answer and sources_info in OpenAI-compatible format
        response = {
            "id": "chatcmpl-123", # Dummy ID
            "object": "chat.completion",
            "created": 1677652288, # Dummy timestamp
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": formatted_response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_message.split()) if user_message else 0,
                "completion_tokens": len(str(formatted_response).split()) if formatted_response else 0,
                "total_tokens": len(user_message.split()) + len(str(formatted_response).split()) if user_message and formatted_response else 0
            }
        }
        
        print("DEBUG: About to return response, all processing completed successfully")
        return response
        
    except Exception as e:
        print(f"API Error: {e}")
        return {"error": f"RAG processing failed: {str(e)}"}

@app.get("/status/{instanceId}")
def get_status(instanceId: str):
    """
    Polling endpoint to check the status of a RAG query by instance ID (trace ID).
    Returns the entire JSON file from output folder with the name matching [instanceId].json
    """
    try:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        json_file_path = os.path.join(output_dir, f"{instanceId}.json")

        if not os.path.exists(json_file_path):
            raise HTTPException(status_code=404, detail=f"Instance ID '{instanceId}' not found")

        with open(json_file_path, "r", encoding="utf-8") as f:
            return json.load(f)  # Return the entire JSON content

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON file for instance ID '{instanceId}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading status for instance ID '{instanceId}': {str(e)}")

if __name__ == "__main__":
    # Updated port and host configuration
    port = int(os.getenv("RAG_API_PORT", "8001"))
    host = os.getenv("RAG_API_HOST", "0.0.0.0")
    
    print(f"Starting RAG API Server with Gemma 3:1b (Production Optimized)...")
    print(f"Server will run on: http://{host}:{port}")
    print(f"OpenWebUI endpoint: http://{host}:{port}/v1")
    print(f"Model: Gemma 3:1b (1B parameters, Context: 4,096 tokens, Top-K: 3)")
    
    uvicorn.run(app, host=host, port=port)
