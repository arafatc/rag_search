import os
from crewai import Agent, LLM
from .tools import document_retrieval_tool

# Initialize the Ollama LLM for the agents - configurable model
# Use environment variables for base URL and model to support Docker deployment
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")  # Configurable model, default to qwen2.5:1.5b

# LLM timeout configuration
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "300"))     # 5 minutes timeout
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))               # Retry attempts

ollama_llm = LLM(
    model=f"ollama/{ollama_model}",  # Using configurable model via environment variable
    base_url=ollama_base_url,
    temperature=0.0,  # Zero temperature for fastest, most deterministic responses
    timeout=LLM_REQUEST_TIMEOUT,  # Configurable timeout (default 5 minutes)
    verbose=False,  # Disable verbose logging to reduce overhead
    # Optimized token configuration for speed
    max_tokens=4096,  # Reduced for faster generation
    num_ctx=8192,     # Smaller context window for speed
    top_p=0.9,        # Higher for more predictable responses
    repeat_penalty=1.1,  # Higher penalty to prevent repetition
)

# --- AGENT 1: The Specialist Retriever ---
# This agent's only job is to call the retrieval tool correctly.
document_researcher = Agent(
    role='Document Researcher',
    goal='Use the Document Retrieval Tool exactly ONCE and return results immediately.',
    backstory=(
        "You are a fast document retriever. Call the tool once and return results. No thinking required."
    ),
    tools=[document_retrieval_tool],
    llm=ollama_llm,
    verbose=True,  # Enable verbose for detailed logging
    allow_delegation=False,
    max_iter=1,  # STRICT: Only 1 iteration allowed
    step_callback=None,  # Disable callbacks for speed
)

# --- AGENT 2: The Specialist Synthesizer ---
# This agent's only job is to write the final answer based on the context it receives.
insight_synthesizer = Agent(
    role='Answer Writer',
    goal='Write concise, accurate answers using only the document content provided.',
    backstory=(
        "You are a fast document summarizer. Write direct answers from the provided text only. "
        "Keep responses focused and under 200 words. No external knowledge."
    ),
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=1,  # STRICT: Only 1 iteration allowed
    tools=[],
    step_callback=None,  # Disable callbacks for speed
)