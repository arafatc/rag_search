import os
from crewai import Agent, LLM
from .tools import document_retrieval_tool

# Initialize the Ollama LLM for the agents - balanced configuration for quality and performance
# Use environment variable for base URL to support Docker deployment
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
ollama_llm = LLM(
    model="ollama/gemma3:1b",  # Using gemma3:1b - lightweight but capable model
    base_url=ollama_base_url,
    temperature=0.1,  # Low temperature for focused responses
    timeout=180,  # Increased timeout to allow proper processing
    verbose=False,  # Disable verbose logging to reduce overhead
    # Balanced token configuration for quality processing
    max_tokens=1024,  # Increased token limit for complete responses
    num_ctx=4096,     # Larger context window to properly process retrieved documents
    top_p=0.8,        # Slightly higher for better context utilization
    repeat_penalty=1.05,  # Standard penalty
)

# --- AGENT 1: The Specialist Retriever ---
# This agent's only job is to call the retrieval tool correctly.
document_researcher = Agent(
    role='Document Researcher',
    goal='Use the Document Retrieval Tool exactly ONCE and immediately return the results.',
    backstory=(
        "Retrieve documents and return them immediately. No analysis needed."
    ),
    tools=[document_retrieval_tool],
    llm=ollama_llm,
    verbose=False,  # Disable verbose for speed
    allow_delegation=False,
    max_iter=1,  # STRICT: Only 1 iteration allowed
    max_execution_time=240,  # Increased timeout for complex retrieval operations
)

# --- AGENT 2: The Specialist Synthesizer ---
# This agent's only job is to write the final answer based on the context it receives.
insight_synthesizer = Agent(
    role='Insight Synthesizer',
    goal='Answer user questions STRICTLY using ONLY the provided retrieved documents. Never use external knowledge.',
    backstory=(
        "You are a document-based assistant that provides helpful answers from retrieved content. "
        "CRITICAL RULES: "
        "1. Use ONLY information from the retrieved documents provided to you "
        "2. If the documents contain relevant information, provide a comprehensive answer based on that content "
        "3. NEVER use your training knowledge about topics like GDPR, EU regulations, or any other external information "
        "4. ONLY reference content that appears in the retrieved document text "
        "5. If the retrieved documents don't contain any relevant information, clearly state it's not available "
        "6. Always quote directly from the provided documents when possible "
        "7. When documents contain related information, explain how it relates to the query "
        "8. Be helpful - if documents contain useful information, share it even if not a perfect match"
    ),
    llm=ollama_llm,
    verbose=False,  # Disable verbose for speed
    allow_delegation=False,
    max_iter=1,  # Synthesis agent doesn't need tools, can stay at 1
    max_execution_time=180,  # Increased timeout to allow for complex synthesis
    # This agent does not need tools; it only processes text.
    tools=[]
)