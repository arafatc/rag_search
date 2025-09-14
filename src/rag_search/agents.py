import os
from crewai import Agent, LLM
from .tools import document_retrieval_tool

# Initialize the Ollama LLM for the agents - using gemma3:1b optimized configuration
# Use environment variable for base URL to support Docker deployment
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
ollama_llm = LLM(
    model="ollama/gemma3:1b",  # Using gemma3:1b - lightweight but capable model
    base_url=ollama_base_url,
    temperature=1,  # Higher temperature like reference implementation
    timeout=300,  # Longer timeout like reference
    verbose=True,  # Enable verbose logging for debugging
    # Optimized token configuration for gemma3:1b  
    max_tokens=131072,  # Use maximum like reference implementation
    num_ctx=131072,     # Large context window like reference
    top_p=0.9,        # Nucleus sampling for better quality
    repeat_penalty=1.1,  # Prevent repetition
)

# --- AGENT 1: The Specialist Retriever ---
# This agent's only job is to call the retrieval tool correctly.
document_researcher = Agent(
    role='Document Researcher',
    goal='Use the Document Retrieval Tool to find information relevant to a user\'s query from the knowledge base.',
    backstory=(
        "You are an information retrieval specialist. Your role is strictly limited to: "
        "1) Analyze the user's query to understand intent, "
        "2) Retrieve relevant text chunks using the Document Retrieval Tool, "
        "3) Return only the raw retrieved context - no interpretation or answers. "
        "DO NOT answer questions using your general knowledge. "
        "DO NOT provide explanations, summaries, or interpretations. "
        "ONLY return the exact text chunks retrieved from the tool for the next agent to use."
    ),
    tools=[document_retrieval_tool],
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=3,  # Use reference implementation value
)

# --- AGENT 2: The Specialist Synthesizer ---
# This agent's only job is to write the final answer based on the context it receives.
insight_synthesizer = Agent(
    role='Insight Synthesizer',
    goal='Answer user questions clearly using only the provided context.',
    backstory=(
        "You create clear answers using only the provided context. "
        "Key rules: "
        "- Answer directly and naturally "
        "- Use ONLY the provided context "
        "- Be concise for simple questions "
        "- Include relevant references naturally "
        "- If context is insufficient, say so clearly "
    ),
    llm=ollama_llm,
    verbose=True,
    allow_delegation=False,
    max_iter=1,  # Synthesis agent doesn't need tools, can stay at 1
    # This agent does not need tools; it only processes text.
    tools=[]
)