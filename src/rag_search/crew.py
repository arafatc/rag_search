import os
import sys
from crewai import Crew, Process, Task
from .agents import document_researcher, insight_synthesizer
from .tools import document_retrieval_tool

# Phoenix integration
try:
    from phoenix.otel import register
    from opentelemetry import trace
    
    # Import our minimal Phoenix prompt manager
    from .phoenix_prompts import get_prompt as get_phoenix_prompt
    
    # Initialize tracer for crew operations
    tracer = trace.get_tracer(__name__)
    
    def get_prompt(name: str, default: str = "") -> str:
        """Get prompt from Phoenix/JSON with environment fallback"""
        # Try Phoenix prompts first
        prompt = get_phoenix_prompt(name)
        if prompt:
            return prompt
        
        # Fallback to existing environment variables (NO BREAKING CHANGES)
        return os.getenv(f"PROMPT_{name.upper()}", default)
    
    def log_interaction(query: str, response: str, metadata=None) -> None:
        """Create Phoenix spans for crew interactions"""
        with tracer.start_as_current_span("crew_interaction") as span:
            span.set_attribute("query.text", query[:200])
            span.set_attribute("response.length", len(response) if response else 0)
            span.set_attribute("interaction.source", "crew")
            
            if metadata:
                for key, value in metadata.items():
                    span.set_attribute(f"metadata.{key}", str(value))
    
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    def get_prompt(name: str, default: str = "") -> str:
        """Fallback when Phoenix is not available"""
        return os.getenv(f"PROMPT_{name.upper()}", default)
    def log_interaction(query: str, response: str, metadata=None) -> None:
        pass

# Fix encoding issues on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

def create_rag_crew(query: str, conversation_context: str = None):
    """Create and execute a RAG crew for document search and analysis."""
    
    # Reset tool call counter for new query
    from .tools import reset_tool_call_counter
    reset_tool_call_counter()
    
    # Task 1: Document retrieval using the specialist retriever
    document_retrieval_task = Task(
        description=f"Find relevant information in the policy and standards documents for the query: '{query}'.",
        expected_output="A block of text containing chunks of the most relevant document sections and their source file names.",
        agent=document_researcher
    )

    # Task 2: Answer synthesis using the context from the previous task
    answer_synthesis_task = Task(
        description=f"Analyze the provided document context from the research task and formulate a comprehensive and accurate answer to the user's original question: '{query}'.",
        expected_output="A professional, well-structured response that directly answers the user's question with proper source citations.",
        agent=insight_synthesizer,
        context=[document_retrieval_task]
    )

    # Create and execute crew with both agents
    crew = Crew(
        agents=[document_researcher, insight_synthesizer],
        tasks=[document_retrieval_task, answer_synthesis_task],
        process=Process.sequential,
        verbose=True
    )

    # Execute the crew and return results
    try:
        result = crew.kickoff()
        print(f"DEBUG: crew execution completed, result type: {type(result)}")
        
        # Log interaction if Phoenix is available
        if PHOENIX_AVAILABLE:
            log_interaction(query, str(result), {"crew_usage": "two_agent_specialized"})
        
        return result
    except Exception as e:
        print(f"ERROR: crew execution failed: {e}")
        return None

def run_rag_crew_with_logging(query: str, conversation_context: str = ""):
    """
    Run RAG crew with Phoenix observability logging and conversation context
    """
    try:
        # Create and run crew with conversation context
        result = create_rag_crew(query, conversation_context)
        
        # Log interaction for Phoenix observability
        if PHOENIX_AVAILABLE:
            log_interaction(
                query=query,
                response=str(result),
                metadata={
                    "crew_type": "rag_system",
                    "agents_count": 2,
                    "tasks_count": 2,
                    "has_conversation_context": bool(conversation_context),
                    "context_length": len(conversation_context) if conversation_context else 0
                }
            )
        
        return result
        
    except Exception as e:
        # Log error for Phoenix observability
        if PHOENIX_AVAILABLE:
            log_interaction(
                query=query,
                response=f"ERROR: {str(e)}",
                metadata={"error": True, "error_type": type(e).__name__}
            )
        raise