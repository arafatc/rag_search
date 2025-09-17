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

def create_rag_crew(query: str, conversation_context: str = None, strategy: str = None):
    """Create and execute a RAG crew for document search and analysis."""
    
    # Use environment variable for default strategy if none provided
    if strategy is None:
        strategy = os.getenv("DEFAULT_RAG_STRATEGY", "semantic")  # Aligned with rag_api_server.py default
    
    # Reset tool call counter for new query
    from .tools import reset_tool_call_counter, set_retrieval_strategy
    reset_tool_call_counter()
    
    # Set the retrieval strategy for this session
    set_retrieval_strategy(strategy)
    
    # Get prompts from Phoenix/JSON with fallback
    if conversation_context:
        # For context-aware prompts, build them with all variables
        retrieval_description = get_prompt("document_retrieval_task_with_context", 
                                         f"Previous conversation context:\n{conversation_context}\n\nCurrent question: '{query}'\n\nFind relevant information in the policy and standards documents for the current question. Consider the conversation history for context if relevant.")
        synthesis_description = get_prompt("answer_synthesis_task_with_context",
                                         f"Previous conversation:\n{conversation_context}\n\nCurrent question: '{query}'\n\nAnalyze the provided document context from the research task and formulate a comprehensive answer to the current question. If relevant, briefly reference our previous conversation.")
        
        # Try to format if template variables exist, otherwise use as-is
        if '{query}' in retrieval_description and '{conversation_context}' in retrieval_description:
            retrieval_description = retrieval_description.format(query=query, conversation_context=conversation_context)
        if '{query}' in synthesis_description and '{conversation_context}' in synthesis_description:
            synthesis_description = synthesis_description.format(query=query, conversation_context=conversation_context)
    else:
        # Use simple prompts without context
        retrieval_description = get_prompt("document_retrieval_task", 
                                         f"Find relevant information in the policy and standards documents for the query: '{query}'.")
        synthesis_description = get_prompt("answer_synthesis_task",
                                         f"Analyze the provided document context from the research task and formulate a comprehensive and accurate answer to the user's original question: '{query}'.")
        
        # Try to format if template variable exists, otherwise use as-is
        if '{query}' in retrieval_description:
            retrieval_description = retrieval_description.format(query=query)
        if '{query}' in synthesis_description:
            synthesis_description = synthesis_description.format(query=query)
    
    # Task 1: Document retrieval using the specialist retriever
    document_retrieval_task = Task(
        description=retrieval_description,
        expected_output="A block of text containing chunks of the most relevant document sections and their source file names.",
        agent=document_researcher
    )

    # Task 2: Answer synthesis using the context from the previous task
    answer_synthesis_task = Task(
        description=synthesis_description,
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

def run_rag_crew_with_logging(query: str, conversation_context: str = "", strategy: str = None):
    """
    Run RAG crew with Phoenix observability logging and conversation context
    """
    # Use environment variable for default strategy if none provided
    if strategy is None:
        strategy = os.getenv("DEFAULT_RAG_STRATEGY", "semantic")  # Aligned with rag_api_server.py default
        
    try:
        # Create and run crew with conversation context and strategy
        result = create_rag_crew(query, conversation_context, strategy)
        
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