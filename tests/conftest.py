"""
pytest configuration and fixtures for the RAG system tests.

This module ensures that environment variables are properly loaded
when running tests through VS Code or other test runners.
"""

import os
import sys
from pathlib import Path
import pytest
from dotenv import load_dotenv

# Add the src directory to the Python path
ROOT_DIR = Path(__file__).parent.parent
src_path = ROOT_DIR / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def pytest_configure(config):
    """Configure pytest and load environment variables."""
    # Set encoding to handle Unicode characters from CrewAI
    import sys
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)
    
    # Set environment variables for proper encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # Load environment variables from .env file
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"[OK] Loaded environment variables from {env_path}")
    
    # Ensure critical environment variables are set for tests
    required_env_vars = {
        "PGVECTOR_CONN": "postgresql://postgres:postgres@localhost:5432/rag_db",
        "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/rag_db", 
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "EMBEDDING_MODEL": "nomic-embed-text:latest",
        "EMBEDDING_DIM": "768",  # Add externalized embedding dimension
        "LLM_MODEL": "gemma3:1b",
        "LLM_PROVIDER": "ollama",
        "EMBEDDING_PROVIDER": "ollama"
    }
    
    for var, default_value in required_env_vars.items():
        if not os.getenv(var):
            os.environ[var] = default_value
            print(f"[OK] Set default {var}={default_value}")

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment automatically for all tests."""
    # This fixture runs automatically for all tests
    # Additional setup can be added here if needed
    yield
    # Cleanup code can go here if needed

@pytest.fixture
def temp_env_var():
    """Fixture to temporarily set environment variables for individual tests."""
    original_env = os.environ.copy()
    yield
    # Restore original environment after test
    os.environ.clear()
    os.environ.update(original_env)
