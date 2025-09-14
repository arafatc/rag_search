# Use a PyTorch base with CUDA + cuDNN preinstalled
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies (minimal, since base has most)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure uv to use system Python
ENV UV_SYSTEM_PYTHON=1

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements_docker.txt ./requirements.txt

# Install dependencies (cached unless requirements change)
RUN uv pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY rag_api_server.py ./ 
COPY src/ ./src/
COPY .env* ./

# Expose app port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "rag_api_server:app", "--host", "0.0.0.0", "--port", "8000"]