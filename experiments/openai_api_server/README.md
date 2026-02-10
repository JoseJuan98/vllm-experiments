# OpenAI API Server

OpenAI-compatible API server deployment using vLLM with an interactive Streamlit UI.

## Overview

This experiment demonstrates how to deploy a production-ready LLM serving infrastructure:
- **API Server**: OpenAI-compatible REST API powered by vLLM
- **Web UI**: Interactive Streamlit interface for testing and interaction

## Requirements

```bash
pip install vllm streamlit requests uvicorn
```

## Quick Start

### 1. Start the API Server

```bash
# Using vLLM's built-in server
python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-125m \
  --host 0.0.0.0 \
  --port 8000
```

Or use the helper script:
```bash
python server.py --model facebook/opt-125m
```

### 2. Launch the Streamlit UI

In a separate terminal:
```bash
streamlit run ui.py
```

The UI will be available at http://localhost:8501

## Features

### API Server
- OpenAI-compatible endpoints (`/v1/completions`, `/v1/chat/completions`)
- High-throughput inference with vLLM
- Automatic batching and optimization
- GPU acceleration

### Streamlit UI
- **Chat Mode**: Interactive chat interface with conversation history
- **Completion Mode**: Single-turn text completion
- **Real-time parameter tuning**: Temperature, top_p, max_tokens
- **Response metadata**: Token counts, timing, and JSON inspection
- **Connection testing**: Verify server availability

## API Usage

### Using cURL

```bash
# Completion
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "San Francisco is a",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Chat
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### Using Python OpenAI Client

```python
import openai

# Configure client
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "EMPTY"

# Generate completion
response = openai.Completion.create(
    model="facebook/opt-125m",
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].text)
```

## Configuration

Server parameters can be customized via command-line arguments:
- `--model`: Model name or path
- `--host`: Server host (default: 0.0.0.0)
- `--port`: Server port (default: 8000)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--gpu-memory-utilization`: GPU memory fraction (0.0-1.0)

## Production Deployment

For production use, consider:
1. Running behind a reverse proxy (nginx, traefik)
2. Implementing authentication/rate limiting
3. Monitoring and logging
4. Load balancing for multiple instances
5. HTTPS/TLS encryption
