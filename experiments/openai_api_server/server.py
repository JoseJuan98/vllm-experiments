"""
OpenAI-compatible API Server using vLLM

This module provides an OpenAI-compatible API server for serving LLMs with vLLM.
"""

import argparse
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Initialize configuration
config = Config()

try:
    from vllm.entrypoints.openai.api_server import run_server
    import uvicorn
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install required packages: pip install vllm uvicorn")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Start OpenAI-compatible API server with vLLM"
    )
    api_config = config.get_api_config()
    vllm_config = config.get_vllm_config()
    
    parser.add_argument(
        "--model",
        type=str,
        default=config.default_model,
        help="Model name or path to serve"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=api_config["host"],
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=api_config["port"],
        help="Server port"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=vllm_config["tensor_parallel_size"],
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=vllm_config["gpu_memory_utilization"],
        help="GPU memory utilization (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting OpenAI-compatible API server...")
    print(f"Model: {args.model}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    
    # Note: This is a simplified version. In production, you would use:
    # python -m vllm.entrypoints.openai.api_server --model <model> --host <host> --port <port>
    
    print("\nTo start the server, run:")
    print(f"python -m vllm.entrypoints.openai.api_server \\")
    print(f"  --model {args.model} \\")
    print(f"  --host {args.host} \\")
    print(f"  --port {args.port} \\")
    print(f"  --tensor-parallel-size {args.tensor_parallel_size} \\")
    print(f"  --gpu-memory-utilization {args.gpu_memory_utilization}")


if __name__ == "__main__":
    main()
