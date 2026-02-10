"""
Shared configuration for vLLM experiments.

This module provides common settings and utilities for all experiments.
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration class for vLLM experiments."""
    
    # Model configurations
    default_model: str = "facebook/opt-125m"  # Small model for testing
    production_models: List[str] = field(default_factory=lambda: [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "mistralai/Mistral-7B-v0.1",
    ])
    
    # vLLM engine parameters
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_num_batched_tokens: int = 4096
    vllm_max_num_seqs: int = 256
    
    # Benchmark parameters
    benchmark_num_prompts: int = 100
    benchmark_prompt_length: int = 128
    benchmark_output_length: int = 128
    benchmark_temperature: float = 0.7
    benchmark_top_p: float = 0.9
    
    # API server configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_timeout: int = 600
    
    # Quantization configuration
    quantization_method: str = "awq"  # AWQ quantization
    quantization_bits: int = 4
    quantization_group_size: int = 128
    
    # Paths (computed)
    experiments_root: str = field(init=False)
    project_root: str = field(init=False)
    docs_root: str = field(init=False)
    plots_dir: str = field(init=False)
    reports_dir: str = field(init=False)
    
    def __post_init__(self):
        """Initialize computed paths."""
        self.experiments_root = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.experiments_root)
        self.docs_root = os.path.join(self.project_root, "docs")
        self.plots_dir = os.path.join(self.docs_root, "plots")
        self.reports_dir = os.path.join(self.docs_root, "reports")
    
    def get_vllm_config(self) -> Dict[str, Any]:
        """Get vLLM configuration as a dictionary."""
        return {
            "tensor_parallel_size": self.vllm_tensor_parallel_size,
            "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "max_num_batched_tokens": self.vllm_max_num_batched_tokens,
            "max_num_seqs": self.vllm_max_num_seqs,
        }
    
    def get_benchmark_config(self) -> Dict[str, Any]:
        """Get benchmark configuration as a dictionary."""
        return {
            "num_prompts": self.benchmark_num_prompts,
            "prompt_length": self.benchmark_prompt_length,
            "output_length": self.benchmark_output_length,
            "temperature": self.benchmark_temperature,
            "top_p": self.benchmark_top_p,
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API server configuration as a dictionary."""
        return {
            "host": self.api_host,
            "port": self.api_port,
            "timeout": self.api_timeout,
        }
    
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration as a dictionary."""
        return {
            "method": self.quantization_method,
            "bits": self.quantization_bits,
            "group_size": self.quantization_group_size,
        }
    
    def get_model_name(self, model_path: Optional[str] = None) -> str:
        """Get the model name from path or use default."""
        return model_path if model_path else self.default_model
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
