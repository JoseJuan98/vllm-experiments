"""
AWQ Quantization for Memory Optimization

This script demonstrates how to use AWQ (Activation-aware Weight Quantization)
to reduce model memory footprint while maintaining accuracy.
"""

import argparse
import sys
import os
import json
from typing import Optional

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Initialize configuration
config = Config()

try:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install required packages: pip install vllm transformers torch")
    sys.exit(1)


def get_model_memory_usage(model_name: str, quantization: Optional[str] = None) -> dict:
    """
    Measure memory usage of a model with optional quantization.
    
    Args:
        model_name: Name or path of the model
        quantization: Quantization method (e.g., 'awq', 'gptq', None)
    
    Returns:
        Dictionary with memory usage statistics
    """
    print(f"\n{'='*60}")
    print(f"Measuring memory usage for: {model_name}")
    if quantization:
        print(f"Quantization: {quantization}")
    else:
        print(f"Quantization: None (FP16)")
    print(f"{'='*60}\n")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Load model with vLLM
    vllm_config = config.get_vllm_config()
    llm = LLM(
        model=model_name,
        quantization=quantization,
        tensor_parallel_size=vllm_config["tensor_parallel_size"],
        gpu_memory_utilization=vllm_config["gpu_memory_utilization"],
    )
    
    # Measure memory
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved_memory = torch.cuda.memory_reserved() / 1024**3  # GB
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    else:
        allocated_memory = 0
        reserved_memory = 0
        peak_memory = 0
    
    results = {
        "model": model_name,
        "quantization": quantization or "none",
        "allocated_memory_gb": round(allocated_memory, 2),
        "reserved_memory_gb": round(reserved_memory, 2),
        "peak_memory_gb": round(peak_memory, 2),
    }
    
    print(f"Allocated Memory: {results['allocated_memory_gb']:.2f} GB")
    print(f"Reserved Memory: {results['reserved_memory_gb']:.2f} GB")
    print(f"Peak Memory: {results['peak_memory_gb']:.2f} GB")
    
    return results


def test_inference(model_name: str, quantization: Optional[str] = None):
    """Test inference quality with quantized model."""
    print(f"\nTesting inference quality...")
    
    vllm_config = config.get_vllm_config()
    llm = LLM(
        model=model_name,
        quantization=quantization,
        tensor_parallel_size=vllm_config["tensor_parallel_size"],
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    test_prompts = [
        "The capital of France is",
        "In machine learning, neural networks are",
        "The quick brown fox",
    ]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
    )
    
    outputs = llm.generate(test_prompts, sampling_params)
    
    print("\nSample Outputs:")
    print("-" * 60)
    for prompt, output in zip(test_prompts, outputs):
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Output: {generated_text}")
    print("-" * 60)


def compare_quantization_methods(model_name: str):
    """Compare different quantization methods."""
    methods = [
        ("None (FP16)", None),
        ("AWQ", "awq"),
    ]
    
    results = []
    
    for method_name, quantization in methods:
        try:
            result = get_model_memory_usage(model_name, quantization)
            result["method_name"] = method_name
            results.append(result)
        except Exception as e:
            print(f"Error with {method_name}: {e}")
            results.append({
                "method_name": method_name,
                "quantization": quantization or "none",
                "error": str(e)
            })
    
    return results


def generate_memory_report(results: list, output_path: str):
    """Generate a detailed memory usage report."""
    config.ensure_directories()
    
    report = {
        "title": "Memory Optimization with AWQ Quantization",
        "results": results,
    }
    
    # Calculate savings
    if len(results) >= 2 and "allocated_memory_gb" in results[0]:
        baseline = results[0]["allocated_memory_gb"]
        quantized = results[1]["allocated_memory_gb"]
        savings = baseline - quantized
        savings_pct = (savings / baseline) * 100 if baseline > 0 else 0
        
        report["summary"] = {
            "baseline_memory_gb": baseline,
            "quantized_memory_gb": quantized,
            "memory_savings_gb": round(savings, 2),
            "memory_savings_percent": round(savings_pct, 2),
        }
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("MEMORY OPTIMIZATION REPORT")
    print(f"{'='*60}")
    
    if "summary" in report:
        summary = report["summary"]
        print(f"\nBaseline (FP16): {summary['baseline_memory_gb']:.2f} GB")
        print(f"Quantized (AWQ): {summary['quantized_memory_gb']:.2f} GB")
        print(f"Memory Savings: {summary['memory_savings_gb']:.2f} GB ({summary['memory_savings_percent']:.1f}%)")
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AWQ Quantization for Memory Optimization"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.default_model,
        help="Model name or path"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["awq", "gptq", None],
        default=None,
        help="Quantization method to use"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare FP16 vs AWQ quantization"
    )
    parser.add_argument(
        "--test-inference",
        action="store_true",
        help="Test inference quality after quantization"
    )
    
    args = parser.parse_args()
    
    print(f"Model: {args.model}")
    
    if args.compare:
        # Compare different methods
        results = compare_quantization_methods(args.model)
        
        # Generate report
        report_path = os.path.join(config.reports_dir, "memory_optimization.json")
        generate_memory_report(results, report_path)
    else:
        # Single method measurement
        result = get_model_memory_usage(args.model, args.quantization)
        
        if args.test_inference:
            test_inference(args.model, args.quantization)


if __name__ == "__main__":
    main()
