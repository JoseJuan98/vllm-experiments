"""
Throughput Benchmark: Comparing Hugging Face Transformers vs vLLM

This script compares the inference throughput of Hugging Face Transformers
and vLLM for large language model serving.
"""

import argparse
import time
from typing import List, Tuple
import json
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Initialize configuration
config = Config()

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from vllm import LLM, SamplingParams
    from matplotlib import pyplot
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install required packages: uv pip install -e .")
    sys.exit(1)


def generate_prompts(tokenizer, num_prompts: int, prompt_length: int) -> List[str]:
    """Generate random prompts for benchmarking."""
    prompts = []
    base_text = "The quick brown fox jumps over the lazy dog. "
    
    for _ in range(num_prompts):
        # Repeat text to reach desired token count
        text = base_text * (prompt_length // 10 + 1)
        tokens = tokenizer.encode(text, max_length=prompt_length, truncation=True)
        prompt = tokenizer.decode(tokens)
        prompts.append(prompt)
    
    return prompts


def benchmark_huggingface(
    model_name: str,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[float, float]:
    """Benchmark Hugging Face Transformers inference."""
    print("\n=== Benchmarking Hugging Face Transformers ===")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    start_time = time.time()
    total_tokens = 0
    
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        
        total_tokens += outputs.shape[1]
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(prompts)} prompts")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    throughput = total_tokens / elapsed_time
    
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    
    return elapsed_time, throughput


def benchmark_vllm(
    model_name: str,
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[float, float]:
    """Benchmark vLLM inference."""
    print("\n=== Benchmarking vLLM ===")
    
    vllm_config = config.get_vllm_config()
    llm = LLM(
        model=model_name,
        tensor_parallel_size=vllm_config["tensor_parallel_size"],
        gpu_memory_utilization=vllm_config["gpu_memory_utilization"],
    )
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    throughput = total_tokens / elapsed_time
    
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Throughput: {throughput:.2f} tokens/s")
    
    return elapsed_time, throughput


def plot_results(hf_results: dict, vllm_results: dict, output_path: str):
    """Generate comparison plots."""
    config.ensure_directories()
    
    fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(14, 6))
    
    # Throughput comparison
    methods = ['Hugging Face', 'vLLM']
    throughputs = [hf_results['throughput'], vllm_results['throughput']]
    colors = ['#3498db', '#e74c3c']
    
    ax1.bar(methods, throughputs, color=colors, alpha=0.7)
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=12)
    ax1.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(throughputs):
        ax1.text(i, v + max(throughputs) * 0.02, f'{v:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Speedup comparison
    speedup = vllm_results['throughput'] / hf_results['throughput']
    ax2.bar(['Speedup'], [speedup], color='#2ecc71', alpha=0.7)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('vLLM Speedup over Hugging Face', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0, speedup + 0.1, f'{speedup:.2f}x', 
            ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    pyplot.tight_layout()
    pyplot.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Hugging Face vs vLLM throughput"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=config.default_model,
        help="Model name or path"
    )
    benchmark_config = config.get_benchmark_config()
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=benchmark_config["num_prompts"],
        help="Number of prompts to benchmark"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=benchmark_config["prompt_length"],
        help="Length of each prompt in tokens"
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=benchmark_config["output_length"],
        help="Maximum output tokens to generate"
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip Hugging Face benchmark (vLLM only)"
    )
    parser.add_argument(
        "--skip-vllm",
        action="store_true",
        help="Skip vLLM benchmark (Hugging Face only)"
    )
    
    args = parser.parse_args()
    
    print(f"Model: {args.model}")
    print(f"Number of prompts: {args.num_prompts}")
    print(f"Prompt length: {args.prompt_length} tokens")
    print(f"Max output length: {args.output_length} tokens")
    
    # Load tokenizer for prompt generation
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generate prompts
    print("\nGenerating prompts...")
    prompts = generate_prompts(tokenizer, args.num_prompts, args.prompt_length)
    
    results = {}
    
    # Run benchmarks
    if not args.skip_hf:
        hf_time, hf_throughput = benchmark_huggingface(
            args.model,
            prompts,
            args.output_length,
            benchmark_config["temperature"],
            benchmark_config["top_p"],
        )
        results['huggingface'] = {
            'time': hf_time,
            'throughput': hf_throughput,
        }
    
    if not args.skip_vllm:
        vllm_time, vllm_throughput = benchmark_vllm(
            args.model,
            prompts,
            args.output_length,
            benchmark_config["temperature"],
            benchmark_config["top_p"],
        )
        results['vllm'] = {
            'time': vllm_time,
            'throughput': vllm_throughput,
        }
    
    # Save and display results
    if 'huggingface' in results and 'vllm' in results:
        print("\n" + "="*50)
        print("RESULTS SUMMARY")
        print("="*50)
        print(f"Hugging Face: {results['huggingface']['throughput']:.2f} tokens/s")
        print(f"vLLM: {results['vllm']['throughput']:.2f} tokens/s")
        speedup = results['vllm']['throughput'] / results['huggingface']['throughput']
        print(f"Speedup: {speedup:.2f}x")
        
        # Save results to JSON
        results_path = os.path.join(config.plots_dir, "throughput_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
        # Generate plot
        plot_path = os.path.join(config.plots_dir, "throughput_comparison.png")
        plot_results(results['huggingface'], results['vllm'], plot_path)


if __name__ == "__main__":
    main()
