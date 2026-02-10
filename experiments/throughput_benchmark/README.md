# Throughput Benchmark

Comparison of inference throughput between Hugging Face Transformers and vLLM.

## Overview

This experiment benchmarks the performance of two popular LLM serving frameworks:
- **Hugging Face Transformers**: Standard PyTorch-based inference
- **vLLM**: Optimized inference engine with PagedAttention

## Requirements

```bash
pip install vllm transformers torch matplotlib numpy
```

## Usage

Basic benchmark with default settings:
```bash
python benchmark.py
```

Custom model and parameters:
```bash
python benchmark.py \
  --model meta-llama/Llama-2-7b-hf \
  --num-prompts 200 \
  --prompt-length 128 \
  --output-length 256
```

Run only vLLM (skip Hugging Face):
```bash
python benchmark.py --skip-hf
```

## Output

The script generates:
- Console output with timing and throughput metrics
- `docs/plots/throughput_comparison.png`: Visual comparison chart
- `docs/plots/throughput_results.json`: Raw benchmark data

## Key Metrics

- **Throughput**: Tokens generated per second
- **Speedup**: vLLM throughput / Hugging Face throughput
- **Total time**: End-to-end execution time

## Expected Results

vLLM typically achieves 2-10x higher throughput than Hugging Face Transformers, 
especially for larger batch sizes and longer sequences, thanks to:
- Continuous batching
- PagedAttention memory optimization
- Optimized CUDA kernels
