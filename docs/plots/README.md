# Benchmark Plots

This directory contains visualizations generated from benchmark experiments.

## Files

- `throughput_comparison.png`: Comparison of Hugging Face vs vLLM throughput
- `throughput_results.json`: Raw benchmark data in JSON format

## Generating Plots

Plots are automatically generated when running experiments:

```bash
cd experiments/throughput_benchmark
python benchmark.py --model facebook/opt-125m
```

The script will save plots to this directory.

## Sample Visualizations

### Throughput Comparison

The throughput comparison plot shows:
- Bar chart comparing tokens/second for Hugging Face vs vLLM
- Speedup factor (vLLM over Hugging Face)

### Expected Results

Typical speedup ranges from 2x to 10x depending on:
- Batch size
- Sequence length
- Model size
- Hardware configuration
