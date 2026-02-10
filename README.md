# vLLM Experiments

A portfolio of infrastructure experiments for LLM serving using **vLLM** - a high-throughput and memory-efficient inference and serving engine for Large Language Models.

## 🎯 Overview

This repository contains practical experiments demonstrating vLLM's capabilities for production LLM deployments:

- **Throughput Optimization**: Benchmark and compare inference performance
- **API Deployment**: Production-ready OpenAI-compatible serving
- **Memory Efficiency**: Quantization techniques for reduced GPU requirements

## 📁 Repository Structure

```
vllm-experiments/
├── experiments/
│   ├── config.py                    # Shared configuration
│   ├── throughput_benchmark/        # HuggingFace vs vLLM comparison
│   │   ├── benchmark.py
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── openai_api_server/          # OpenAI-compatible API + Streamlit UI
│   │   ├── server.py
│   │   ├── ui.py
│   │   ├── requirements.txt
│   │   └── README.md
│   └── quantization_memory/        # AWQ quantization implementation
│       ├── quantize.py
│       ├── requirements.txt
│       └── README.md
├── docs/
│   ├── plots/                       # Benchmark visualizations
│   │   └── README.md
│   └── reports/                     # Technical analysis reports
│       └── README.md
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- GPU with compute capability 7.0+ (V100, A100, RTX 3090, etc.)

### Installation

Each experiment has its own dependencies. Install globally or use virtual environments:

```bash
# Create virtual environment
python -m venv vllm-env
source vllm-env/bin/activate  # On Windows: vllm-env\Scripts\activate

# Install for specific experiment
cd experiments/throughput_benchmark
pip install -r requirements.txt
```

## 🧪 Experiments

### 1. Throughput Benchmark

Compare inference throughput between Hugging Face Transformers and vLLM.

```bash
cd experiments/throughput_benchmark
python benchmark.py --model facebook/opt-125m --num-prompts 100
```

**Key Results:**
- vLLM achieves 2-10x higher throughput
- Continuous batching and PagedAttention optimizations
- Automated visualization of performance gains

[📖 Full Documentation](experiments/throughput_benchmark/README.md)

### 2. OpenAI API Server

Deploy a production-ready OpenAI-compatible API server with interactive UI.

```bash
# Terminal 1: Start API server
cd experiments/openai_api_server
python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-125m \
  --host 0.0.0.0 \
  --port 8000

# Terminal 2: Launch Streamlit UI
streamlit run ui.py
```

**Features:**
- OpenAI-compatible REST API (`/v1/completions`, `/v1/chat/completions`)
- Interactive Streamlit web interface
- Real-time parameter tuning
- Chat and completion modes

[📖 Full Documentation](experiments/openai_api_server/README.md)

### 3. Quantization Memory Optimization

Implement AWQ quantization to reduce memory footprint by ~70%.

```bash
cd experiments/quantization_memory
python quantize.py --model facebook/opt-125m --compare
```

**Benefits:**
- 3-4x memory reduction with AWQ 4-bit quantization
- Minimal quality degradation (<1%)
- Enables larger models on consumer GPUs
- Detailed memory profiling and reports

[📖 Full Documentation](experiments/quantization_memory/README.md)

## 📊 Results & Documentation

### Benchmark Visualizations

Results are automatically saved to `docs/plots/`:
- Throughput comparison charts
- Speedup metrics
- Raw benchmark data (JSON)

### Technical Reports

Detailed analysis in `docs/reports/`:
- Memory usage breakdowns
- Quantization impact analysis
- Performance metrics

## 🛠️ Configuration

Shared configuration in `experiments/config.py`:

```python
# Model configurations
DEFAULT_MODEL = "facebook/opt-125m"

# vLLM engine parameters
VLLM_CONFIG = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.9,
}

# Benchmark parameters
BENCHMARK_CONFIG = {
    "num_prompts": 100,
    "prompt_length": 128,
    "output_length": 128,
}
```

## 📚 Learn More

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [AWQ Quantization Paper](https://arxiv.org/abs/2306.00978)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new experiments
- Improve existing code
- Report issues or suggest features

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

- [vLLM Team](https://github.com/vllm-project/vllm) for the excellent inference engine
- Hugging Face for model hosting and transformers library
- AutoAWQ for quantization tools