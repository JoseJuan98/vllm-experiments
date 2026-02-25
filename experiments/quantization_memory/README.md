# Quantization Memory Optimization

Implementation of AWQ (Activation-aware Weight Quantization) for reducing memory footprint.

## Overview

This experiment demonstrates memory optimization techniques using quantization:
- **AWQ**: 4-bit quantization that preserves model quality
- **Memory profiling**: Track GPU memory usage before and after quantization
- **Inference testing**: Validate model quality post-quantization

## Requirements

All dependencies are managed in the root `pyproject.toml`. Install with optional quantization dependencies:

```bash
# From the repository root
uv pip install -e ".[quantization]"
```

Note: For AWQ quantization, you need models that are already quantized with AWQ or use AutoAWQ to quantize models.

## Usage

### Compare FP16 vs AWQ

```bash
python quantize.py --model facebook/opt-125m --compare
```

### Measure specific quantization

```bash
# FP16 baseline
python quantize.py --model facebook/opt-125m

# AWQ quantized
python quantize.py --model TheBloke/Llama-2-7B-AWQ --quantization awq
```

### Test inference quality

```bash
python quantize.py --model TheBloke/Llama-2-7B-AWQ \
  --quantization awq \
  --test-inference
```

## Output

The script generates:
- Console output with memory usage statistics
- `docs/reports/memory_optimization.json`: Detailed memory analysis

## Key Metrics

- **Allocated Memory**: GPU memory actively used by the model
- **Reserved Memory**: Total GPU memory reserved
- **Peak Memory**: Maximum memory usage during loading
- **Memory Savings**: Reduction from baseline to quantized

## Expected Results

AWQ quantization typically achieves:
- **~3-4x memory reduction** (from FP16 to 4-bit)
- **Minimal quality loss** (<1% accuracy degradation)
- **Faster inference** due to reduced memory bandwidth

### Example Memory Comparison

| Method | Memory Usage | Savings |
|--------|-------------|---------|
| FP16   | 13.5 GB     | -       |
| AWQ    | 3.5 GB      | 74%     |

## Quantization Methods

### AWQ (Activation-aware Weight Quantization)
- **Precision**: 4-bit weights
- **Group size**: 128 (default)
- **Advantages**: Preserves activation outliers, better quality than naive quantization
- **Use case**: Production deployment with limited GPU memory

### GPTQ (Generative Pre-trained Transformer Quantization)
- **Precision**: 4-bit weights
- **Method**: One-shot quantization
- **Use case**: Alternative to AWQ with similar performance

## Using Pre-quantized Models

Many models are available pre-quantized on Hugging Face:
- `TheBloke/Llama-2-7B-AWQ`
- `TheBloke/Mistral-7B-v0.1-AWQ`
- Search for models with `-AWQ` or `-GPTQ` suffix

## Quantizing Your Own Models

To quantize a custom model with AWQ:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "your-model-name"
quant_path = "your-model-awq"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize
model.quantize(tokenizer, quant_config={
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
})

# Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
```

## Production Considerations

1. **Model Selection**: Choose appropriate quantization for your use case
2. **Validation**: Always test inference quality after quantization
3. **Memory Planning**: Account for activation memory in addition to weights
4. **Hardware**: Ensure GPU supports required compute capabilities
