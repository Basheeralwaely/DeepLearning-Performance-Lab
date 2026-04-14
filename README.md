# DeepLearning Performance Lab

Performance-focused PyTorch tutorials covering training tricks, fast inference, TensorRT acceleration, pruning, quantization, and distributed multi-GPU training.

Each tutorial is a standalone `.py` file with rich logging output, detailed inline explanations, and measurable before/after benchmarks.

**Target audience:** ML engineers who want to make their models faster.

## Tutorials

| Category | Folder | Topics |
|----------|--------|--------|
| Profiling & Diagnostics | `profiling/` | PyTorch Profiler, memory profiling, DataLoader tuning, torch.compile |
| Mixed Precision | `mixed_precision/` | AMP, BF16 vs FP16, FP8 with Transformer Engine |
| Model Compression | `compression/` | Pruning, knowledge distillation |
| Inference Optimization | `inference/` | TensorRT, ONNX Runtime, TorchScript |
| Distributed Training | `distributed_training/` | DDP, FSDP, model parallelism, DeepSpeed |
| Pruning | `pruning/` | Structured/unstructured pruning techniques |
| Attention | `attention/` | Attention variants, Flash Attention via SDPA |
| Transformers | `transformers/` | DiT vs UNet architectures |

## Quick Start

```bash
# Run any tutorial directly
python profiling/reference_tutorial.py

# All tutorials use shared utilities from utils/
python -c "from utils import get_device; print(get_device())"
```

## Tutorial Format

Every tutorial follows these conventions:

- **Module docstring** explaining the technique and when to use it
- **Section headers** printed to stdout for clear visual structure
- **Logging** for technique details, measurements, and explanations
- **Benchmark table** at the end comparing baseline vs optimized performance

See `profiling/reference_tutorial.py` for the canonical example.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU recommended (tutorials gracefully fall back to CPU)
- Additional requirements vary by tutorial (TensorRT, DeepSpeed, etc.)

## Project Structure

```
├── profiling/              # Profiling and diagnostics tutorials
├── mixed_precision/        # Mixed precision training tutorials
├── compression/            # Model compression tutorials
├── inference/              # Inference optimization tutorials
├── distributed_training/   # Distributed training tutorials
├── pruning/                # Pruning technique tutorials
├── attention/              # Attention variants and Flash Attention tutorials
├── transformers/           # Transformer architecture tutorials (DiT vs UNet)
└── utils/                  # Shared utilities (logging, benchmarks, models)
    ├── __init__.py         # Public API: setup_logging, benchmark, models, etc.
    ├── logging_config.py   # Standardized logging with timestamped format
    ├── benchmark.py        # @benchmark decorator and comparison tables
    ├── models.py           # SimpleCNN, SimpleMLP, get_sample_batch
    └── device.py           # GPU/CPU detection and hardware info
```
