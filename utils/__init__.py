"""Shared utilities for DeepLearning Performance Lab tutorials.

This package provides common helpers used across all tutorials:
- Logging configuration with standardized formatting
- Benchmark decorator and comparison table utilities
- Simple model factories (CNN, MLP) for demonstration purposes
- Device detection and hardware information display
"""

from utils.logging_config import setup_logging
from utils.benchmark import benchmark, compare_results, print_benchmark_table
from utils.models import SimpleCNN, SimpleMLP, SimpleViT, get_sample_batch
from utils.device import get_device, get_gpu_capability, print_device_info

__all__ = [
    "setup_logging",
    "benchmark",
    "compare_results",
    "print_benchmark_table",
    "SimpleCNN",
    "SimpleMLP",
    "SimpleViT",
    "get_sample_batch",
    "get_device",
    "get_gpu_capability",
    "print_device_info",
]
