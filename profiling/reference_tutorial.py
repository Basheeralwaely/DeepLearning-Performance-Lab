"""
Reference Tutorial: torch.inference_mode() vs Standard Forward Pass
====================================================================

torch.inference_mode() is a context manager that disables gradient computation
and autograd tracking entirely, providing faster execution for inference
workloads compared to even torch.no_grad(). While torch.no_grad() simply
prevents gradient accumulation, inference_mode() additionally disables
view-tracking and version-counting on tensors, yielding extra speedup.

When to use:
  - Any time you run model inference and do NOT need gradients
  - Serving, evaluation, benchmarking, feature extraction

What this tutorial demonstrates:
  1. Baseline forward pass with default autograd tracking
  2. Optimized forward pass using torch.inference_mode()
  3. Side-by-side benchmark comparison table
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import (
    setup_logging,
    benchmark,
    compare_results,
    SimpleCNN,
    get_sample_batch,
    get_device,
    print_device_info,
)

logger = setup_logging("reference_tutorial")

NUM_ITERATIONS = 100
BATCH_SIZE = 64


def main():
    # Section headers use print() per D-04
    print("\n" + "=" * 60)
    print("  Reference Tutorial: torch.inference_mode() Performance")
    print("=" * 60 + "\n")

    # Device info
    device = get_device()
    print_device_info()

    # Setup model and data
    logger.info("Preparing model and sample data...")
    model = SimpleCNN().to(device)
    model.eval()
    x, _ = get_sample_batch(batch_size=BATCH_SIZE, device=device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: SimpleCNN with {total_params:,} parameters")
    logger.info(f"Input: batch_size={BATCH_SIZE}, shape={tuple(x.shape)}")
    logger.info(f"Iterations per benchmark: {NUM_ITERATIONS}")

    # Warm-up: ensure CUDA kernels are compiled before timed runs
    logger.info("Running warm-up pass to initialize CUDA kernels...")
    with torch.inference_mode():
        for _ in range(10):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ----------------------------------------------------------------
    # Section 1: Baseline - Standard forward pass (autograd enabled)
    # ----------------------------------------------------------------
    print("\n--- Section 1: Standard Forward Pass (autograd enabled) ---\n")
    logger.info("Running forward pass with gradient tracking enabled...")
    logger.info("Autograd will build computation graph (unnecessary for inference)")

    @benchmark
    def baseline_forward():
        for _ in range(NUM_ITERATIONS):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    baseline = baseline_forward()
    logger.info(f"Baseline completed in {baseline['time_seconds']:.4f}s")

    # ----------------------------------------------------------------
    # Section 2: Optimized - inference_mode() forward pass
    # ----------------------------------------------------------------
    print("\n--- Section 2: inference_mode() Forward Pass ---\n")
    logger.info("Running forward pass with torch.inference_mode()...")
    logger.info("Autograd tracking fully disabled -- no graph, no version counting")

    @benchmark
    def optimized_forward():
        with torch.inference_mode():
            for _ in range(NUM_ITERATIONS):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    optimized = optimized_forward()
    logger.info(f"Optimized completed in {optimized['time_seconds']:.4f}s")

    # ----------------------------------------------------------------
    # Results comparison per D-05
    # ----------------------------------------------------------------
    print("\n--- Results ---\n")
    compare_results(baseline, optimized, "torch.inference_mode()")

    # ----------------------------------------------------------------
    # Key Takeaways
    # ----------------------------------------------------------------
    print("\n--- Key Takeaways ---\n")
    speedup = baseline["time_seconds"] / optimized["time_seconds"] if optimized["time_seconds"] > 0 else float("inf")
    logger.info(f"Speedup achieved: {speedup:.2f}x")
    logger.info("inference_mode() disables autograd entirely (faster than no_grad())")
    logger.info("Use inference_mode() for all inference workloads where gradients are not needed")
    logger.info("Speedup varies by model size, batch size, and hardware")
    logger.info("On GPU, the difference is often smaller because compute dominates")
    logger.info("On CPU, autograd overhead is a larger fraction of total time")

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
