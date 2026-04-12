"""
Automatic Mixed Precision (AMP) Training
=========================================

Automatic Mixed Precision uses torch.amp.autocast to run forward passes in
float16 while keeping master weights in float32, and torch.amp.GradScaler to
prevent gradient underflow. This gives significant speedup and memory savings
on NVIDIA GPUs with Tensor Cores (Volta and newer).

NOTE: This tutorial uses the modern torch.amp API (not the deprecated
torch.cuda.amp API which produces FutureWarning in PyTorch 2.x).

When to use:
  - Training any model on GPU where speed matters more than FP64 precision
  - Models with large batch sizes that are memory-constrained

What this tutorial demonstrates:
  1. Baseline FP32 training loop
  2. AMP training with autocast + GradScaler
  3. Side-by-side benchmark comparison (throughput and memory)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from utils import (
    setup_logging,
    benchmark,
    compare_results,
    SimpleCNN,
    get_sample_batch,
    get_device,
    print_device_info,
)

logger = setup_logging("amp_training")

NUM_ITERATIONS = 100
BATCH_SIZE = 64
WARMUP_ITERATIONS = 10


def main():
    # ================================================================
    # Section 0: Setup
    # ================================================================
    print("\n" + "=" * 60)
    print("  AMP Training: Automatic Mixed Precision")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    if device.type != "cuda":
        logger.warning("AMP requires a CUDA GPU. Running on CPU will not show speedup.")
        logger.warning("Results will still run but benchmark differences will be minimal.")

    model = SimpleCNN().to(device)
    inputs, labels = get_sample_batch(batch_size=BATCH_SIZE, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: SimpleCNN with {total_params:,} parameters")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Training iterations per benchmark: {NUM_ITERATIONS}")
    logger.info(f"Warmup iterations: {WARMUP_ITERATIONS}")

    # ================================================================
    # Section 1: Warmup
    # ================================================================
    logger.info("Running warmup passes to compile CUDA kernels...")
    logger.info("Warmup matters because the first CUDA kernel launches include")
    logger.info("JIT compilation and memory allocation overhead that would skew benchmarks.")

    for _ in range(WARMUP_ITERATIONS):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    logger.info("Warmup complete. CUDA kernels compiled and memory pools initialized.")

    # ================================================================
    # Section 2: Baseline FP32 Training
    # ================================================================
    print("\n--- Section 1: Baseline FP32 Training ---\n")

    logger.info("All operations run in float32 -- full precision but slower.")
    logger.info("This is the default PyTorch behavior with no mixed precision.")

    # Re-initialize model for fair comparison
    model = SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    @benchmark
    def fp32_training():
        for _ in range(NUM_ITERATIONS):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    baseline = fp32_training()
    logger.info(f"FP32 baseline completed in {baseline['time_seconds']:.4f}s")

    # ================================================================
    # Section 3: AMP Training (FP16 + GradScaler)
    # ================================================================
    print("\n--- Section 2: AMP Training (FP16 + GradScaler) ---\n")

    logger.info("autocast automatically casts operations to float16 where safe")
    logger.info("GradScaler prevents gradient underflow by scaling the loss before backward")
    logger.info("Master weights remain in float32 for numerical stability")

    # Re-initialize model for fair comparison
    model = SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler('cuda')

    @benchmark
    def amp_training():
        for _ in range(NUM_ITERATIONS):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()

    optimized = amp_training()
    logger.info(f"AMP training completed in {optimized['time_seconds']:.4f}s")

    # ================================================================
    # Section 4: Results
    # ================================================================
    print("\n--- Results ---\n")
    compare_results(baseline, optimized, "Automatic Mixed Precision (AMP)")

    # ================================================================
    # Section 5: Key Takeaways
    # ================================================================
    print("\n--- Key Takeaways ---\n")

    speedup = baseline["time_seconds"] / optimized["time_seconds"] if optimized["time_seconds"] > 0 else float("inf")
    logger.info(f"Speedup achieved: {speedup:.2f}x")
    logger.info("AMP is the simplest path to faster training -- just wrap forward pass in autocast")
    logger.info("GradScaler is REQUIRED for FP16 to handle gradient underflow")
    logger.info("GradScaler is NOT needed for BF16 (covered in bf16_vs_fp16.py tutorial)")
    logger.info("Memory savings come from storing activations in FP16 during forward pass")
    logger.info("Master weights stay in FP32 -- no accuracy loss from reduced precision weights")

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
