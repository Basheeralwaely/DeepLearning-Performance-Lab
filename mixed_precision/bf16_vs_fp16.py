"""
BF16 vs FP16: Precision Format Comparison
==========================================

BFloat16 (BF16) and Float16 (FP16) are both 16-bit formats but with different
tradeoffs. FP16 has 10 mantissa bits (more precision) but only 5 exponent bits
(limited range, max ~65504). BF16 has 7 mantissa bits (less precision) but 8
exponent bits (same range as FP32, max ~3.4e38).

Key practical difference: BF16 does NOT need GradScaler because its wide
dynamic range prevents gradient underflow. FP16 REQUIRES GradScaler.

When to use:
  - BF16: Preferred on Ampere+ GPUs (A100, H100) -- simpler code, no scaler needed
  - FP16: Required on older GPUs (Volta, Turing) or when extra mantissa precision matters

What this tutorial demonstrates:
  1. Numerical stability differences (overflow/underflow behavior)
  2. FP16 training with GradScaler
  3. BF16 training without GradScaler
  4. Side-by-side benchmark comparison
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
    print_benchmark_table,
    SimpleCNN,
    get_sample_batch,
    get_device,
    print_device_info,
)

logger = setup_logging("bf16_vs_fp16")

NUM_ITERATIONS = 100
BATCH_SIZE = 64
WARMUP_ITERATIONS = 10


def main():
    # ================================================================
    # Section 0: Setup
    # ================================================================
    print("\n" + "=" * 60)
    print("  BF16 vs FP16: Precision Format Comparison")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    if device.type != "cuda":
        logger.warning("Mixed precision requires a CUDA GPU for meaningful benchmarks.")
        logger.warning("Results will still run but performance differences will be minimal.")

    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Training iterations per benchmark: {NUM_ITERATIONS}")
    logger.info(f"Warmup iterations: {WARMUP_ITERATIONS}")

    inputs, labels = get_sample_batch(batch_size=BATCH_SIZE, device=device)
    criterion = nn.CrossEntropyLoss()

    # ================================================================
    # Section 1: Numerical Stability Differences
    # ================================================================
    print("\n--- Section 1: Numerical Stability Differences ---\n")

    logger.info("Comparing how FP16 and BF16 handle extreme values:")
    logger.info("")

    # Overflow test
    large_val = torch.tensor(70000.0)
    fp16_val = large_val.to(torch.float16)
    bf16_val = large_val.to(torch.bfloat16)
    logger.info(f"Value 70000.0 -> FP16: {fp16_val.item()} (OVERFLOW! max FP16 = 65504)")
    logger.info(f"Value 70000.0 -> BF16: {bf16_val.item()} (OK -- BF16 has FP32 range)")
    logger.info("")

    # Underflow test
    small_val = torch.tensor(1e-8)
    fp16_val = small_val.to(torch.float16)
    bf16_val = small_val.to(torch.bfloat16)
    logger.info(f"Value 1e-8 -> FP16: {fp16_val.item()} (UNDERFLOW! below FP16 min normal)")
    logger.info(f"Value 1e-8 -> BF16: {bf16_val.item()} (OK -- BF16 min ~1e-38)")
    logger.info("")

    # Precision test
    precise_val = torch.tensor(1.0009765625)  # Needs >7 mantissa bits
    fp16_val = precise_val.to(torch.float16)
    bf16_val = precise_val.to(torch.bfloat16)
    logger.info(f"Value 1.0009765625 -> FP16: {fp16_val.item()} (exact -- 10 mantissa bits)")
    logger.info(f"Value 1.0009765625 -> BF16: {bf16_val.item()} (rounded -- only 7 mantissa bits)")
    logger.info("")

    # Summary table
    logger.info("Format comparison:")
    logger.info("  FP16: 5 exp bits, 10 mantissa bits, max 65504, needs GradScaler")
    logger.info("  BF16: 8 exp bits, 7 mantissa bits, max ~3.4e38, NO GradScaler needed")
    logger.info("")
    logger.info("Takeaway: BF16 trades precision for range -- safer for gradients,")
    logger.info("          FP16 trades range for precision -- needs GradScaler to be safe.")

    # ================================================================
    # Section 2: Warmup
    # ================================================================
    print("\n--- Warmup ---\n")

    logger.info("Running warmup passes to compile CUDA kernels...")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(WARMUP_ITERATIONS):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    logger.info("Warmup complete. CUDA kernels compiled and memory pools initialized.")

    # ================================================================
    # Section 3: FP16 Training with GradScaler
    # ================================================================
    print("\n--- Section 2: FP16 Training (with GradScaler) ---\n")

    logger.info("FP16 REQUIRES GradScaler to prevent gradient underflow")
    logger.info("GradScaler dynamically adjusts loss scale factor")
    logger.info("Without GradScaler, small gradients underflow to zero and training diverges")

    model = SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler('cuda')

    @benchmark
    def fp16_training():
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

    fp16_result = fp16_training()
    logger.info(f"FP16 + GradScaler completed in {fp16_result['time_seconds']:.4f}s")
    fp16_throughput = NUM_ITERATIONS * BATCH_SIZE / fp16_result["time_seconds"]
    logger.info(f"FP16 throughput: {fp16_throughput:.0f} samples/sec")

    # ================================================================
    # Section 4: BF16 Training without GradScaler
    # ================================================================
    print("\n--- Section 3: BF16 Training (no GradScaler) ---\n")

    logger.info("BF16 does NOT need GradScaler -- same dynamic range as FP32")
    logger.info("Simpler code: no scaler.scale(), scaler.step(), scaler.update()")
    logger.info("This is the recommended path on Ampere+ GPUs (A100, H100)")

    model = SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    @benchmark
    def bf16_training():
        for _ in range(NUM_ITERATIONS):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            loss.backward()  # No scaler needed!
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    bf16_result = bf16_training()
    logger.info(f"BF16 (no scaler) completed in {bf16_result['time_seconds']:.4f}s")
    bf16_throughput = NUM_ITERATIONS * BATCH_SIZE / bf16_result["time_seconds"]
    logger.info(f"BF16 throughput: {bf16_throughput:.0f} samples/sec")

    # ================================================================
    # Section 5: Results
    # ================================================================
    print("\n--- Results ---\n")

    print_benchmark_table([
        {
            "name": "FP16 + GradScaler",
            "time_seconds": fp16_result["time_seconds"],
            "memory_mb": fp16_result.get("memory_mb"),
        },
        {
            "name": "BF16 (no scaler)",
            "time_seconds": bf16_result["time_seconds"],
            "memory_mb": bf16_result.get("memory_mb"),
        },
    ])

    logger.info(f"FP16 throughput: {fp16_throughput:.0f} samples/sec")
    logger.info(f"BF16 throughput: {bf16_throughput:.0f} samples/sec")

    if fp16_result["time_seconds"] > 0 and bf16_result["time_seconds"] > 0:
        if bf16_result["time_seconds"] < fp16_result["time_seconds"]:
            ratio = fp16_result["time_seconds"] / bf16_result["time_seconds"]
            logger.info(f"BF16 is {ratio:.2f}x faster than FP16 on this GPU")
        else:
            ratio = bf16_result["time_seconds"] / fp16_result["time_seconds"]
            logger.info(f"FP16 is {ratio:.2f}x faster than BF16 on this GPU")

    # ================================================================
    # Section 6: Key Takeaways
    # ================================================================
    print("\n--- Key Takeaways ---\n")

    logger.info("BF16 is simpler (no GradScaler) and safer (no overflow at typical gradient magnitudes)")
    logger.info("FP16 has higher precision (10 vs 7 mantissa bits) -- matters for some loss functions")
    logger.info("On Ampere+ GPUs (A100, H100), BF16 is the recommended default")
    logger.info("On Turing GPUs (RTX 2070), BF16 runs in software emulation -- FP16 will be faster")

    # Detect and report which is faster on this hardware
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        if capability[0] >= 8:
            logger.info(f"This GPU ({gpu_name}, SM {capability[0]}.{capability[1]}): BF16 has native support -- recommended")
        else:
            logger.info(f"This GPU ({gpu_name}, SM {capability[0]}.{capability[1]}): BF16 is emulated -- FP16 will be faster")
    else:
        logger.info("Running on CPU -- both formats are emulated, no hardware acceleration")

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
