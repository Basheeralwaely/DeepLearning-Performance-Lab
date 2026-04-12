"""
GPU Memory Profiling: Memory Lifecycle, OOM Debugging & Gradient Checkpointing
================================================================================

GPU memory during training is consumed by four main components:
  1. Model parameters: Weights and biases stored on GPU
  2. Activations: Intermediate tensors saved for backward pass (largest consumer)
  3. Gradients: Same size as parameters, computed during backward
  4. Optimizer state: Momentum buffers, Adam statistics (1-2x parameter size)

When to use memory profiling:
  - Debugging Out-of-Memory (OOM) errors -- find what's consuming memory
  - Optimizing batch size -- understand memory scaling to maximize GPU utilization
  - Detecting memory leaks -- track allocations across training steps
  - Evaluating memory-saving techniques -- quantify impact of checkpointing, mixed precision

What this tutorial demonstrates:
  1. Memory allocated at each stage of a training step (model, activations, gradients)
  2. How memory scales with batch size (with OOM detection)
  3. Graceful OOM recovery pattern (catch, clear cache, retry with smaller batch)
  4. Gradient checkpointing: trading compute for memory by recomputing activations
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

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

logger = setup_logging("memory_profiling")

SMALL_BATCH = 32
SMALL_INPUT_SIZE = 32
LARGE_INPUT_SIZE = 224
NUM_CLASSES = 10


def log_memory_state(label):
    """Log current GPU memory allocation, reservation, and peak usage.

    Args:
        label: A descriptive string identifying the measurement point.
    """
    if not torch.cuda.is_available():
        logger.info(f"[{label}] CUDA not available -- skipping memory stats")
        return

    alloc = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    logger.info(
        f"[{label}] Allocated: {alloc:.1f} MB | Reserved: {reserved:.1f} MB | Peak: {peak:.1f} MB"
    )


def main():
    # ================================================================
    # Section 1: Setup
    # ================================================================
    print("\n" + "=" * 60)
    print("  GPU Memory Profiling: Lifecycle, OOM & Checkpointing")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    if device.type != "cuda":
        logger.warning("This tutorial requires a CUDA GPU for memory profiling.")
        logger.warning("Running on CPU -- memory tracking will be limited.")
        print("\n" + "=" * 60)
        print("  Tutorial requires CUDA GPU -- exiting")
        print("=" * 60 + "\n")
        return

    total_gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**2
    logger.info(f"Total GPU memory: {total_gpu_mem:.0f} MB")

    # ================================================================
    # Section 2: Memory Lifecycle of a Training Step
    # ================================================================
    print("\n--- Section 2: Memory Lifecycle of a Training Step ---\n")

    logger.info("Tracking GPU memory at each stage of a single training step.")
    logger.info(f"Using SimpleCNN with input_size={SMALL_INPUT_SIZE}, batch_size={SMALL_BATCH}")
    logger.info("This reveals how memory grows: model -> inputs -> activations -> gradients -> optimizer state\n")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    log_memory_state("Before model creation")

    model = SimpleCNN(input_size=SMALL_INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
    log_memory_state("After model.to(device)")

    x, labels = get_sample_batch(
        batch_size=SMALL_BATCH,
        height=SMALL_INPUT_SIZE,
        width=SMALL_INPUT_SIZE,
        num_classes=NUM_CLASSES,
        device=device,
    )
    log_memory_state("After input allocation")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Forward pass -- activations stored for backward
    output = model(x)
    loss = criterion(output, labels)
    log_memory_state("After forward pass (activations stored)")

    # Backward pass -- gradients computed
    loss.backward()
    log_memory_state("After backward pass (gradients computed)")

    # Optimizer step -- update parameters using gradients + momentum buffers
    optimizer.step()
    log_memory_state("After optimizer.step()")

    # Zero gradients -- free gradient tensors
    optimizer.zero_grad()
    log_memory_state("After zero_grad()")

    logger.info("\nDetailed memory summary from PyTorch:")
    print(torch.cuda.memory_summary(abbreviated=True))

    # Cleanup -- synchronize before deleting to ensure all GPU ops are complete
    torch.cuda.synchronize()
    del model, x, labels, output, loss, optimizer, criterion
    torch.cuda.empty_cache()

    # ================================================================
    # Section 3: Memory Scaling with Batch Size
    # ================================================================
    print("\n--- Section 3: Memory Scaling with Batch Size ---\n")

    logger.info(f"Sweeping batch sizes with SimpleCNN(input_size={LARGE_INPUT_SIZE})")
    logger.info("Larger inputs (224x224) amplify memory differences between batch sizes.")
    logger.info("Watch how peak memory grows roughly linearly with batch size.\n")

    # Warm up CUBLAS with a small operation to ensure handles are initialized
    _warmup = torch.randn(2, 2, device=device) @ torch.randn(2, 2, device=device)
    del _warmup
    torch.cuda.synchronize()

    batch_sizes = [8, 16, 32, 64, 128]
    results = []

    for bs in batch_sizes:
        # Clean slate for each batch size
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        m = None
        try:
            m = SimpleCNN(input_size=LARGE_INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
            opt = torch.optim.SGD(m.parameters(), lr=0.01)
            crit = nn.CrossEntropyLoss()

            inp, lbl = get_sample_batch(
                batch_size=bs,
                height=LARGE_INPUT_SIZE,
                width=LARGE_INPUT_SIZE,
                num_classes=NUM_CLASSES,
                device=device,
            )

            start_time = time.perf_counter()

            out = m(inp)
            l = crit(out, lbl)
            l.backward()
            opt.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2

            results.append({
                "name": f"batch={bs}",
                "time_seconds": elapsed,
                "memory_mb": peak_mem,
            })
            logger.info(f"batch_size={bs}: peak={peak_mem:.1f} MB, time={elapsed:.4f}s")

            # Cleanup
            del m, opt, crit, inp, lbl, out, l
            m = None
            torch.cuda.empty_cache()

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"OOM/CUDA error at batch_size={bs}: {type(e).__name__}")
            logger.warning("Clearing cache and stopping batch size sweep.")
            # Ensure all tensors are freed before clearing cache
            del m
            torch.cuda.empty_cache()
            break

    if results:
        logger.info("\nBatch size scaling results:")
        print_benchmark_table(results)

        if len(results) >= 2:
            mem_first = results[0]["memory_mb"]
            mem_last = results[-1]["memory_mb"]
            bs_first = batch_sizes[0]
            bs_last = batch_sizes[len(results) - 1]
            logger.info(
                f"Memory grew from {mem_first:.0f} MB (batch={bs_first}) "
                f"to {mem_last:.0f} MB (batch={bs_last}) -- "
                f"{mem_last / mem_first:.1f}x increase for "
                f"{bs_last / bs_first:.1f}x batch size increase"
            )

    # ================================================================
    # Section 4: Catching OOM Gracefully
    # ================================================================
    print("\n--- Section 4: Catching OOM Gracefully ---\n")

    logger.info("Demonstrating the OOM recovery pattern:")
    logger.info("  1. Try a large batch size that may exceed GPU memory")
    logger.info("  2. Catch torch.cuda.OutOfMemoryError")
    logger.info("  3. Clear the CUDA cache to free fragmented memory")
    logger.info("  4. Retry with a smaller batch size\n")

    torch.cuda.empty_cache()
    oom_batch_size = 256

    logger.info(f"Attempting batch_size={oom_batch_size} with input_size={LARGE_INPUT_SIZE}...")

    try:
        m = SimpleCNN(input_size=LARGE_INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        inp, lbl = get_sample_batch(
            batch_size=oom_batch_size,
            height=LARGE_INPUT_SIZE,
            width=LARGE_INPUT_SIZE,
            num_classes=NUM_CLASSES,
            device=device,
        )
        out = m(inp)
        l = crit(out, lbl)
        l.backward()
        logger.info(f"Success with batch_size={oom_batch_size} -- no OOM occurred.")
        logger.info("(Your GPU has enough memory for this batch size.)")
        del m, opt, crit, inp, lbl, out, l

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        logger.warning(f"OOM with batch_size={oom_batch_size}! ({type(e).__name__})")
        logger.info("Clearing CUDA cache to recover memory...")
        torch.cuda.empty_cache()
        log_memory_state("After OOM recovery")

        # Retry with half the batch size
        reduced_batch = oom_batch_size // 2
        logger.info(f"Retrying with batch_size={reduced_batch}...")

        try:
            m = SimpleCNN(input_size=LARGE_INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
            opt = torch.optim.SGD(m.parameters(), lr=0.01)
            crit = nn.CrossEntropyLoss()
            inp, lbl = get_sample_batch(
                batch_size=reduced_batch,
                height=LARGE_INPUT_SIZE,
                width=LARGE_INPUT_SIZE,
                num_classes=NUM_CLASSES,
                device=device,
            )
            out = m(inp)
            l = crit(out, lbl)
            l.backward()
            logger.info(f"Success with reduced batch_size={reduced_batch}!")
            del m, opt, crit, inp, lbl, out, l
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e2:
            logger.warning(f"Still OOM with batch_size={reduced_batch}. ({type(e2).__name__})")
            logger.warning("Consider using a smaller model, mixed precision, or gradient checkpointing.")

    torch.cuda.empty_cache()

    # ================================================================
    # Section 5: Gradient Checkpointing Memory Impact
    # ================================================================
    print("\n--- Section 5: Gradient Checkpointing Memory Impact ---\n")

    logger.info("Gradient checkpointing trades compute for memory:")
    logger.info("  - Without checkpointing: all intermediate activations are stored")
    logger.info("  - With checkpointing: activations are recomputed during backward pass")
    logger.info("  - Result: lower peak memory at the cost of ~20-30% more compute time\n")

    ckpt_batch_size = 32
    logger.info(f"Comparing with SimpleCNN(input_size={LARGE_INPUT_SIZE}), batch_size={ckpt_batch_size}")

    try:
        # --- Without checkpointing ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        m = SimpleCNN(input_size=LARGE_INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()

        inp, lbl = get_sample_batch(
            batch_size=ckpt_batch_size,
            height=LARGE_INPUT_SIZE,
            width=LARGE_INPUT_SIZE,
            num_classes=NUM_CLASSES,
            device=device,
        )

        @benchmark
        def run_without_checkpoint():
            out = m(inp)
            loss = crit(out, lbl)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if device.type == "cuda":
                torch.cuda.synchronize()

        baseline_result = run_without_checkpoint()
        peak_no_ckpt = torch.cuda.max_memory_allocated() / 1024**2
        logger.info(f"Without checkpointing: peak memory = {peak_no_ckpt:.1f} MB")

        del m, opt, crit
        torch.cuda.empty_cache()

        # --- With checkpointing ---
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        m = SimpleCNN(input_size=LARGE_INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()

        @benchmark
        def run_with_checkpoint():
            # Checkpoint the feature extractor -- recompute activations during backward
            features_out = checkpoint(m.features, inp, use_reentrant=False)
            features_out = features_out.view(features_out.size(0), -1)
            out = m.classifier(features_out)
            loss = crit(out, lbl)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if device.type == "cuda":
                torch.cuda.synchronize()

        ckpt_result = run_with_checkpoint()
        peak_with_ckpt = torch.cuda.max_memory_allocated() / 1024**2
        logger.info(f"With checkpointing: peak memory = {peak_with_ckpt:.1f} MB")

        # Show comparison
        logger.info("\nMemory comparison:")
        # Override memory_mb with our precise measurements
        baseline_result["memory_mb"] = peak_no_ckpt
        ckpt_result["memory_mb"] = peak_with_ckpt

        compare_results(baseline_result, ckpt_result, "Gradient Checkpointing")

        mem_saved = peak_no_ckpt - peak_with_ckpt
        mem_pct = (mem_saved / peak_no_ckpt) * 100 if peak_no_ckpt > 0 else 0
        logger.info(f"Memory saved: {mem_saved:.1f} MB ({mem_pct:.1f}% reduction)")
        logger.info("Checkpointing recomputes feature activations during backward pass,")
        logger.info("so training is slightly slower but uses significantly less memory.")
        logger.info("This enables larger batch sizes or bigger models on the same GPU.")

        # Cleanup
        del m, opt, crit, inp, lbl
        torch.cuda.empty_cache()

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        logger.warning(f"Could not run checkpointing comparison: {type(e).__name__}")
        logger.warning("GPU memory insufficient for this demonstration.")
        logger.info("On a larger GPU, this section shows memory savings from checkpointing.")
        torch.cuda.empty_cache()

    # ================================================================
    # Section 6: Key Takeaways
    # ================================================================
    print("\n--- Section 6: Key Takeaways ---\n")

    logger.info("1. MEMORY STAGES DURING TRAINING:")
    logger.info("   - Model parameters: fixed cost, proportional to parameter count")
    logger.info("   - Activations: proportional to batch_size * model_depth (largest consumer)")
    logger.info("   - Gradients: same size as parameters")
    logger.info("   - Optimizer state: 1x params (SGD+momentum) to 2x params (Adam)")
    logger.info("")
    logger.info("2. BATCH SIZE IS THE PRIMARY MEMORY KNOB:")
    logger.info("   - Memory scales roughly linearly with batch size")
    logger.info("   - Halving batch size roughly halves activation memory")
    logger.info("   - Use binary search to find max batch size for your GPU")
    logger.info("")
    logger.info("3. OOM RECOVERY PATTERN:")
    logger.info("   - Always wrap large allocations in try/except torch.cuda.OutOfMemoryError")
    logger.info("   - Call torch.cuda.empty_cache() immediately after OOM")
    logger.info("   - Retry with reduced batch size (halve it)")
    logger.info("   - Log the failure for debugging")
    logger.info("")
    logger.info("4. GRADIENT CHECKPOINTING TRADEOFF:")
    logger.info("   - Saves memory by recomputing activations during backward pass")
    logger.info("   - Costs ~20-30% more compute time (one extra forward through checkpointed layers)")
    logger.info("   - Best for models with deep feature extractors (ResNet, Transformers)")
    logger.info("   - Use torch.utils.checkpoint.checkpoint() with use_reentrant=False")
    logger.info("")
    logger.info("5. WHEN TO USE WHICH TOOL:")
    logger.info("   - torch.cuda.memory_summary(): Detailed breakdown in a single snapshot")
    logger.info("   - Manual log_memory_state(): Track memory at specific code points")
    logger.info("   - torch.cuda.memory_allocated(): Quick check of current usage")
    logger.info("   - torch.cuda.max_memory_allocated(): Peak usage since last reset")

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
