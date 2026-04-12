"""
PyTorch Profiler: CPU/GPU Bottleneck Analysis & Chrome Trace Export
====================================================================

torch.profiler is PyTorch's built-in profiling tool that captures detailed
timing information for every operation (convolutions, matrix multiplications,
memory copies, etc.) executed on CPU and GPU. It answers the question: "Where
is my training step actually spending time?"

When to use:
  - Identifying which operators dominate GPU time (e.g., convolutions vs. data copies)
  - Finding CPU-bound bottlenecks in a GPU training pipeline
  - Understanding memory allocation patterns during training
  - Generating Chrome trace files for visual timeline analysis

What this tutorial produces:
  1. Console tables of top operations sorted by GPU time and CPU memory
  2. Console table grouped by input shape to spot dimension-specific costs
  3. A Chrome trace JSON file (profiler_output/training_trace.json) for visual inspection
  4. Benchmark comparison of training vs. inference performance
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule, record_function

from utils import (
    setup_logging,
    benchmark,
    print_benchmark_table,
    SimpleCNN,
    get_sample_batch,
    get_device,
    print_device_info,
)

logger = setup_logging("pytorch_profiler")

BATCH_SIZE = 64
NUM_ITERATIONS = 10  # total steps for profiler schedule
PROFILER_WAIT = 1
PROFILER_WARMUP = 1
PROFILER_ACTIVE = 3
PROFILER_REPEAT = 1


def main():
    # ================================================================
    # Section 1: Setup
    # ================================================================
    print("\n" + "=" * 60)
    print("  PyTorch Profiler: CPU/GPU Bottleneck Analysis")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    logger.info("Preparing model and sample data...")
    model = SimpleCNN().to(device)
    x, labels = get_sample_batch(batch_size=BATCH_SIZE, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: SimpleCNN with {total_params:,} parameters")
    logger.info(f"Input: batch_size={BATCH_SIZE}, shape={tuple(x.shape)}")
    logger.info(f"Device: {device}")

    # ================================================================
    # Section 2: Profile a Training Step
    # ================================================================
    print("\n--- Section 2: Profile a Training Step ---\n")

    logger.info("Setting up PyTorch Profiler with schedule:")
    logger.info(f"  wait={PROFILER_WAIT}, warmup={PROFILER_WARMUP}, "
                f"active={PROFILER_ACTIVE}, repeat={PROFILER_REPEAT}")
    logger.info("The profiler skips 'wait' steps, warms up for 'warmup' steps,")
    logger.info("then actively records for 'active' steps. This avoids capturing")
    logger.info("cold-start overhead in the profiled data.")

    # Create output directory for trace files, anchored to script location
    # so the output lands in the same place regardless of the working directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "profiler_output")
    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, "training_trace.json")

    total_steps = (PROFILER_WAIT + PROFILER_WARMUP + PROFILER_ACTIVE) * PROFILER_REPEAT

    logger.info(f"Running {total_steps} profiled training steps...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=PROFILER_WAIT,
            warmup=PROFILER_WARMUP,
            active=PROFILER_ACTIVE,
            repeat=PROFILER_REPEAT,
        ),
        on_trace_ready=lambda p: p.export_chrome_trace(trace_path),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(total_steps):
            with record_function("training_step"):
                # Forward pass
                output = model(x)
                loss = criterion(output, labels)

                # Backward pass
                loss.backward()

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

            prof.step()
            logger.info(f"  Step {step + 1}/{total_steps} complete")

    logger.info("Profiling complete!")

    # ================================================================
    # Section 3: Analyze Console Output
    # ================================================================
    print("\n--- Section 3: Profiler Analysis Tables ---\n")

    # Table 1: Top operations by GPU time
    logger.info("Top Operations by GPU Time")
    logger.info("This table shows which operators consume the most CUDA time.")
    logger.info("Look for convolution, matmul, and memory copy operations.")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    # Table 2: Top operations by CPU memory
    print()
    logger.info("Top Operations by CPU Memory")
    logger.info("This table reveals which operations allocate the most CPU memory.")
    logger.info("High CPU memory usage may indicate unnecessary host-side copies.")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    # Table 3: Top operations grouped by input shape
    print()
    logger.info("Top Operations by GPU Time (grouped by input shape)")
    logger.info("Grouping by input shape helps identify which tensor dimensions")
    logger.info("are causing the most compute cost -- useful for batch/resolution tuning.")
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cuda_time_total", row_limit=10
    ))

    # ================================================================
    # Section 4: Profile Inference vs Training
    # ================================================================
    print("\n--- Section 4: Inference vs Training Benchmark ---\n")

    logger.info("Comparing training (with gradients) vs inference (no gradients)...")
    logger.info("Training requires storing activations for backward pass,")
    logger.info("while inference_mode() disables autograd entirely.")

    num_bench_steps = 10

    @benchmark
    def training_loop():
        model.train()
        for _ in range(num_bench_steps):
            output = model(x)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()

    @benchmark
    def inference_loop():
        model.eval()
        with torch.inference_mode():
            for _ in range(num_bench_steps):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Warm up
    with torch.inference_mode():
        for _ in range(5):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    train_result = training_loop()
    infer_result = inference_loop()

    results = [
        {
            "name": "Training (with grad)",
            "time_seconds": train_result["time_seconds"],
            "memory_mb": train_result["memory_mb"],
        },
        {
            "name": "Inference (no grad)",
            "time_seconds": infer_result["time_seconds"],
            "memory_mb": infer_result["memory_mb"],
        },
    ]
    print_benchmark_table(results)

    speedup = train_result["time_seconds"] / infer_result["time_seconds"] if infer_result["time_seconds"] > 0 else float("inf")
    logger.info(f"Inference is {speedup:.2f}x faster than training")
    if train_result["memory_mb"] and infer_result["memory_mb"]:
        mem_savings = train_result["memory_mb"] - infer_result["memory_mb"]
        logger.info(f"Memory savings in inference: {mem_savings:.1f} MB "
                     f"(no activation storage for backward pass)")

    # ================================================================
    # Section 5: Chrome Trace Instructions
    # ================================================================
    print("\n--- Section 5: Viewing the Chrome Trace ---\n")

    abs_trace_path = os.path.abspath(trace_path)
    logger.info("A Chrome trace file has been exported for visual analysis.")
    logger.info(f"Trace file: {abs_trace_path}")
    logger.info("")
    logger.info("To view the trace:")
    logger.info("  1. Open Chrome browser and navigate to chrome://tracing")
    logger.info("  2. Click 'Load' and select profiler_output/training_trace.json")
    logger.info("  3. Use W/S to zoom, A/D to pan, and click events for details")
    logger.info("")
    logger.info("The trace shows a timeline of CPU and GPU operations, revealing:")
    logger.info("  - Kernel launch latency (gap between CPU dispatch and GPU execution)")
    logger.info("  - GPU idle time (bubbles where GPU is waiting for CPU)")
    logger.info("  - Overlapping operations (data transfer while compute runs)")

    # ================================================================
    # Section 6: Key Takeaways
    # ================================================================
    print("\n--- Section 6: Key Takeaways ---\n")

    logger.info("1. PROFILER SCHEDULE PHASES:")
    logger.info("   - wait: Skip initial steps (cold caches, JIT compilation)")
    logger.info("   - warmup: Run steps but discard data (GPU frequency scaling)")
    logger.info("   - active: Record detailed timing data for analysis")
    logger.info("   - repeat: How many wait+warmup+active cycles to run")
    logger.info("")
    logger.info("2. WHICH sort_by TO USE:")
    logger.info("   - 'cuda_time_total': Find GPU bottlenecks (most common)")
    logger.info("   - 'cpu_time_total': Find CPU-bound operations")
    logger.info("   - 'self_cpu_memory_usage': Find memory-heavy CPU operations")
    logger.info("   - 'self_cuda_memory_usage': Find GPU memory allocators")
    logger.info("")
    logger.info("3. INFERENCE vs TRAINING MEMORY:")
    logger.info("   - Training stores activations for backward pass (more memory)")
    logger.info("   - inference_mode() disables autograd entirely (fastest + least memory)")
    logger.info("   - torch.no_grad() prevents gradient accumulation but still tracks tensors")
    logger.info("")
    logger.info("4. CHROME TRACE vs CONSOLE TABLES:")
    logger.info("   - Console tables: Quick summary, sort by different metrics, scriptable")
    logger.info("   - Chrome trace: Visual timeline, see overlaps and gaps, interactive")
    logger.info("   - Use tables first for quick diagnosis, traces for deep investigation")

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
