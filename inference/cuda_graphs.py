"""
CUDA Graphs: Eliminating Kernel Launch Overhead for Inference
==============================================================

Every time PyTorch executes an operation on GPU, it launches a CUDA kernel.
Each kernel launch has a fixed overhead (~5-10 microseconds) for:
  - CPU-side dispatch and argument marshaling
  - CUDA driver API calls
  - Kernel scheduling on the GPU

For large models with expensive ops, this overhead is negligible. But for
small models or inference with small batch sizes, the cumulative launch
overhead can dominate total runtime -- the GPU spends more time WAITING
for kernels than RUNNING them.

CUDA Graphs solve this by recording a sequence of GPU operations once,
then replaying the entire sequence with a single CPU-side launch. The
graph captures:
  - Kernel launches, their arguments, and execution order
  - Memory allocations and data movement
  - Synchronization points

The result: one launch replays hundreds of kernels, cutting CPU overhead
to near zero.

When to use:
  - Inference with small batch sizes where launch overhead dominates
  - Latency-sensitive serving (real-time, interactive)
  - Repetitive computations with fixed tensor shapes

When NOT to use:
  - Dynamic shapes (CUDA graphs require fixed tensor sizes)
  - Training (optimizer state changes are hard to capture)
  - Models with data-dependent control flow

What this tutorial demonstrates:
  1. Baseline eager inference (standard kernel launches)
  2. CUDA graph-captured inference (single replay)
  3. Latency comparison at various batch sizes
  4. Limitations and shape-change pitfalls
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

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

logger = setup_logging("cuda_graphs")

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
NUM_ITERATIONS = 500
WARMUP_ITERATIONS = 20
BATCH_SIZES = [1, 4, 16, 64]


def capture_cuda_graph(model, sample_input):
    """Capture a model's forward pass as a CUDA graph.

    Records the sequence of GPU operations for a fixed input shape, then
    returns the graph and the static output tensor. Subsequent runs replay
    the graph by copying new data into the static input buffer.

    Args:
        model: Model in eval mode on CUDA.
        sample_input: A tensor with the shape that will be used for all
                      future inference calls. Data doesn't matter -- only shape.

    Returns:
        Tuple of (graph, static_input, static_output) where:
          - graph: The captured torch.cuda.CUDAGraph
          - static_input: Buffer to copy real input data into before replay
          - static_output: Buffer that holds the output after replay
    """
    # Allocate static tensors that the graph will read from / write to.
    # These stay at the same memory address across replays.
    static_input = sample_input.clone()
    static_output = None

    # Warm up: ensure all lazy CUDA initializations are done
    with torch.inference_mode():
        for _ in range(3):
            _ = model(static_input)
    torch.cuda.synchronize()

    # Capture the graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = model(static_input)

    return graph, static_input, static_output


def run_eager_inference(model, inputs, num_iterations):
    """Run standard eager inference for timing.

    Args:
        model: Model in eval mode.
        inputs: Input tensor.
        num_iterations: Number of forward passes.
    """
    with torch.inference_mode():
        for _ in range(num_iterations):
            _ = model(inputs)


def run_graph_inference(graph, static_input, real_input, num_iterations):
    """Run CUDA graph replay inference for timing.

    Copies real input data into the static buffer, then replays the
    captured graph. The static_output tensor is updated in-place.

    Args:
        graph: Captured CUDAGraph.
        static_input: Static input buffer (same address used during capture).
        real_input: Actual input data to process.
        num_iterations: Number of graph replays.
    """
    with torch.inference_mode():
        for _ in range(num_iterations):
            static_input.copy_(real_input)
            graph.replay()


def main():
    # ==============================================================
    # SETUP
    # ==============================================================
    print("\n" + "=" * 60)
    print("  CUDA Graphs: Eliminating Kernel Launch Overhead")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    if device.type != "cuda":
        logger.error(
            "CUDA Graphs require a CUDA GPU. This tutorial cannot run on CPU."
        )
        logger.info("Please run on a machine with an NVIDIA GPU.")
        return

    model = SimpleCNN().to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: SimpleCNN with {total_params:,} parameters")
    logger.info(f"Inference iterations per benchmark: {NUM_ITERATIONS}")
    logger.info(f"Batch sizes to test: {BATCH_SIZES}")

    # ==============================================================
    # SECTION 1: Eager vs CUDA Graph at Default Batch Size
    # ==============================================================
    print("\n--- Section 1: Eager vs CUDA Graph (batch_size=1) ---\n")
    logger.info(
        "Batch size 1 is where kernel launch overhead hurts the most. "
        "Each kernel does very little work, so the CPU dispatch time "
        "becomes a significant fraction of total latency."
    )

    inputs_1, _ = get_sample_batch(batch_size=1, device=device)

    # Warmup eager
    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            _ = model(inputs_1)
    torch.cuda.synchronize()

    @benchmark
    def eager_bs1():
        run_eager_inference(model, inputs_1, NUM_ITERATIONS)
        torch.cuda.synchronize()

    eager_result = eager_bs1()
    eager_per_iter = eager_result["time_seconds"] / NUM_ITERATIONS
    logger.info(
        f"Eager (bs=1): {eager_result['time_seconds']:.4f}s total, "
        f"{eager_per_iter * 1000:.3f}ms per inference"
    )

    # Capture and run CUDA graph
    logger.info("Capturing CUDA graph...")
    graph, static_input, static_output = capture_cuda_graph(model, inputs_1)
    logger.info(f"Graph captured. Output shape: {static_output.shape}")

    # Warmup graph replay
    for _ in range(WARMUP_ITERATIONS):
        static_input.copy_(inputs_1)
        graph.replay()
    torch.cuda.synchronize()

    @benchmark
    def graph_bs1():
        run_graph_inference(graph, static_input, inputs_1, NUM_ITERATIONS)
        torch.cuda.synchronize()

    graph_result = graph_bs1()
    graph_per_iter = graph_result["time_seconds"] / NUM_ITERATIONS
    logger.info(
        f"CUDA Graph (bs=1): {graph_result['time_seconds']:.4f}s total, "
        f"{graph_per_iter * 1000:.3f}ms per inference"
    )

    compare_results(eager_result, graph_result, "CUDA Graphs (batch_size=1)")

    # Verify correctness
    with torch.inference_mode():
        eager_out = model(inputs_1)
    static_input.copy_(inputs_1)
    graph.replay()
    torch.cuda.synchronize()
    max_diff = (eager_out - static_output).abs().max().item()
    logger.info(f"Max difference between eager and graph output: {max_diff:.2e}")
    if max_diff < 1e-5:
        logger.info("Outputs match -- CUDA graph produces identical results.")
    else:
        logger.warning("Outputs differ! Check for non-deterministic operations.")

    # ==============================================================
    # SECTION 2: Batch Size Scaling Comparison
    # ==============================================================
    print("\n--- Section 2: Batch Size Scaling ---\n")
    logger.info(
        "As batch size increases, the actual compute per kernel grows, "
        "and launch overhead becomes a smaller fraction of total time. "
        "CUDA graphs help less at large batch sizes."
    )

    all_results = []

    for bs in BATCH_SIZES:
        logger.info(f"\n  batch_size={bs}:")
        inputs_bs, _ = get_sample_batch(batch_size=bs, device=device)

        # Eager
        with torch.inference_mode():
            for _ in range(WARMUP_ITERATIONS):
                _ = model(inputs_bs)
        torch.cuda.synchronize()

        @benchmark
        def eager_run(inp=inputs_bs):
            run_eager_inference(model, inp, NUM_ITERATIONS)
            torch.cuda.synchronize()

        e_result = eager_run()

        # CUDA Graph (need new graph for different batch size)
        g, s_in, s_out = capture_cuda_graph(model, inputs_bs)
        for _ in range(WARMUP_ITERATIONS):
            s_in.copy_(inputs_bs)
            g.replay()
        torch.cuda.synchronize()

        @benchmark
        def graph_run(gr=g, si=s_in, inp=inputs_bs):
            run_graph_inference(gr, si, inp, NUM_ITERATIONS)
            torch.cuda.synchronize()

        g_result = graph_run()

        e_per = e_result["time_seconds"] / NUM_ITERATIONS * 1000
        g_per = g_result["time_seconds"] / NUM_ITERATIONS * 1000
        speedup = e_result["time_seconds"] / g_result["time_seconds"] if g_result["time_seconds"] > 0 else float("inf")

        logger.info(f"    Eager:      {e_per:.3f}ms per inference")
        logger.info(f"    CUDA Graph: {g_per:.3f}ms per inference")
        logger.info(f"    Speedup:    {speedup:.2f}x")

        all_results.append({
            "name": f"Eager (bs={bs})",
            "time_seconds": e_result["time_seconds"],
            "memory_mb": e_result.get("memory_mb"),
        })
        all_results.append({
            "name": f"CUDA Graph (bs={bs})",
            "time_seconds": g_result["time_seconds"],
            "memory_mb": g_result.get("memory_mb"),
        })

    print("\n--- Benchmark Results ---\n")
    print_benchmark_table(all_results)

    # ==============================================================
    # SECTION 3: Limitations and Pitfalls
    # ==============================================================
    print("\n--- Section 3: Limitations and Pitfalls ---\n")

    logger.info("FIXED SHAPES REQUIREMENT:")
    logger.info(
        "  CUDA graphs record operations for a specific tensor shape. "
        "If the input shape changes, you need a new graph. This means "
        "you cannot use a single graph for variable-length sequences "
        "or dynamic batch sizes."
    )

    logger.info("\nDemonstrating shape mismatch detection:")
    try:
        # Try to use the batch_size=1 graph with batch_size=4 input
        wrong_input, _ = get_sample_batch(batch_size=4, device=device)
        static_input.copy_(wrong_input)  # static_input is shape (1, 3, 32, 32)
        logger.info("  Shape mismatch would cause incorrect results or crash!")
    except RuntimeError as e:
        logger.info(f"  Caught expected error: {e}")

    logger.info("\nOTHER LIMITATIONS:")
    logger.info("  - No CPU operations inside the captured region")
    logger.info("  - No dynamic control flow (if/else based on tensor values)")
    logger.info("  - No operations that change tensor shapes")
    logger.info("  - Memory allocations during capture become static")
    logger.info("  - Not suitable for training (optimizer updates are tricky)")

    logger.info("\nWORKAROUNDS:")
    logger.info(
        "  - For variable batch sizes: capture one graph per batch size, "
        "or pad inputs to the maximum size"
    )
    logger.info(
        "  - For variable sequence lengths: pad to a fixed maximum length "
        "and use attention masks"
    )
    logger.info(
        "  - torch.compile(mode='reduce-overhead') uses CUDA graphs "
        "internally -- consider using that instead of manual capture"
    )

    # ==============================================================
    # Key Takeaways
    # ==============================================================
    print("\n--- Key Takeaways ---\n")
    logger.info(
        "1. CUDA graphs eliminate kernel launch overhead by replaying a "
        "recorded sequence of GPU operations with a single CPU launch."
    )
    logger.info(
        "2. The speedup is largest for small batch sizes and lightweight "
        "models where launch overhead dominates compute time."
    )
    logger.info(
        "3. All tensor shapes must be fixed at capture time -- dynamic "
        "shapes require separate graphs or padding strategies."
    )
    logger.info(
        "4. For most use cases, torch.compile(mode='reduce-overhead') "
        "provides similar benefits with less manual effort. Use manual "
        "CUDA graphs when you need maximum control."
    )
    logger.info(
        "5. Always verify correctness: compare graph output against eager "
        "output to catch non-determinism or capture errors."
    )

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
