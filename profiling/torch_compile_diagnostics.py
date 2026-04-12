"""
torch.compile Diagnostics: Understanding Compilation Behavior
==============================================================

torch.compile is PyTorch 2.x's primary compilation path, replacing
TorchScript for most use cases. Under the hood it works in two stages:

  1. TorchDynamo (graph capture): Traces Python code into FX graphs by
     intercepting bytecode execution. When it encounters unsupported
     operations, it creates a "graph break" -- splitting the model into
     multiple subgraphs.

  2. TorchInductor (optimization): Compiles each captured graph into
     optimized kernels (CUDA, CPU, or Triton), applying operator fusion,
     memory planning, and hardware-specific optimizations.

This tutorial focuses on DIAGNOSTICS -- understanding compilation overhead,
mode tradeoffs, and graph break detection. The goal is to teach you to
diagnose before optimizing:

  - How much does compilation cost (one-time overhead)?
  - Which compile mode gives the best runtime vs compile-time tradeoff?
  - Does your model have graph breaks, and what causes them?

Production deployment of torch.compile is covered in Phase 5 (Inference
Optimization). Here we focus purely on measurement and understanding.

What this tutorial demonstrates:
  1. Eager (uncompiled) baseline measurement
  2. Compilation overhead per mode (first-pass timing)
  3. Runtime comparison across default, reduce-overhead, max-autotune
  4. Graph break detection with torch._dynamo.explain
  5. Intentional graph breaks from data-dependent control flow
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import torch
import torch.nn as nn
import torch._dynamo as dynamo
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

logger = setup_logging("torch_compile_diagnostics")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
INPUT_SIZE = 32
NUM_WARMUP = 3
NUM_TIMED = 50
COMPILE_MODES = ["default", "reduce-overhead", "max-autotune"]


# ---------------------------------------------------------------------------
# Model with intentional graph breaks (Section 5)
# ---------------------------------------------------------------------------
class ModelWithGraphBreaks(nn.Module):
    """A model that uses data-dependent control flow, causing graph breaks.

    TorchDynamo cannot trace through Python if/else that depends on tensor
    values at runtime, because the graph must be static. This forces a
    graph break -- the model gets split into multiple subgraphs that are
    compiled and called separately, reducing optimization opportunities.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3 * INPUT_SIZE * INPUT_SIZE, 256)
        self.linear2 = nn.Linear(256, 10)
        self.linear3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.linear1(x))
        if x.sum() > 0:  # Data-dependent control flow = graph break
            return self.linear2(x)
        else:
            return self.linear3(x)


def main():
    # ==================================================================
    # Section 1: Setup
    # ==================================================================
    print("\n" + "=" * 60)
    print("  torch.compile Diagnostics: Understanding Compilation")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    model = SimpleCNN(input_size=INPUT_SIZE).to(device)
    model.eval()
    x, _ = get_sample_batch(
        batch_size=BATCH_SIZE,
        height=INPUT_SIZE,
        width=INPUT_SIZE,
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: SimpleCNN with {total_params:,} parameters")
    logger.info(f"Input: batch_size={BATCH_SIZE}, shape={tuple(x.shape)}")
    logger.info(f"Warm-up iterations: {NUM_WARMUP}")
    logger.info(f"Timed iterations: {NUM_TIMED}")
    logger.info(f"Compile modes to test: {COMPILE_MODES}")

    # Warm up CUDA
    logger.info("Running CUDA warm-up...")
    with torch.inference_mode():
        for _ in range(10):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # ==================================================================
    # Section 2: Eager vs Compiled Baseline
    # ==================================================================
    print("\n--- Section 2: Eager (Uncompiled) Baseline ---\n")
    logger.info(
        "Measuring eager mode (no compilation) as the baseline. "
        "This is standard PyTorch execution with Python overhead."
    )

    # Additional warm-up for eager
    with torch.inference_mode():
        for _ in range(NUM_WARMUP):
            _ = model(x)

    @benchmark
    def eager_forward():
        with torch.inference_mode():
            for _ in range(NUM_TIMED):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    eager_result = eager_forward()
    eager_per_iter = eager_result["time_seconds"] / NUM_TIMED
    logger.info(
        f"Eager baseline: {eager_result['time_seconds']:.4f}s total, "
        f"{eager_per_iter * 1000:.2f}ms per iteration"
    )

    # ==================================================================
    # Section 3: Compilation Overhead Measurement
    # ==================================================================
    print("\n--- Section 3: Compile Mode Comparison ---\n")
    logger.info(
        "For each compile mode, we measure two things separately:\n"
        "  1. Compilation time (first forward pass includes JIT compilation)\n"
        "  2. Runtime (subsequent passes, compilation already cached)"
    )

    all_results = [
        {
            "name": "eager",
            "time_seconds": eager_result["time_seconds"],
            "memory_mb": eager_result.get("memory_mb"),
        }
    ]

    for mode in COMPILE_MODES:
        logger.info(f"\n  Compiling with mode='{mode}'...")

        # Reset dynamo cache so each mode starts fresh
        dynamo.reset()

        compiled_model = torch.compile(model, mode=mode)

        # Measure compilation time (first forward pass)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compile_start = time.perf_counter()
        with torch.inference_mode():
            _ = compiled_model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        compile_time = time.perf_counter() - compile_start
        logger.info(f"  Compilation time for {mode}: {compile_time:.2f}s")

        # Additional warm-up after compilation
        with torch.inference_mode():
            for _ in range(NUM_WARMUP):
                _ = compiled_model(x)

        # Measure runtime (post-compilation)
        @benchmark
        def compiled_forward(m=compiled_model):
            with torch.inference_mode():
                for _ in range(NUM_TIMED):
                    _ = m(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        result = compiled_forward()
        per_iter = result["time_seconds"] / NUM_TIMED
        logger.info(
            f"  Runtime for {mode}: {result['time_seconds']:.4f}s total, "
            f"{per_iter * 1000:.2f}ms per iteration"
        )

        all_results.append({
            "name": f"compile({mode})",
            "time_seconds": result["time_seconds"],
            "memory_mb": result.get("memory_mb"),
        })

    # Print comparison table
    print_benchmark_table(all_results)

    # Log speedups
    eager_time = eager_result["time_seconds"]
    for r in all_results[1:]:
        speedup = eager_time / r["time_seconds"] if r["time_seconds"] > 0 else float("inf")
        logger.info(f"  {r['name']} speedup vs eager: {speedup:.2f}x")

    # ==================================================================
    # Section 4: Graph Break Detection
    # ==================================================================
    print("\n--- Section 4: Graph Break Detection (Clean Model) ---\n")
    logger.info(
        "Using torch._dynamo.explain() to analyze the model's traceability. "
        "Graph breaks occur when Dynamo cannot trace through Python code "
        "(e.g., data-dependent control flow, unsupported operations)."
    )

    dynamo.reset()
    explanation = dynamo.explain(model)(x)

    logger.info(f"  Graph count: {explanation.graph_count}")
    logger.info(f"  Graph break count: {explanation.graph_break_count}")

    if hasattr(explanation, "ops_per_graph") and explanation.ops_per_graph:
        logger.info(f"  Ops per graph: {explanation.ops_per_graph}")

    if explanation.graph_break_count > 0 and hasattr(explanation, "break_reasons"):
        logger.info("  Graph break reasons:")
        for reason in explanation.break_reasons:
            reason_text = str(reason.reason).split("\n")[0] if hasattr(reason, "reason") else str(reason).split("\n")[0]
            logger.info(f"    - {reason_text}")
    else:
        logger.info(
            "  Model is fully traceable -- no graph breaks detected."
        )
        logger.info(
            "  This means Dynamo can capture the entire forward pass as a "
            "single graph, enabling maximum optimization."
        )

    # ==================================================================
    # Section 5: Demonstrating Graph Breaks
    # ==================================================================
    print("\n--- Section 5: Demonstrating Graph Breaks ---\n")
    logger.info(
        "Creating a model with intentional graph breaks from "
        "data-dependent control flow (if x.sum() > 0)."
    )

    broken_model = ModelWithGraphBreaks().to(device)
    broken_model.eval()

    dynamo.reset()
    broken_explanation = dynamo.explain(broken_model)(x)

    logger.info(f"  Graph count: {broken_explanation.graph_count}")
    logger.info(f"  Graph break count: {broken_explanation.graph_break_count}")

    if hasattr(broken_explanation, "ops_per_graph") and broken_explanation.ops_per_graph:
        logger.info(f"  Ops per graph: {broken_explanation.ops_per_graph}")

    if broken_explanation.graph_break_count > 0:
        logger.info("  Graph break reasons:")
        if hasattr(broken_explanation, "break_reasons"):
            for reason in broken_explanation.break_reasons:
                reason_text = str(reason.reason).split("\n")[0] if hasattr(reason, "reason") else str(reason).split("\n")[0]
                logger.info(f"    - {reason_text}")
        logger.info(
            "\n  The data-dependent 'if x.sum() > 0' forces Dynamo to split "
            "the graph because the branch cannot be determined at trace time."
        )
    else:
        logger.info(
            "  Note: Newer PyTorch versions may handle simple control flow "
            "without graph breaks via guard specialization."
        )

    # Compare compile speedup: clean model vs graph-broken model
    logger.info("\n  Comparing compile speedup of clean vs graph-broken model:")

    # Measure eager baseline for broken model so we compare apples-to-apples.
    # Using eager_time (from SimpleCNN) for the broken model would conflate
    # architectural differences with the effect of graph breaks.
    with torch.inference_mode():
        for _ in range(NUM_WARMUP):
            _ = broken_model(x)

    @benchmark
    def broken_eager():
        with torch.inference_mode():
            for _ in range(NUM_TIMED):
                _ = broken_model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    broken_eager_result = broken_eager()
    broken_eager_time = broken_eager_result["time_seconds"]

    dynamo.reset()
    compiled_clean = torch.compile(model, mode="default")
    with torch.inference_mode():
        for _ in range(NUM_WARMUP + 1):
            _ = compiled_clean(x)

    @benchmark
    def clean_compiled():
        with torch.inference_mode():
            for _ in range(NUM_TIMED):
                _ = compiled_clean(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    clean_result = clean_compiled()

    dynamo.reset()
    compiled_broken = torch.compile(broken_model, mode="default")
    with torch.inference_mode():
        for _ in range(NUM_WARMUP + 1):
            _ = compiled_broken(x)

    @benchmark
    def broken_compiled():
        with torch.inference_mode():
            for _ in range(NUM_TIMED):
                _ = compiled_broken(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    broken_result = broken_compiled()

    compare_results(clean_result, broken_result, "Graph Breaks Impact")

    clean_speedup = eager_time / clean_result["time_seconds"] if clean_result["time_seconds"] > 0 else float("inf")
    broken_speedup = broken_eager_time / broken_result["time_seconds"] if broken_result["time_seconds"] > 0 else float("inf")
    logger.info(f"  Clean model compile speedup vs eager: {clean_speedup:.2f}x")
    logger.info(f"  Broken model compile speedup vs its own eager: {broken_speedup:.2f}x")
    logger.info(
        "  Graph breaks reduce the compiler's ability to optimize, "
        "resulting in less speedup (or even slowdown)."
    )

    # ==================================================================
    # Section 6: Key Takeaways
    # ==================================================================
    print("\n--- Key Takeaways ---\n")
    logger.info(
        "1. Compilation is a one-time cost: The first forward pass is slow "
        "(JIT compilation), but subsequent calls use cached compiled code. "
        "This cost is amortized over the entire training run."
    )
    logger.info(
        "2. Mode tradeoffs:\n"
        "     - default: Fast compilation, moderate runtime improvement\n"
        "     - reduce-overhead: Less kernel launch overhead (good for small models)\n"
        "     - max-autotune: Best runtime but slowest compilation (benchmarks kernels)"
    )
    logger.info(
        "3. Graph breaks reduce optimization: Each break creates a separate "
        "subgraph, preventing cross-graph optimizations like operator fusion."
    )
    logger.info(
        "4. Always check for graph breaks before deploying compiled models: "
        "Use torch._dynamo.explain(model)(sample_input) to identify breaks "
        "and their causes."
    )
    logger.info(
        "5. torch.compile is diagnostic-first: Understand your model's "
        "compilation behavior before expecting speedups. Not all models "
        "benefit equally from compilation."
    )

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
