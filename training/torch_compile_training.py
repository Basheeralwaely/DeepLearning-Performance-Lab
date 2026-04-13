"""
torch.compile() Training Speedup: From Eager to Compiled Training Loops
========================================================================

torch.compile is the single biggest free speedup in modern PyTorch (2.x+).
While profiling/torch_compile_diagnostics.py covers how compilation works
and how to diagnose graph breaks, THIS tutorial shows the practical payoff:
compiling an actual training loop and measuring the before/after difference.

How it works:
  torch.compile wraps your model (or a function) and, on the first call,
  traces the computation graph via TorchDynamo, then compiles it to optimized
  kernels via TorchInductor. Subsequent calls reuse the compiled graph,
  skipping Python overhead entirely.

Why it speeds up training:
  1. Operator fusion: multiple small ops become one big kernel
  2. Memory planning: reduces intermediate tensor allocations
  3. Eliminates Python interpreter overhead in the hot path
  4. Hardware-specific code generation (Triton kernels for GPU)

What this tutorial demonstrates:
  1. Baseline eager training loop (standard PyTorch)
  2. Compiled training loop with torch.compile(model)
  3. Compiled training with different modes (default, reduce-overhead, max-autotune)
  4. Side-by-side benchmark comparison of throughput and memory
  5. Practical tips: what to compile, when compilation hurts, warm-up costs
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

logger = setup_logging("torch_compile_training")

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
BATCH_SIZE = 64
NUM_ITERATIONS = 100
WARMUP_ITERATIONS = 10
COMPILE_MODES = ["default", "reduce-overhead", "max-autotune"]


# ---------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------

def train_loop(model, optimizer, criterion, device, num_iterations):
    """Run a training loop for a fixed number of iterations.

    Args:
        model: The model to train (already on device).
        optimizer: The optimizer.
        criterion: Loss function.
        device: Target device.
        num_iterations: Number of training steps.

    Returns:
        Final loss value (float).
    """
    model.train()
    loss_val = 0.0
    for _ in range(num_iterations):
        inputs, labels = get_sample_batch(batch_size=BATCH_SIZE, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
    return loss_val


def main():
    # ==============================================================
    # SETUP
    # ==============================================================
    print("\n" + "=" * 60)
    print("  torch.compile Training: Eager vs Compiled Training Loops")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    if device.type != "cuda":
        logger.warning(
            "torch.compile works on CPU but the speedup is most visible on GPU. "
            "Results on CPU may show minimal or no improvement."
        )

    # We create a fresh model for each test to ensure fair comparison
    def make_model():
        m = SimpleCNN().to(device)
        return m

    total_params = sum(p.numel() for p in make_model().parameters())
    logger.info(f"Model: SimpleCNN with {total_params:,} parameters")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Training iterations per benchmark: {NUM_ITERATIONS}")
    logger.info(f"Warmup iterations: {WARMUP_ITERATIONS}")
    logger.info(f"Compile modes to test: {COMPILE_MODES}")

    # ==============================================================
    # SECTION 1: Eager Baseline
    # ==============================================================
    print("\n--- Section 1: Eager (Uncompiled) Training Baseline ---\n")
    logger.info(
        "Standard PyTorch training -- every forward/backward pass goes through "
        "the Python interpreter, dispatching individual CUDA kernels one at a time."
    )

    model_eager = make_model()
    optimizer = torch.optim.SGD(model_eager.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Warm up CUDA and model
    logger.info("Running warmup passes...")
    train_loop(model_eager, optimizer, criterion, device, WARMUP_ITERATIONS)

    # Reset optimizer state for fair timing
    model_eager = make_model()
    optimizer = torch.optim.SGD(model_eager.parameters(), lr=0.01, momentum=0.9)

    @benchmark
    def eager_training():
        return train_loop(model_eager, optimizer, criterion, device, NUM_ITERATIONS)

    eager_result = eager_training()
    eager_per_iter = eager_result["time_seconds"] / NUM_ITERATIONS
    logger.info(
        f"Eager training: {eager_result['time_seconds']:.4f}s total, "
        f"{eager_per_iter * 1000:.2f}ms per step"
    )
    if eager_result.get("memory_mb") is not None:
        logger.info(f"Eager peak memory: {eager_result['memory_mb']:.1f} MB")

    # ==============================================================
    # SECTION 2: Compiled Training (default mode)
    # ==============================================================
    print("\n--- Section 2: Compiled Training (default mode) ---\n")
    logger.info(
        "torch.compile(model) replaces the eager forward pass with a compiled "
        "graph. The backward pass is automatically compiled too because autograd "
        "traces through the compiled forward."
    )
    logger.info(
        "The first few iterations are SLOWER because compilation happens on "
        "the first call. After that, every iteration uses the fast compiled path."
    )

    dynamo.reset()
    model_compiled = make_model()
    model_compiled = torch.compile(model_compiled, mode="default")
    optimizer_c = torch.optim.SGD(model_compiled.parameters(), lr=0.01, momentum=0.9)

    # Measure compilation overhead (first forward+backward)
    logger.info("Measuring compilation overhead (first training step)...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    compile_start = time.perf_counter()
    inputs, labels = get_sample_batch(batch_size=BATCH_SIZE, device=device)
    optimizer_c.zero_grad(set_to_none=True)
    outputs = model_compiled(inputs)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    optimizer_c.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    compile_time = time.perf_counter() - compile_start
    logger.info(f"First step (includes compilation): {compile_time:.2f}s")

    # Warm up compiled path
    train_loop(model_compiled, optimizer_c, criterion, device, WARMUP_ITERATIONS)

    # Now benchmark the compiled training
    model_compiled_bench = make_model()
    model_compiled_bench = torch.compile(model_compiled_bench, mode="default")
    optimizer_cb = torch.optim.SGD(model_compiled_bench.parameters(), lr=0.01, momentum=0.9)
    # Trigger compilation
    train_loop(model_compiled_bench, optimizer_cb, criterion, device, WARMUP_ITERATIONS)

    # Reset for fair measurement
    model_compiled_bench = make_model()
    model_compiled_bench = torch.compile(model_compiled_bench, mode="default")
    optimizer_cb = torch.optim.SGD(model_compiled_bench.parameters(), lr=0.01, momentum=0.9)
    # Trigger compilation once
    train_loop(model_compiled_bench, optimizer_cb, criterion, device, 2)

    @benchmark
    def compiled_training():
        return train_loop(
            model_compiled_bench, optimizer_cb, criterion, device, NUM_ITERATIONS
        )

    compiled_result = compiled_training()
    compiled_per_iter = compiled_result["time_seconds"] / NUM_ITERATIONS
    logger.info(
        f"Compiled training: {compiled_result['time_seconds']:.4f}s total, "
        f"{compiled_per_iter * 1000:.2f}ms per step"
    )
    if compiled_result.get("memory_mb") is not None:
        logger.info(f"Compiled peak memory: {compiled_result['memory_mb']:.1f} MB")

    compare_results(eager_result, compiled_result, "torch.compile (default)")

    # ==============================================================
    # SECTION 3: All Compile Modes Comparison
    # ==============================================================
    print("\n--- Section 3: All Compile Modes Comparison ---\n")
    logger.info(
        "PyTorch offers three compilation modes with different tradeoffs:\n"
        "  - default: Balanced compile time vs runtime speedup\n"
        "  - reduce-overhead: Minimizes kernel launch overhead using CUDA graphs\n"
        "    internally. Best for small models where launch overhead dominates.\n"
        "  - max-autotune: Tries many kernel variants and picks the fastest.\n"
        "    Slowest to compile but potentially fastest runtime."
    )

    all_results = [
        {
            "name": "eager (no compile)",
            "time_seconds": eager_result["time_seconds"],
            "memory_mb": eager_result.get("memory_mb"),
        }
    ]

    for mode in COMPILE_MODES:
        logger.info(f"\n  Testing mode='{mode}'...")
        dynamo.reset()

        model_m = make_model()
        model_m = torch.compile(model_m, mode=mode)
        opt_m = torch.optim.SGD(model_m.parameters(), lr=0.01, momentum=0.9)

        # Trigger compilation and warm up
        train_loop(model_m, opt_m, criterion, device, WARMUP_ITERATIONS)

        # Fresh model with same mode for fair timing
        dynamo.reset()
        model_m = make_model()
        model_m = torch.compile(model_m, mode=mode)
        opt_m = torch.optim.SGD(model_m.parameters(), lr=0.01, momentum=0.9)
        train_loop(model_m, opt_m, criterion, device, 2)  # trigger compile

        @benchmark
        def mode_training(m=model_m, o=opt_m):
            return train_loop(m, o, criterion, device, NUM_ITERATIONS)

        result = mode_training()
        per_iter = result["time_seconds"] / NUM_ITERATIONS
        logger.info(
            f"  {mode}: {result['time_seconds']:.4f}s total, "
            f"{per_iter * 1000:.2f}ms per step"
        )

        all_results.append({
            "name": f"compile({mode})",
            "time_seconds": result["time_seconds"],
            "memory_mb": result.get("memory_mb"),
        })

    # Print comparison table
    print("\n--- Benchmark Results ---\n")
    print_benchmark_table(all_results)

    # Print speedups
    eager_time = eager_result["time_seconds"]
    for r in all_results[1:]:
        speedup = eager_time / r["time_seconds"] if r["time_seconds"] > 0 else float("inf")
        logger.info(f"  {r['name']} speedup vs eager: {speedup:.2f}x")

    # ==============================================================
    # SECTION 4: Practical Tips
    # ==============================================================
    print("\n--- Section 4: Practical Tips ---\n")

    logger.info("WHAT TO COMPILE:")
    logger.info("  - torch.compile(model) is the most common pattern")
    logger.info("  - You can also compile individual functions with @torch.compile")
    logger.info("  - The optimizer step is NOT compiled -- only forward + backward")

    logger.info("\nWHEN COMPILATION HURTS:")
    logger.info("  - Very short training runs (compilation cost > runtime savings)")
    logger.info("  - Models with heavy data-dependent control flow (graph breaks)")
    logger.info("  - Dynamic shapes that change every iteration (recompilation)")
    logger.info("  - Debugging (compiled stack traces are harder to read)")

    logger.info("\nBEST PRACTICES:")
    logger.info("  - Always warm up after compilation before benchmarking")
    logger.info(
        "  - Use torch._dynamo.explain(model)(input) to check for graph breaks "
        "(see profiling/torch_compile_diagnostics.py)"
    )
    logger.info("  - For production: compile once, train for many epochs")
    logger.info(
        "  - Combine with AMP for maximum speedup: "
        "torch.compile + torch.amp.autocast"
    )
    logger.info(
        "  - Use set_to_none=True in optimizer.zero_grad() -- it helps the "
        "compiler optimize memory reuse"
    )

    # ==============================================================
    # Key Takeaways
    # ==============================================================
    print("\n--- Key Takeaways ---\n")
    logger.info(
        "1. torch.compile provides a meaningful training speedup with one line "
        "of code -- no model changes required"
    )
    logger.info(
        "2. Compilation is a one-time cost amortized over the full training run. "
        "The longer you train, the more you benefit."
    )
    logger.info(
        "3. 'reduce-overhead' mode is best for small models; 'max-autotune' "
        "is best for large models where you can afford slow compilation."
    )
    logger.info(
        "4. Combine torch.compile with AMP (mixed_precision/amp_training.py) "
        "and gradient checkpointing (training/gradient_checkpointing.py) for "
        "maximum training performance."
    )

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
