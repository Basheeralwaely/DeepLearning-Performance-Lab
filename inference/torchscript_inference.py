"""
TorchScript JIT Compilation
============================

TorchScript is PyTorch's mechanism for serializing and optimizing models.
It converts eager-mode Python models into an intermediate representation (IR)
that can be optimized by the PyTorch JIT compiler and deployed without Python
via LibTorch (C++ runtime).

Two approaches exist for converting models to TorchScript:

- **Tracing** (torch.jit.trace): Records the operations executed with a
  specific example input. Fast and simple, but cannot capture control flow
  (if/else, loops that depend on data).

- **Scripting** (torch.jit.script): Analyzes the Python source code directly
  and compiles it, preserving control flow and dynamic behavior.

What this tutorial covers:
  1. Baseline eager-mode PyTorch inference
  2. TorchScript tracing -- record-and-replay approach
  3. TorchScript scripting -- source-code compilation approach
  4. Tracing vs Scripting control flow comparison
  5. Serialization and deployment patterns
"""

import os
import sys
import tempfile
import warnings

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

logger = setup_logging("torchscript_inference")

NUM_ITERATIONS = 100
BATCH_SIZE = 64
WARMUP_ITERATIONS = 10


class BranchingModel(nn.Module):
    """Model with control flow to demonstrate tracing limitations."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return self.linear(x)
        else:
            return self.linear(x) * -1


def main():
    temp_files = []

    try:
        # ==============================================================
        # Section 1: PyTorch Baseline Inference
        # ==============================================================
        print("\n" + "=" * 60)
        print("  Section 1: PyTorch Baseline Inference")
        print("=" * 60 + "\n")

        device = get_device()
        print_device_info()

        model = SimpleCNN().eval().to(device)
        inputs, _ = get_sample_batch(batch_size=BATCH_SIZE, device=device)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: SimpleCNN with {total_params:,} parameters")
        logger.info(f"Input shape: {tuple(inputs.shape)}")
        logger.info(f"Benchmark iterations: {NUM_ITERATIONS}")

        # Warmup
        logger.info(f"Running {WARMUP_ITERATIONS} warmup iterations...")
        with torch.inference_mode():
            for _ in range(WARMUP_ITERATIONS):
                _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()

        @benchmark
        def pytorch_inference(m, x, num_iters):
            with torch.inference_mode():
                for _ in range(num_iters):
                    out = m(x)
            return out

        baseline = pytorch_inference(model, inputs, NUM_ITERATIONS)
        logger.info(f"Baseline eager inference: {baseline['time_seconds']:.4f}s")

        # Store eager output for numerical verification later
        with torch.inference_mode():
            eager_out = model(inputs)

        # ==============================================================
        # Section 2: TorchScript Tracing
        # ==============================================================
        print("\n" + "=" * 60)
        print("  Section 2: TorchScript Tracing")
        print("=" * 60 + "\n")

        logger.info("Tracing records operations executed with a specific example input.")
        logger.info("The resulting graph is a fixed sequence of ops -- no Python overhead.")

        dummy_input = torch.randn(1, 3, 32, 32, device=device)
        traced_model = torch.jit.trace(model, dummy_input)

        logger.info(f"Traced graph has {len(list(traced_model.graph.nodes()))} nodes")
        logger.info("Pros: Works with any model, no source code changes needed")
        logger.info("Cons: Cannot capture control flow (if/else), records only one execution path")

        # Save and reload to demonstrate serialization
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp.close()
        temp_files.append(tmp.name)
        traced_model.save(tmp.name)
        loaded_traced = torch.jit.load(tmp.name, map_location=device)
        file_size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        logger.info(f"Saved traced model: {file_size_mb:.2f} MB")

        # Warmup traced model
        with torch.inference_mode():
            for _ in range(WARMUP_ITERATIONS):
                _ = traced_model(inputs)

        @benchmark
        def traced_inference(m, x, num_iters):
            with torch.inference_mode():
                for _ in range(num_iters):
                    out = m(x)
            return out

        traced_result = traced_inference(traced_model, inputs, NUM_ITERATIONS)
        compare_results(baseline, traced_result, "TorchScript Tracing")

        # Store traced output for numerical verification
        with torch.inference_mode():
            traced_out = traced_model(inputs)

        # ==============================================================
        # Section 3: TorchScript Scripting
        # ==============================================================
        print("\n" + "=" * 60)
        print("  Section 3: TorchScript Scripting")
        print("=" * 60 + "\n")

        logger.info("Scripting analyzes Python source code and compiles it,")
        logger.info("preserving control flow (if/else, loops) in the graph.")

        scripted_model = torch.jit.script(model)

        logger.info(f"Scripted model type: {type(scripted_model).__name__}")
        logger.info("Pros: Handles control flow (if/else, loops), preserves full model logic")
        logger.info("Cons: More restrictive on Python features, may require type annotations")

        # Warmup scripted model
        with torch.inference_mode():
            for _ in range(WARMUP_ITERATIONS):
                _ = scripted_model(inputs)

        @benchmark
        def scripted_inference(m, x, num_iters):
            with torch.inference_mode():
                for _ in range(num_iters):
                    out = m(x)
            return out

        scripted_result = scripted_inference(scripted_model, inputs, NUM_ITERATIONS)
        compare_results(baseline, scripted_result, "TorchScript Scripting")

        # Store scripted output for numerical verification
        with torch.inference_mode():
            scripted_out = scripted_model(inputs)

        # ==============================================================
        # Section 4: Tracing vs Scripting -- Control Flow Demo
        # ==============================================================
        print("\n" + "=" * 60)
        print("  Section 4: Tracing vs Scripting -- Control Flow Demo")
        print("=" * 60 + "\n")

        logger.info("Demonstrating how tracing and scripting handle control flow differently.")
        logger.info("BranchingModel has an if/else in forward() that depends on input values.\n")

        branch_model = BranchingModel().eval().to(device)
        positive_input = torch.ones(1, 10, device=device)
        negative_input = -torch.ones(1, 10, device=device)

        # Eager mode reference outputs
        with torch.inference_mode():
            eager_pos = branch_model(positive_input)
            eager_neg = branch_model(negative_input)

        logger.info(f"Eager model -- positive input output[0]: {eager_pos[0, 0].item():.4f}")
        logger.info(f"Eager model -- negative input output[0]: {eager_neg[0, 0].item():.4f}")

        # Tracing demo: trace with positive input
        logger.info("\n--- Tracing the BranchingModel (traced with positive input) ---")
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            traced_branch = torch.jit.trace(branch_model, positive_input)

        for w in caught_warnings:
            if "TracerWarning" in str(w.category.__name__) or "Tracer" in str(w.message):
                logger.info(f"Caught warning: {w.message}")

        with torch.inference_mode():
            traced_pos = traced_branch(positive_input)
            traced_neg = traced_branch(negative_input)

        logger.info(f"Traced model -- positive input output[0]: {traced_pos[0, 0].item():.4f}")
        logger.info(f"Traced model -- negative input output[0]: {traced_neg[0, 0].item():.4f}")
        logger.info("Traced model ignores the if/else branch -- always takes the path seen during tracing")

        # Scripting demo: script correctly handles both branches
        logger.info("\n--- Scripting the BranchingModel ---")
        scripted_branch = torch.jit.script(branch_model)

        with torch.inference_mode():
            script_pos = scripted_branch(positive_input)
            script_neg = scripted_branch(negative_input)

        logger.info(f"Scripted model -- positive input output[0]: {script_pos[0, 0].item():.4f}")
        logger.info(f"Scripted model -- negative input output[0]: {script_neg[0, 0].item():.4f}")
        logger.info("Scripted model correctly handles both branches")

        # Summary comparison
        logger.info("\nSummary:")
        logger.info("  Tracing: fast, simple, but misses control flow")
        logger.info("  Scripting: handles control flow, slightly more restrictive on Python features")

        # ==============================================================
        # Section 5: Serialization and Deployment
        # ==============================================================
        print("\n" + "=" * 60)
        print("  Section 5: Serialization and Deployment")
        print("=" * 60 + "\n")

        # Save traced model
        traced_path = tempfile.NamedTemporaryFile(suffix="_traced.pt", delete=False)
        traced_path.close()
        temp_files.append(traced_path.name)
        traced_model.save(traced_path.name)
        traced_size = os.path.getsize(traced_path.name) / (1024 * 1024)

        # Save scripted model
        scripted_path = tempfile.NamedTemporaryFile(suffix="_scripted.pt", delete=False)
        scripted_path.close()
        temp_files.append(scripted_path.name)
        scripted_model.save(scripted_path.name)
        scripted_size = os.path.getsize(scripted_path.name) / (1024 * 1024)

        logger.info(f"Traced model file size:   {traced_size:.2f} MB")
        logger.info(f"Scripted model file size:  {scripted_size:.2f} MB")

        # Load and verify round-trip
        loaded_traced_model = torch.jit.load(traced_path.name, map_location=device)
        loaded_scripted_model = torch.jit.load(scripted_path.name, map_location=device)

        with torch.inference_mode():
            loaded_traced_output = loaded_traced_model(inputs)
            loaded_scripted_output = loaded_scripted_model(inputs)

        traced_match = torch.allclose(traced_out, loaded_traced_output, atol=1e-6)
        scripted_match = torch.allclose(scripted_out, loaded_scripted_output, atol=1e-6)

        logger.info(f"Traced save/load round-trip match:   {traced_match}")
        logger.info(f"Scripted save/load round-trip match: {scripted_match}")
        logger.info("TorchScript models can be loaded in C++ via LibTorch for production deployment without Python")

        # ==============================================================
        # Section 6: Final Benchmark Comparison
        # ==============================================================
        print("\n" + "=" * 60)
        print("  Section 6: Final Benchmark Comparison")
        print("=" * 60 + "\n")

        results = [
            {
                "name": "PyTorch Eager",
                "time_seconds": baseline["time_seconds"],
                "memory_mb": baseline.get("memory_mb"),
            },
            {
                "name": "TorchScript Traced",
                "time_seconds": traced_result["time_seconds"],
                "memory_mb": traced_result.get("memory_mb"),
            },
            {
                "name": "TorchScript Scripted",
                "time_seconds": scripted_result["time_seconds"],
                "memory_mb": scripted_result.get("memory_mb"),
            },
        ]

        print_benchmark_table(results)

        # Determine fastest
        fastest = min(results, key=lambda r: r["time_seconds"])
        baseline_time = baseline["time_seconds"]
        speedup = baseline_time / fastest["time_seconds"] if fastest["time_seconds"] > 0 else float("inf")
        logger.info(f"Fastest approach: {fastest['name']} ({speedup:.2f}x vs eager baseline)")

        # Numerical equivalence verification
        eager_vs_traced = torch.allclose(eager_out, traced_out, atol=1e-6)
        eager_vs_scripted = torch.allclose(eager_out, scripted_out, atol=1e-6)
        logger.info(f"Numerical equivalence (eager vs traced):   {eager_vs_traced}")
        logger.info(f"Numerical equivalence (eager vs scripted): {eager_vs_scripted}")

        print("\n" + "=" * 60)
        print("  Tutorial Complete")
        print("=" * 60 + "\n")

    finally:
        # Cleanup temp files
        for path in temp_files:
            try:
                os.unlink(path)
                logger.info(f"Cleaned up temp file: {path}")
            except OSError:
                pass


if __name__ == "__main__":
    main()
