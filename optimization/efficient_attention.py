"""
Efficient Attention: Manual vs Scaled Dot-Product Attention (SDPA)
===================================================================

The attention mechanism is the computational bottleneck in Transformers.
The naive implementation computes Q @ K^T (an NxN matrix for sequence
length N), applies softmax, then multiplies by V. This is O(N^2) in both
time and memory.

PyTorch 2.0+ provides torch.nn.functional.scaled_dot_product_attention
(SDPA), which automatically selects the best available backend:

  1. FlashAttention-2: Fuses the entire attention computation into a
     single kernel, never materializing the full NxN attention matrix.
     O(N) memory instead of O(N^2). Requires: SM80+ (A100/H100).

  2. Memory-Efficient Attention (xFormers): Similar fusion with broader
     hardware support. Works on older GPUs.

  3. Math fallback: Standard PyTorch implementation when neither
     optimized backend is available (still faster than manual code due
     to C++ implementation).

When to use:
  - Any Transformer model (NLP, Vision, Audio)
  - Long sequence lengths where the N^2 attention matrix is the bottleneck
  - When you want automatic backend selection without manual kernel calls

What this tutorial demonstrates:
  1. Manual attention implementation (the naive way)
  2. SDPA implementation (one function call)
  3. Performance comparison across sequence lengths
  4. Memory comparison (where the biggest win is)
  5. Backend detection and availability
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (
    setup_logging,
    benchmark,
    compare_results,
    print_benchmark_table,
    get_device,
    print_device_info,
)

logger = setup_logging("efficient_attention")

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
BATCH_SIZE = 8
NUM_HEADS = 8
HEAD_DIM = 64
EMBED_DIM = NUM_HEADS * HEAD_DIM  # 512
NUM_ITERATIONS = 100
WARMUP_ITERATIONS = 10
SEQUENCE_LENGTHS = [64, 128, 256, 512, 1024]


# ---------------------------------------------------------------
# Attention implementations
# ---------------------------------------------------------------

class ManualAttention(nn.Module):
    """Standard multi-head attention computed step by step.

    This is the textbook implementation that materializes the full NxN
    attention matrix. Memory usage scales as O(batch * heads * seq_len^2).
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, S, _ = x.shape
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores -- this creates the full S x S matrix
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, S, S)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, H, S, D)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.out_proj(out)


class SDPAttention(nn.Module):
    """Multi-head attention using torch.nn.functional.scaled_dot_product_attention.

    SDPA automatically selects the best backend (FlashAttention, Memory-Efficient,
    or Math) based on hardware and input properties. It avoids materializing the
    full NxN attention matrix when an optimized backend is available.
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # One function call replaces matmul + scale + softmax + matmul
        # The backend handles fusion and memory optimization automatically
        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, S, D)

        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def run_attention_benchmark(model, x, num_iterations):
    """Run forward passes for timing."""
    with torch.inference_mode():
        for _ in range(num_iterations):
            _ = model(x)


def main():
    # ==============================================================
    # SETUP
    # ==============================================================
    print("\n" + "=" * 60)
    print("  Efficient Attention: Manual vs SDPA")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Embed dim: {EMBED_DIM} ({NUM_HEADS} heads x {HEAD_DIM} head dim)")
    logger.info(f"Iterations per benchmark: {NUM_ITERATIONS}")
    logger.info(f"Sequence lengths to test: {SEQUENCE_LENGTHS}")

    # ==============================================================
    # SECTION 1: Backend Detection
    # ==============================================================
    print("\n--- Section 1: SDPA Backend Availability ---\n")

    logger.info("Checking which SDPA backends are available on this system...")

    # Create a small test input to probe backends
    test_q = torch.randn(1, NUM_HEADS, 32, HEAD_DIM, device=device)
    test_k = torch.randn(1, NUM_HEADS, 32, HEAD_DIM, device=device)
    test_v = torch.randn(1, NUM_HEADS, 32, HEAD_DIM, device=device)

    backends = {
        "FlashAttention": torch.backends.cuda.flash_sdp_enabled() if hasattr(torch.backends.cuda, "flash_sdp_enabled") else False,
        "Memory-Efficient": torch.backends.cuda.mem_efficient_sdp_enabled() if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled") else False,
        "Math (fallback)": torch.backends.cuda.math_sdp_enabled() if hasattr(torch.backends.cuda, "math_sdp_enabled") else True,
    }

    for name, available in backends.items():
        status = "AVAILABLE" if available else "NOT AVAILABLE"
        logger.info(f"  {name}: {status}")

    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        logger.info(f"\n  GPU Compute Capability: {cap[0]}.{cap[1]}")
        if cap[0] >= 8:
            logger.info("  SM80+ detected -- FlashAttention should be available.")
        else:
            logger.info(
                "  SM < 80 -- FlashAttention requires A100/H100 or newer. "
                "Memory-Efficient backend will be used instead."
            )
    else:
        logger.info("\n  Running on CPU -- Math fallback will be used.")
        logger.info("  SDPA still provides speedup over manual code on CPU.")

    # ==============================================================
    # SECTION 2: Correctness Check
    # ==============================================================
    print("\n--- Section 2: Correctness Verification ---\n")

    seq_len = 64
    x = torch.randn(BATCH_SIZE, seq_len, EMBED_DIM, device=device)

    manual_attn = ManualAttention(EMBED_DIM, NUM_HEADS).to(device).eval()
    sdpa_attn = SDPAttention(EMBED_DIM, NUM_HEADS).to(device).eval()

    # Copy weights so both models are identical
    sdpa_attn.qkv.weight.data.copy_(manual_attn.qkv.weight.data)
    sdpa_attn.qkv.bias.data.copy_(manual_attn.qkv.bias.data)
    sdpa_attn.out_proj.weight.data.copy_(manual_attn.out_proj.weight.data)
    sdpa_attn.out_proj.bias.data.copy_(manual_attn.out_proj.bias.data)

    with torch.inference_mode():
        manual_out = manual_attn(x)
        sdpa_out = sdpa_attn(x)

    max_diff = (manual_out - sdpa_out).abs().max().item()
    mean_diff = (manual_out - sdpa_out).abs().mean().item()
    logger.info(f"Max absolute difference: {max_diff:.2e}")
    logger.info(f"Mean absolute difference: {mean_diff:.2e}")

    if max_diff < 1e-3:
        logger.info("Outputs match -- SDPA produces equivalent results.")
    else:
        logger.warning(
            "Larger difference detected. This can happen with FlashAttention "
            "due to different accumulation order (still numerically valid)."
        )

    # ==============================================================
    # SECTION 3: Performance Comparison Across Sequence Lengths
    # ==============================================================
    print("\n--- Section 3: Performance Across Sequence Lengths ---\n")
    logger.info(
        "As sequence length grows, the O(N^2) attention matrix becomes the "
        "bottleneck. SDPA's fused kernels avoid materializing this matrix, "
        "so the speedup grows with sequence length."
    )

    all_results = []

    for seq_len in SEQUENCE_LENGTHS:
        logger.info(f"\n  seq_len={seq_len}:")
        x = torch.randn(BATCH_SIZE, seq_len, EMBED_DIM, device=device)

        # Fresh models for each test
        manual = ManualAttention(EMBED_DIM, NUM_HEADS).to(device).eval()
        sdpa = SDPAttention(EMBED_DIM, NUM_HEADS).to(device).eval()
        sdpa.qkv.weight.data.copy_(manual.qkv.weight.data)
        sdpa.qkv.bias.data.copy_(manual.qkv.bias.data)
        sdpa.out_proj.weight.data.copy_(manual.out_proj.weight.data)
        sdpa.out_proj.bias.data.copy_(manual.out_proj.bias.data)

        # Warmup
        with torch.inference_mode():
            for _ in range(WARMUP_ITERATIONS):
                _ = manual(x)
                _ = sdpa(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark manual
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        @benchmark
        def manual_run(m=manual, inp=x):
            run_attention_benchmark(m, inp, NUM_ITERATIONS)
            if device.type == "cuda":
                torch.cuda.synchronize()

        manual_result = manual_run()

        manual_mem = manual_result.get("memory_mb")

        # Benchmark SDPA
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        @benchmark
        def sdpa_run(m=sdpa, inp=x):
            run_attention_benchmark(m, inp, NUM_ITERATIONS)
            if device.type == "cuda":
                torch.cuda.synchronize()

        sdpa_result = sdpa_run()

        sdpa_mem = sdpa_result.get("memory_mb")

        m_per = manual_result["time_seconds"] / NUM_ITERATIONS * 1000
        s_per = sdpa_result["time_seconds"] / NUM_ITERATIONS * 1000
        speedup = manual_result["time_seconds"] / sdpa_result["time_seconds"] if sdpa_result["time_seconds"] > 0 else float("inf")

        logger.info(f"    Manual: {m_per:.3f}ms per call")
        logger.info(f"    SDPA:   {s_per:.3f}ms per call")
        logger.info(f"    Speedup: {speedup:.2f}x")

        if manual_mem is not None and sdpa_mem is not None:
            mem_savings = manual_mem - sdpa_mem
            logger.info(f"    Memory: Manual={manual_mem:.1f}MB, SDPA={sdpa_mem:.1f}MB (saved {mem_savings:.1f}MB)")

        all_results.append({
            "name": f"Manual (seq={seq_len})",
            "time_seconds": manual_result["time_seconds"],
            "memory_mb": manual_mem,
        })
        all_results.append({
            "name": f"SDPA (seq={seq_len})",
            "time_seconds": sdpa_result["time_seconds"],
            "memory_mb": sdpa_mem,
        })

    print("\n--- Benchmark Results ---\n")
    print_benchmark_table(all_results)

    # ==============================================================
    # SECTION 4: Memory Scaling Analysis
    # ==============================================================
    print("\n--- Section 4: Memory Scaling ---\n")

    if device.type == "cuda":
        logger.info(
            "The key advantage of SDPA is memory efficiency. Manual attention "
            "allocates an (B, H, S, S) attention matrix. At seq_len=1024 with "
            f"batch={BATCH_SIZE}, heads={NUM_HEADS}: that's "
            f"{BATCH_SIZE * NUM_HEADS * 1024 * 1024 * 4 / (1024**2):.0f} MB "
            "just for the attention matrix alone."
        )
        logger.info(
            "SDPA with FlashAttention never materializes this matrix, computing "
            "attention in tiled blocks that fit in SRAM."
        )
    else:
        logger.info("Memory analysis is most meaningful on GPU (run with CUDA).")

    # ==============================================================
    # Key Takeaways
    # ==============================================================
    print("\n--- Key Takeaways ---\n")
    logger.info(
        "1. F.scaled_dot_product_attention is a drop-in replacement for "
        "manual Q @ K^T / sqrt(d) + softmax + @ V computation."
    )
    logger.info(
        "2. SDPA automatically selects the best backend: FlashAttention "
        "(SM80+), Memory-Efficient (older GPUs), or Math fallback (CPU)."
    )
    logger.info(
        "3. The speedup grows with sequence length because SDPA avoids "
        "materializing the O(N^2) attention matrix."
    )
    logger.info(
        "4. Memory savings are the biggest win: FlashAttention uses O(N) "
        "memory instead of O(N^2), enabling much longer sequences."
    )
    logger.info(
        "5. For existing Transformer models: replace manual attention with "
        "one SDPA call. For new models: use nn.MultiheadAttention which "
        "already uses SDPA internally in PyTorch 2.0+."
    )

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
