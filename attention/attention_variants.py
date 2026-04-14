"""
Attention Variants: Self, Cross, Multi-Head, and Causal
========================================================

This is Tutorial 1 of Phase 7. It teaches the four canonical attention
variants from first principles by implementing each one with explicit
``torch.matmul`` + ``torch.softmax`` calls -- NO fused SDPA kernel,
NO ``nn`` multi-head module. Tutorial 2 (FlashAttention / SDPA) is
where the fused-kernel speedup is unveiled; this file deliberately
stays hand-written so that contrast lands.

What each variant is and when to use it:

  * Self-attention   - Q, K, V come from the same sequence. Used in
                       encoder layers (BERT, ViT) and decoder self-attn
                       blocks. Each token attends to every other token.
  * Cross-attention  - Q comes from one sequence, K/V from another
                       (Nq != Nk). Used in encoder-decoder models,
                       DiT (denoised tokens attend to text/class
                       conditioning), Stable Diffusion U-Net cross-attn.
  * Multi-head       - Splits D into H independent heads of size D/H,
                       runs scaled_dot_product per head in parallel,
                       then concatenates and projects back. Lets the
                       model attend to different subspaces at once.
  * Causal           - Self-attention with a lower-triangular mask so
                       token i can only see tokens 0..i. Used in every
                       autoregressive decoder (GPT, LLaMA).

What this tutorial demonstrates:
  1. Hand-written ``scaled_dot_product`` primitive (matmul + softmax).
  2. Four variants built on top of that primitive.
  3. A correctness check that the causal mask actually prevents
     future tokens from leaking into past tokens' outputs.
  4. Per-variant timing + peak GPU memory in a side-by-side table.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from utils import (
    setup_logging,
    benchmark,
    print_benchmark_table,
    get_device,
    print_device_info,
)

logger = setup_logging("attention_variants")


# ----------------------------------------------------------------------
# Sizing constants (taken from RESEARCH.md "Suggested numeric scale")
# ----------------------------------------------------------------------
B = 4              # batch size
N = 512            # sequence length (Nq for self/causal/multi-head)
N_KV = 256         # K/V sequence length for cross-attention (Nq != Nk)
D = 256            # model dim
H = 8              # number of heads (d_h = D / H = 32)
D_H = D // H
NUM_ITERATIONS = 100
WARMUP_ITERATIONS = 10


# ----------------------------------------------------------------------
# Hand-written scaled-dot-product primitive
# ----------------------------------------------------------------------
def scaled_dot_product(q, k, v, mask=None):
    """Scaled dot-product attention, written out explicitly.

    Args:
        q: query tensor of shape [B, H, Nq, d_h]
        k: key tensor of shape   [B, H, Nk, d_h]
        v: value tensor of shape [B, H, Nk, d_h]
        mask: optional [Nq, Nk] (or broadcastable) tensor where 0 entries
              are masked out (set to -inf before softmax).

    Returns:
        Attention output of shape [B, H, Nq, d_h].

    NOTE: This is intentionally hand-written (matmul + softmax). Tutorial 2
    will swap this primitive for the fused SDPA / FlashAttention kernel and
    show the speedup.
    """
    d_h = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_h ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


# ----------------------------------------------------------------------
# Four attention variants -- all built on scaled_dot_product
# ----------------------------------------------------------------------
def self_attention(q, k, v):
    """Self-attention: Q, K, V from the same sequence, no mask."""
    return scaled_dot_product(q, k, v)


def cross_attention(q, k, v):
    """Cross-attention: Q from one sequence, K/V from another (Nq != Nk)."""
    return scaled_dot_product(q, k, v)


def multi_head_attention(x, q_proj, k_proj, v_proj, o_proj):
    """End-to-end multi-head attention: project, head-split, attend, merge.

    Demonstrates Pattern 2 from RESEARCH.md (multi-head shape discipline):
    [B, N, D] -> [B, N, H, D_H] -> transpose -> [B, H, N, D_H] for the
    per-head matmul, then transpose+contiguous+view to merge heads back.
    """
    B_, N_, D_ = x.shape
    q = q_proj(x).view(B_, N_, H, D_H).transpose(1, 2)  # [B, H, N, D_H]
    k = k_proj(x).view(B_, N_, H, D_H).transpose(1, 2)
    v = v_proj(x).view(B_, N_, H, D_H).transpose(1, 2)
    out = scaled_dot_product(q, k, v)                    # [B, H, N, D_H]
    out = out.transpose(1, 2).contiguous().view(B_, N_, D_)
    return o_proj(out)


def causal_attention(q, k, v):
    """Causal self-attention: lower-triangular mask blocks future tokens."""
    Nq, Nk = q.size(-2), k.size(-2)
    mask = torch.tril(torch.ones(Nq, Nk, device=q.device))
    return scaled_dot_product(q, k, v, mask=mask)


# ----------------------------------------------------------------------
# Correctness check: causal mask actually blocks future tokens
# ----------------------------------------------------------------------
def causal_correctness_check(device, tol: float = 1e-5) -> bool:
    """Verify that perturbing token i+1 does not change the output at i.

    Pitfall 3 from RESEARCH.md: a buggy causal mask (e.g., ``triu`` instead
    of ``tril``) silently leaks future information into past positions
    during training. We catch that here with a small concrete test.
    """
    torch.manual_seed(0)
    b, h, n, d_h = 1, 1, 8, 4
    q = torch.randn(b, h, n, d_h, device=device)
    k = torch.randn(b, h, n, d_h, device=device)
    v = torch.randn(b, h, n, d_h, device=device)

    out1 = causal_attention(q, k, v)

    # Perturb token at position 4 in K and V; positions 0..3 must be unchanged.
    k2 = k.clone()
    v2 = v.clone()
    k2[:, :, 4, :] += 10.0
    v2[:, :, 4, :] += 10.0
    out2 = causal_attention(q, k2, v2)

    diff_past = (out1[:, :, :4, :] - out2[:, :, :4, :]).abs().max().item()
    diff_future = (out1[:, :, 4:, :] - out2[:, :, 4:, :]).abs().max().item()

    passed = diff_past < tol and diff_future > tol
    if passed:
        logger.info(
            "Causal correctness: PASS (past max-diff=%.2e < tol=%.0e, "
            "future max-diff=%.2e > tol)", diff_past, tol, diff_future
        )
    else:
        logger.error(
            "Causal correctness: FAIL (past max-diff=%.2e, future max-diff=%.2e, "
            "tol=%.0e)", diff_past, diff_future, tol
        )
    return passed


# ----------------------------------------------------------------------
# main()
# ----------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  Attention Variants: Self / Cross / Multi-Head / Causal")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    logger.info(
        "Parameters: B=%d, N=%d, N_KV=%d, D=%d, H=%d, D_H=%d, "
        "iterations=%d (warmup=%d)",
        B, N, N_KV, D, H, D_H, NUM_ITERATIONS, WARMUP_ITERATIONS,
    )

    # ------------------------------------------------------------------
    # Prepare inputs ONCE (so timings reflect attention work, not setup)
    # ------------------------------------------------------------------
    logger.info("Preparing input tensors and projection layers...")
    torch.manual_seed(42)

    # Self / multi-head / causal inputs: [B, N, D]
    x = torch.randn(B, N, D, device=device)

    # Multi-head projection layers
    q_proj = nn.Linear(D, D).to(device)
    k_proj = nn.Linear(D, D).to(device)
    v_proj = nn.Linear(D, D).to(device)
    o_proj = nn.Linear(D, D).to(device)

    # Pre-computed head-split Q/K/V for the self / causal benchmarks
    # (so those benchmarks don't redo the projections and dilute the
    # attention-only cost).
    with torch.no_grad():
        q_self = q_proj(x).view(B, N, H, D_H).transpose(1, 2)  # [B, H, N, D_H]
        k_self = k_proj(x).view(B, N, H, D_H).transpose(1, 2)
        v_self = v_proj(x).view(B, N, H, D_H).transpose(1, 2)

    # Cross-attention: Q from main sequence (length N), K/V from context
    # sequence (length N_KV). Separate projections for the context side.
    x_ctx = torch.randn(B, N_KV, D, device=device)
    k_ctx_proj = nn.Linear(D, D).to(device)
    v_ctx_proj = nn.Linear(D, D).to(device)
    with torch.no_grad():
        q_cross = q_proj(x).view(B, N, H, D_H).transpose(1, 2)         # [B, H, N, D_H]
        k_cross = k_ctx_proj(x_ctx).view(B, N_KV, H, D_H).transpose(1, 2)  # [B, H, N_KV, D_H]
        v_cross = v_ctx_proj(x_ctx).view(B, N_KV, H, D_H).transpose(1, 2)

    # ------------------------------------------------------------------
    # Warm-up OUTSIDE the benchmark wrappers (compile CUDA kernels)
    # ------------------------------------------------------------------
    logger.info("Warm-up: running %d iterations of each variant...",
                WARMUP_ITERATIONS)
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            _ = self_attention(q_self, k_self, v_self)
            _ = cross_attention(q_cross, k_cross, v_cross)
            _ = multi_head_attention(x, q_proj, k_proj, v_proj, o_proj)
            _ = causal_attention(q_self, k_self, v_self)
    if device.type == "cuda":
        torch.cuda.synchronize()

    results = []

    # ------------------------------------------------------------------
    # Section 1: Self-Attention
    # ------------------------------------------------------------------
    print("\n--- Section 1: Self-Attention ---\n")
    logger.info("Self-attention: Q, K, V from the SAME sequence (length N=%d).", N)
    logger.info("This is the building block of BERT, ViT, and every encoder.")

    @benchmark
    def run_self():
        with torch.no_grad():
            for _ in range(NUM_ITERATIONS):
                _ = self_attention(q_self, k_self, v_self)
            if device.type == "cuda":
                torch.cuda.synchronize()

    r = run_self()
    logger.info("Self-attention: %d iters in %.4fs (peak mem %.1f MB)",
                NUM_ITERATIONS, r["time_seconds"],
                r["memory_mb"] if r["memory_mb"] is not None else float("nan"))
    results.append({
        "name": "self",
        "time_seconds": r["time_seconds"],
        "memory_mb": r["memory_mb"],
    })

    # ------------------------------------------------------------------
    # Section 2: Cross-Attention
    # ------------------------------------------------------------------
    print("\n--- Section 2: Cross-Attention ---\n")
    logger.info("Cross-attention: Q from sequence A (Nq=%d), K/V from sequence B (Nk=%d).",
                N, N_KV)
    logger.info("Used in encoder-decoder, DiT (text/class conditioning), "
                "Stable Diffusion U-Net cross-attn.")

    @benchmark
    def run_cross():
        with torch.no_grad():
            for _ in range(NUM_ITERATIONS):
                _ = cross_attention(q_cross, k_cross, v_cross)
            if device.type == "cuda":
                torch.cuda.synchronize()

    r = run_cross()
    logger.info("Cross-attention: %d iters in %.4fs (peak mem %.1f MB)",
                NUM_ITERATIONS, r["time_seconds"],
                r["memory_mb"] if r["memory_mb"] is not None else float("nan"))
    results.append({
        "name": "cross",
        "time_seconds": r["time_seconds"],
        "memory_mb": r["memory_mb"],
    })

    # ------------------------------------------------------------------
    # Section 3: Multi-Head Attention
    # ------------------------------------------------------------------
    print("\n--- Section 3: Multi-Head Attention ---\n")
    logger.info("Multi-head attention: split D=%d into H=%d heads of size d_h=%d, "
                "attend per-head in parallel, then concat + project.",
                D, H, D_H)
    logger.info("Includes Q/K/V/O projections end-to-end, so per-iteration cost "
                "is higher than the bare self-attention benchmark above.")

    @benchmark
    def run_multi_head():
        with torch.no_grad():
            for _ in range(NUM_ITERATIONS):
                _ = multi_head_attention(x, q_proj, k_proj, v_proj, o_proj)
            if device.type == "cuda":
                torch.cuda.synchronize()

    r = run_multi_head()
    logger.info("Multi-head attention: %d iters in %.4fs (peak mem %.1f MB)",
                NUM_ITERATIONS, r["time_seconds"],
                r["memory_mb"] if r["memory_mb"] is not None else float("nan"))
    results.append({
        "name": "multi_head",
        "time_seconds": r["time_seconds"],
        "memory_mb": r["memory_mb"],
    })

    # ------------------------------------------------------------------
    # Section 4: Causal Attention
    # ------------------------------------------------------------------
    print("\n--- Section 4: Causal Attention ---\n")
    logger.info("Causal self-attention: same shape as self-attention but with a "
                "lower-triangular mask (torch.tril) so token i sees only 0..i.")
    logger.info("Used in every autoregressive decoder (GPT, LLaMA, etc.).")

    @benchmark
    def run_causal():
        with torch.no_grad():
            for _ in range(NUM_ITERATIONS):
                _ = causal_attention(q_self, k_self, v_self)
            if device.type == "cuda":
                torch.cuda.synchronize()

    r = run_causal()
    logger.info("Causal attention: %d iters in %.4fs (peak mem %.1f MB)",
                NUM_ITERATIONS, r["time_seconds"],
                r["memory_mb"] if r["memory_mb"] is not None else float("nan"))
    results.append({
        "name": "causal",
        "time_seconds": r["time_seconds"],
        "memory_mb": r["memory_mb"],
    })

    # ------------------------------------------------------------------
    # Causal correctness check (Pitfall 3 from RESEARCH.md)
    # ------------------------------------------------------------------
    print("\n--- Causal Correctness Check ---\n")
    logger.info("Verifying that perturbing a future token does NOT affect past outputs...")
    causal_correctness_check(device, tol=1e-5)

    # ------------------------------------------------------------------
    # Final benchmark table -- four variants side-by-side
    # ------------------------------------------------------------------
    print("\n--- Benchmark Summary ---\n")
    logger.info("Per-variant timing and peak GPU memory (lower = better):")
    print_benchmark_table(results)

    print("\n--- Key Takeaways ---\n")
    logger.info("All four variants share one primitive: scaled_dot_product (matmul + softmax).")
    logger.info("Cross-attention costs scale with Nq * Nk -- shrinking the K/V sequence helps.")
    logger.info("Multi-head adds Q/K/V/O Linear cost on top of the attention itself.")
    logger.info("Causal mask adds an O(Nq*Nk) masked_fill -- cheap, but the softmax still runs.")
    logger.info("Tutorial 2 swaps scaled_dot_product for FlashAttention/SDPA and benchmarks the speedup.")

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
