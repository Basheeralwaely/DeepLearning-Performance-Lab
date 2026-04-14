"""
Flash Attention: Naive vs SDPA
===============================

The self-attention operation ``softmax(Q K^T / sqrt(d)) V`` requires
materializing a ``[B, H, N, N]`` score tensor. That tensor is the dominant
memory cost of the naive implementation: for ``B=16, H=8, N=4096`` in
bf16 it is ``16 * 8 * 4096 * 4096 * 2 bytes = 8 GiB`` -- just for the
scores, not counting gradients. This O(N^2) memory blowup is what makes
sequence length the single biggest cost driver in transformers, and it is
exactly what flash attention avoids by computing softmax tile-by-tile and
never materializing the full score matrix.

This tutorial demonstrates the problem and the fix:

  1. Baseline: a hand-written ``softmax(QK^T/sqrt(d))V`` that explicitly
     materializes the ``[B, H, N, N]`` tensor. We sweep
     ``seq_len in {256, 1024, 4096}`` x ``batch in {4, 16, 64}`` and let
     the naive path OOM on high-seq / high-batch configs. **The OOM is
     the lesson**: we catch it, log it, and move on.

  2. Optimized: the same sweep, but via
     ``torch.nn.functional.scaled_dot_product_attention`` (SDPA), which
     ships with PyTorch 2.x and automatically dispatches to a flash or
     memory-efficient kernel. No external flash-attn or xFormers install
     is required (that is the whole point of D-06).

  3. Backend selection: we then re-run a single mid-size config under
     ``torch.nn.attention.sdpa_kernel`` forcing FLASH, EFFICIENT, and
     MATH explicitly, so you can see which kernel PyTorch picks and how
     much slower the ``MATH`` fallback is.

bf16 is used on SM >= 8.0 hardware so the FLASH backend is actually
exercised (FLASH requires fp16 or bf16). On older hardware we fall back
to fp16.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from utils import (
    benchmark,
    get_device,
    get_gpu_capability,
    print_benchmark_table,
    print_device_info,
    setup_logging,
)

logger = setup_logging("flash_attention_sdpa")

# Sweep matrix is locked by D-07 -- do not alter.
SEQ_LENS = [256, 1024, 4096]
BATCHES  = [4, 16, 64]
HEADS = 8
D_H = 64            # per-head dim; total model dim = HEADS * D_H = 512
NUM_ITERATIONS = 50
WARMUP_ITERATIONS = 10


def pick_dtype() -> torch.dtype:
    """Pick the best dtype for SDPA flash/efficient kernels.

    FLASH_ATTENTION and EFFICIENT_ATTENTION require fp16 or bf16 inputs.
    bf16 is preferred on SM >= 8.0 (Ampere+) for its larger dynamic range;
    on pre-Ampere CUDA hardware we fall back to fp16. On CPU we use fp32.
    """
    cap = get_gpu_capability()
    if cap is not None and cap[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Hand-written softmax(QK^T / sqrt(d)) V attention.

    Explicitly materializes the ``[B, H, N, N]`` score tensor -- this is
    the whole point of the "before" baseline. On high seq_len / batch
    configs this will OOM, and the caller must catch it.
    """
    d_h = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_h ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Optimized path via torch.nn.functional.scaled_dot_product_attention.

    PyTorch 2.x dispatches this to a flash / memory-efficient / math
    kernel based on dtype, hardware, and input shape. No external
    dependency required.
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sweep(device: torch.device, dtype: torch.dtype) -> tuple[list[dict], int]:
    """Run the naive vs SDPA sweep across (B, N) configurations.

    Returns a (results, oom_count) tuple. OOM on the naive path is
    expected on the high-seq / high-batch corner -- we catch it, log it
    as the lesson, and record a row with ``memory_mb=None`` so the
    benchmark table renders ``N/A``.
    """
    results: list[dict] = []
    oom_count = 0

    for B in BATCHES:
        for N in SEQ_LENS:
            print(f"\n--- Config: B={B}, N={N}, H={HEADS}, d_h={D_H} ---")
            q = torch.randn(B, HEADS, N, D_H, device=device, dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Warmup: naive. OOM during warmup is possible on huge configs;
            # handle it so the script doesn't abort before SDPA gets a chance.
            naive_warm_ok = True
            try:
                for _ in range(WARMUP_ITERATIONS):
                    naive_attention(q, k, v)
                _sync_if_cuda()
            except torch.cuda.OutOfMemoryError:
                _free_cuda()
                naive_warm_ok = False
                logger.info(
                    "naive warmup OOM at B=%d, N=%d -- skipping measured naive run",
                    B, N,
                )

            # Measured naive run (only if warmup succeeded).
            if naive_warm_ok:
                @benchmark
                def run_naive():
                    for _ in range(NUM_ITERATIONS):
                        naive_attention(q, k, v)
                    _sync_if_cuda()

                try:
                    r = run_naive()
                    results.append({
                        "name": f"naive B={B:>2} N={N:>4}",
                        "time_seconds": r["time_seconds"],
                        "memory_mb": r["memory_mb"],
                    })
                    logger.info(
                        "naive B=%d N=%d -> %.4fs, peak %.1f MB",
                        B, N, r["time_seconds"],
                        r["memory_mb"] if r["memory_mb"] is not None else -1.0,
                    )
                except torch.cuda.OutOfMemoryError:
                    _free_cuda()
                    oom_count += 1
                    logger.info(
                        "naive OOM at B=%d, N=%d -- exactly the lesson SDPA avoids",
                        B, N,
                    )
                    results.append({
                        "name": f"naive B={B:>2} N={N:>4} (OOM)",
                        "time_seconds": 0.0,
                        "memory_mb": None,
                    })
            else:
                oom_count += 1
                results.append({
                    "name": f"naive B={B:>2} N={N:>4} (OOM)",
                    "time_seconds": 0.0,
                    "memory_mb": None,
                })

            # Reset peak memory stats so SDPA measurement isn't polluted
            # by the naive peak in the same process.
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            _free_cuda()

            # Warmup: SDPA.
            try:
                for _ in range(WARMUP_ITERATIONS):
                    sdpa_attention(q, k, v)
                _sync_if_cuda()
            except torch.cuda.OutOfMemoryError:
                _free_cuda()
                logger.info(
                    "sdpa warmup OOM at B=%d, N=%d -- unexpected, skipping",
                    B, N,
                )
                del q, k, v
                _free_cuda()
                continue

            # Measured SDPA run.
            @benchmark
            def run_sdpa():
                for _ in range(NUM_ITERATIONS):
                    sdpa_attention(q, k, v)
                _sync_if_cuda()

            try:
                r = run_sdpa()
                results.append({
                    "name": f"sdpa  B={B:>2} N={N:>4}",
                    "time_seconds": r["time_seconds"],
                    "memory_mb": r["memory_mb"],
                })
                logger.info(
                    "sdpa  B=%d N=%d -> %.4fs, peak %.1f MB",
                    B, N, r["time_seconds"],
                    r["memory_mb"] if r["memory_mb"] is not None else -1.0,
                )
            except torch.cuda.OutOfMemoryError:
                _free_cuda()
                logger.info(
                    "sdpa OOM at B=%d, N=%d -- unexpected but handled",
                    B, N,
                )
                results.append({
                    "name": f"sdpa  B={B:>2} N={N:>4} (OOM)",
                    "time_seconds": 0.0,
                    "memory_mb": None,
                })

            # Free tensors between configs (Pitfall 4).
            del q, k, v
            _free_cuda()

    return results, oom_count


def run_explicit_backends(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict], list[str]]:
    """Force specific SDPA backends via sdpa_kernel and compare.

    Some backends may not be available on all hardware (e.g., FLASH
    requires SM >= 7.5 and fp16/bf16, CUDNN_ATTENTION requires cuDNN
    support). We wrap each in try/except so an unavailable backend just
    gets logged and skipped.
    """
    B, N = 16, 1024
    print(f"\n--- Explicit backend comparison: B={B}, N={N} ---")
    q = torch.randn(B, HEADS, N, D_H, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    backends = [
        ("FLASH", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
        ("MATH", SDPBackend.MATH),
    ]

    rows: list[dict] = []
    skipped: list[str] = []

    for label, backend in backends:
        try:
            # Warmup under the forced backend.
            with sdpa_kernel(backend):
                for _ in range(WARMUP_ITERATIONS):
                    F.scaled_dot_product_attention(q, k, v, is_causal=False)
                _sync_if_cuda()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            @benchmark
            def run_forced():
                with sdpa_kernel(backend):
                    for _ in range(NUM_ITERATIONS):
                        F.scaled_dot_product_attention(q, k, v, is_causal=False)
                    _sync_if_cuda()

            r = run_forced()
            rows.append({
                "name": f"sdpa[{label}] B={B} N={N}",
                "time_seconds": r["time_seconds"],
                "memory_mb": r["memory_mb"],
            })
            logger.info(
                "backend=%s -> %.4fs, peak %.1f MB",
                label, r["time_seconds"],
                r["memory_mb"] if r["memory_mb"] is not None else -1.0,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            _free_cuda()
            skipped.append(label)
            logger.info("backend=%s unavailable or failed: %s", label, exc)

    del q, k, v
    _free_cuda()
    return rows, skipped


def main() -> None:
    print("\n" + "=" * 64)
    print("  Flash Attention: Naive vs SDPA")
    print("=" * 64 + "\n")

    device = get_device()
    print_device_info()

    cap = get_gpu_capability()
    dtype = pick_dtype()
    if cap is not None and cap[0] >= 8 and dtype == torch.bfloat16:
        logger.info(
            "dtype=bfloat16 selected: SM >= 8.0 available (cap=%s), "
            "FLASH backend requires fp16/bf16", cap,
        )
    elif dtype == torch.float16:
        logger.info(
            "dtype=float16 selected: cap=%s is pre-Ampere, bf16 not native", cap,
        )
    else:
        logger.info("dtype=float32 selected: CUDA unavailable, running on CPU")

    logger.info(
        "Sweep matrix: seq_lens=%s, batches=%s, heads=%d, d_h=%d",
        SEQ_LENS, BATCHES, HEADS, D_H,
    )
    logger.info(
        "Per-config iterations: %d measured (+ %d warmup)",
        NUM_ITERATIONS, WARMUP_ITERATIONS,
    )

    print("\n--- Section 1: Naive vs SDPA sweep ---")
    sweep_results, oom_count = run_sweep(device, dtype)

    print("\n--- Section 2: Explicit SDPA backend selection ---")
    backend_rows, skipped = run_explicit_backends(device, dtype)

    all_results = sweep_results + backend_rows

    print("\n--- Final benchmark table ---")
    print_benchmark_table(all_results)

    print("\n--- Summary ---\n")
    logger.info("naive OOM configs: %d of %d", oom_count, len(BATCHES) * len(SEQ_LENS))
    if skipped:
        logger.info("skipped backends (unavailable on this hardware): %s", skipped)
    else:
        logger.info("all explicit backends ran successfully")
    logger.info(
        "Takeaway: naive attention materializes [B, H, N, N] -> O(N^2) memory; "
        "SDPA avoids it via tiled flash/efficient kernels -- same math, bounded memory."
    )

    print("\n" + "=" * 64)
    print("  Tutorial Complete")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
