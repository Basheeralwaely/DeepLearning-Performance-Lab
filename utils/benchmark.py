"""Benchmark timing and comparison utilities for performance tutorials.

Provides a decorator for measuring execution time and GPU memory usage,
plus formatted comparison tables that show before/after results for each
performance technique.
"""

import functools
import time
from typing import Any, Callable, Optional

import torch


def benchmark(func: Callable) -> Callable:
    """Decorator that measures wall-clock time and optional GPU memory usage.

    Wraps a function to track execution time using time.perf_counter and,
    when CUDA is available, peak GPU memory via torch.cuda.max_memory_allocated.

    Args:
        func: The function to benchmark.

    Returns:
        A wrapped function that returns a dict with keys:
            - "result": the original function's return value
            - "time_seconds": wall-clock time in seconds (float)
            - "memory_mb": peak GPU memory in MB (float), or None if no CUDA
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = func(*args, **kwargs)

        if cuda_available:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        memory_mb: Optional[float] = None
        if cuda_available:
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return {
            "result": result,
            "time_seconds": elapsed,
            "memory_mb": memory_mb,
        }

    return wrapper


def compare_results(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
    technique_name: str,
) -> None:
    """Print a formatted comparison table between baseline and optimized runs.

    Args:
        baseline: Dict from a @benchmark-decorated function call (baseline run).
        optimized: Dict from a @benchmark-decorated function call (optimized run).
        technique_name: Name of the technique being compared.
    """
    b_time = baseline["time_seconds"]
    o_time = optimized["time_seconds"]
    speedup = b_time / o_time if o_time > 0 else float("inf")

    b_mem = baseline.get("memory_mb")
    o_mem = optimized.get("memory_mb")

    header = f" Benchmark Results: {technique_name} "

    print()
    print(f"+{'-' * 58}+")
    print(f"|{header:<58}|")
    print(f"+{'-' * 14}+{'-' * 14}+{'-' * 14}+{'-' * 14}+")
    print(f"| {'Metric':<12} | {'Baseline':>12} | {'Optimized':>12} | {'Change':>12} |")
    print(f"+{'-' * 14}+{'-' * 14}+{'-' * 14}+{'-' * 14}+")

    print(
        f"| {'Time (s)':<12} | {b_time:>12.4f} | {o_time:>12.4f} | {speedup:>11.2f}x |"
    )

    if b_mem is not None and o_mem is not None:
        mem_diff = o_mem - b_mem
        sign = "+" if mem_diff >= 0 else ""
        print(
            f"| {'Memory (MB)':<12} | {b_mem:>12.1f} | {o_mem:>12.1f} | {sign}{mem_diff:>10.1f} |"
        )

    print(f"+{'-' * 14}+{'-' * 14}+{'-' * 14}+{'-' * 14}+")
    print()


def print_benchmark_table(results: list[dict[str, Any]]) -> None:
    """Print a multi-row benchmark table for comparing several configurations.

    Each result dict should contain:
        - "name": str - label for this configuration
        - "time_seconds": float - execution time
        - "memory_mb": float or None - GPU memory usage

    Args:
        results: List of benchmark result dicts, each with "name",
                 "time_seconds", and optionally "memory_mb".
    """
    if not results:
        print("No benchmark results to display.")
        return

    # Validate required keys up front
    for i, r in enumerate(results):
        if "name" not in r:
            raise ValueError(
                f"Result at index {i} is missing required key 'name'. "
                f"Got keys: {list(r.keys())}"
            )
        if "time_seconds" not in r:
            raise ValueError(
                f"Result at index {i} is missing required key 'time_seconds'."
            )

    has_memory = any(r.get("memory_mb") is not None for r in results)

    print()
    if has_memory:
        print(f"+{'-' * 26}+{'-' * 14}+{'-' * 14}+")
        print(f"| {'Configuration':<24} | {'Time (s)':>12} | {'Memory (MB)':>12} |")
        print(f"+{'-' * 26}+{'-' * 14}+{'-' * 14}+")
        for r in results:
            mem = r.get("memory_mb")
            mem_str = f"{mem:>12.1f}" if mem is not None else f"{'N/A':>12}"
            print(
                f"| {r['name']:<24} | {r['time_seconds']:>12.4f} | {mem_str} |"
            )
        print(f"+{'-' * 26}+{'-' * 14}+{'-' * 14}+")
    else:
        print(f"+{'-' * 26}+{'-' * 14}+")
        print(f"| {'Configuration':<24} | {'Time (s)':>12} |")
        print(f"+{'-' * 26}+{'-' * 14}+")
        for r in results:
            print(f"| {r['name']:<24} | {r['time_seconds']:>12.4f} |")
        print(f"+{'-' * 26}+{'-' * 14}+")
    print()
