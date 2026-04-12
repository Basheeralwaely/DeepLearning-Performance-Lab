---
phase: 02-profiling-diagnostics
reviewed: 2026-04-12T12:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - .gitignore
  - profiling/pytorch_profiler.py
  - profiling/memory_profiling.py
  - profiling/dataloader_tuning.py
  - profiling/torch_compile_diagnostics.py
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-04-12T12:00:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Reviewed the four new profiling tutorials and the updated `.gitignore`. The tutorials are well-structured, follow project conventions (standalone `.py` files, rich logging, educational inline commentary), and correctly use shared utilities from the `utils` package. No security issues or critical bugs were found.

Three warnings were identified: incomplete GPU tensor cleanup in an OOM handler that could prevent memory recovery, a misleading cross-architecture benchmark comparison, and use of relative paths for file output. Two informational items note minor code quality improvements.

## Warnings

### WR-01: Incomplete tensor cleanup in OOM exception handler

**File:** `profiling/memory_profiling.py:209-215`
**Issue:** The OOM catch block in the batch size sweep only deletes `m` but not `opt`, `crit`, `inp`, `lbl`, `out`, or `l`. If OOM occurs after some of these GPU tensors are allocated (e.g., during the forward pass on line 186), they remain referenced in the local scope, preventing `torch.cuda.empty_cache()` from reclaiming that GPU memory. This defeats the purpose of the OOM recovery pattern the tutorial is demonstrating.
**Fix:**
```python
except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    logger.warning(f"OOM/CUDA error at batch_size={bs}: {type(e).__name__}")
    logger.warning("Clearing cache and stopping batch size sweep.")
    # Delete all potentially-allocated tensors before clearing cache
    for var_name in ('m', 'opt', 'crit', 'inp', 'lbl', 'out', 'l'):
        if var_name in locals():
            del locals()[var_name]
    torch.cuda.empty_cache()
    break
```
Or more explicitly, use the same pattern as line 205 with a try/except around the del:
```python
except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    logger.warning(f"OOM/CUDA error at batch_size={bs}: {type(e).__name__}")
    logger.warning("Clearing cache and stopping batch size sweep.")
    # Clean up any tensors that were allocated before the OOM
    for name in ['m', 'opt', 'crit', 'inp', 'lbl', 'out', 'l']:
        obj = locals().get(name)
        if obj is not None:
            del obj
    torch.cuda.empty_cache()
    break
```

### WR-02: Misleading cross-architecture speedup comparison

**File:** `profiling/torch_compile_diagnostics.py:340-343`
**Issue:** `broken_speedup` divides `eager_time` (measured on `SimpleCNN`) by `broken_result["time_seconds"]` (measured on `ModelWithGraphBreaks`). These are architecturally different models (CNN vs MLP with branching), so the "speedup vs eager" ratio conflates architectural differences with the effect of graph breaks. The comparison suggests graph breaks cause a slowdown relative to eager SimpleCNN, when the meaningful comparison is whether compilation helps the broken model relative to its own eager baseline.
**Fix:** Measure an eager baseline for `ModelWithGraphBreaks` and use that for the broken model's speedup calculation:
```python
# Add eager baseline for broken model
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

# Then use broken_eager_time for the broken speedup
broken_speedup = broken_eager_time / broken_result["time_seconds"] if broken_result["time_seconds"] > 0 else float("inf")
logger.info(f"  Broken model compile speedup vs its own eager: {broken_speedup:.2f}x")
```

### WR-03: Relative path for profiler output directory

**File:** `profiling/pytorch_profiler.py:87-88`
**Issue:** `os.makedirs("profiler_output", exist_ok=True)` uses a relative path, so the output directory is created relative to the current working directory at runtime, not relative to the script. If a user runs the tutorial from a different directory (e.g., `python profiling/pytorch_profiler.py` from the repo root vs `python pytorch_profiler.py` from `profiling/`), the trace file lands in different locations. This is inconsistent with the project constraint that tutorials should be independently runnable.
**Fix:** Anchor the output path relative to the script's location:
```python
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "profiler_output")
os.makedirs(output_dir, exist_ok=True)
trace_path = os.path.join(output_dir, "training_trace.json")
```

## Info

### IN-01: Broad RuntimeError catch could mask non-CUDA errors

**File:** `profiling/memory_profiling.py:209`
**Issue:** The except clause catches both `torch.cuda.OutOfMemoryError` and generic `RuntimeError`. While CUDA errors sometimes surface as `RuntimeError` in older PyTorch versions, this also catches unrelated RuntimeErrors (e.g., shape mismatches, invalid arguments), which would be silently swallowed as "OOM" and cause the sweep to stop with a misleading log message. The same pattern appears at lines 267 and 293.
**Fix:** Consider checking the error message for CUDA-related keywords before treating it as OOM:
```python
except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
    if isinstance(e, RuntimeError) and "CUDA" not in str(e) and "out of memory" not in str(e):
        raise  # Re-raise non-CUDA RuntimeErrors
    logger.warning(f"OOM at batch_size={bs}: {type(e).__name__}")
    ...
```

### IN-02: Unused import

**File:** `profiling/memory_profiling.py:28`
**Issue:** `import time` is used only for manual `time.perf_counter()` calls in Section 3's batch size sweep (lines 184, 194), while the rest of the file uses the `@benchmark` decorator for timing. This is not wrong -- the manual timing is intentional for measuring just the forward/backward portion -- but it may confuse readers expecting consistency. A brief inline comment would clarify the intent.
**Fix:** Add a comment at the usage site:
```python
# Manual timing here (not @benchmark) to measure only forward+backward, excluding setup
start_time = time.perf_counter()
```

---

_Reviewed: 2026-04-12T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
