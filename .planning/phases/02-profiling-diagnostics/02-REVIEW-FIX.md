---
phase: 02-profiling-diagnostics
fixed_at: 2026-04-12T12:15:00Z
review_path: .planning/phases/02-profiling-diagnostics/02-REVIEW.md
iteration: 1
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 02: Code Review Fix Report

**Fixed at:** 2026-04-12T12:15:00Z
**Source review:** .planning/phases/02-profiling-diagnostics/02-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 3
- Fixed: 3
- Skipped: 0

## Fixed Issues

### WR-01: Incomplete tensor cleanup in OOM exception handler

**Files modified:** `profiling/memory_profiling.py`
**Commit:** 0cebc62
**Applied fix:** Expanded the OOM catch block to delete all seven potentially-allocated GPU tensors (`m`, `opt`, `crit`, `inp`, `lbl`, `out`, `l`) before calling `torch.cuda.empty_cache()`. Each deletion is wrapped in its own try/except NameError to handle the case where OOM occurred before the variable was assigned. The previous code only deleted `m`, leaving other tensors referenced and preventing full GPU memory recovery.

### WR-02: Misleading cross-architecture speedup comparison

**Files modified:** `profiling/torch_compile_diagnostics.py`
**Commit:** dc6f36b
**Applied fix:** Added a dedicated eager baseline measurement for `ModelWithGraphBreaks` before the compiled benchmarks. The broken model's compile speedup is now computed against its own eager time (`broken_eager_time`) rather than the `eager_time` from `SimpleCNN`. This gives an apples-to-apples comparison that isolates the effect of graph breaks from architectural differences between the two models. Updated the log message to read "vs its own eager" for clarity.

### WR-03: Relative path for profiler output directory

**Files modified:** `profiling/pytorch_profiler.py`
**Commit:** 6c20df3
**Applied fix:** Replaced the relative path `"profiler_output"` with a path anchored to the script's directory using `os.path.dirname(os.path.abspath(__file__))`. The trace output directory is now created at `{script_dir}/profiler_output/` regardless of the user's current working directory, making the tutorial behave consistently whether run as `python profiling/pytorch_profiler.py` from the repo root or `python pytorch_profiler.py` from inside `profiling/`.

---

_Fixed: 2026-04-12T12:15:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
