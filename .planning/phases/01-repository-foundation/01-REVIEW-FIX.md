---
phase: 01-repository-foundation
fixed_at: 2026-04-12T00:00:00Z
review_path: .planning/phases/01-repository-foundation/01-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 01: Code Review Fix Report

**Fixed at:** 2026-04-12
**Source review:** .planning/phases/01-repository-foundation/01-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 4
- Fixed: 4
- Skipped: 0

## Fixed Issues

### WR-01: `get_sample_batch` hardcodes label range to 10, ignoring `num_classes`

**Files modified:** `utils/models.py`
**Commit:** c2daf20
**Applied fix:** Added `num_classes: int = 10` parameter to `get_sample_batch`. The label generation now uses `torch.randint(0, num_classes, ...)` instead of the hardcoded `0, 10`. Updated docstring to document the new parameter. Default value of 10 preserves backward compatibility.

---

### WR-02: `SimpleCNN` silently produces zero-sized linear layer for small `input_size`

**Files modified:** `utils/models.py`
**Commit:** c1145ea
**Applied fix:** Added a guard after `reduced_size = input_size // 8` that raises a `ValueError` with a clear message when `reduced_size == 0` (i.e., `input_size < 8`). This converts a silent correctness failure into an immediate, informative error.

---

### WR-03: `print_benchmark_table` raises `KeyError` on missing `"name"` key

**Files modified:** `utils/benchmark.py`
**Commit:** 69b434b
**Applied fix:** Added an early validation loop after the empty-list check in `print_benchmark_table`. Each result dict is verified to contain both `"name"` and `"time_seconds"` before any printing occurs. Missing keys raise a `ValueError` with the index and actual keys present, replacing the opaque `KeyError`.

---

### WR-04: No GPU warm-up before baseline benchmark inflates measured speedup

**Files modified:** `profiling/reference_tutorial.py`
**Commit:** 1264fcf
**Applied fix:** Added a 10-iteration warm-up block using `torch.inference_mode()` immediately after model and data setup, before the timed baseline section. Includes `torch.cuda.synchronize()` when running on CUDA. This ensures CUDA kernel compilation overhead is excluded from both timed runs, producing accurate speedup measurements.

---

_Fixed: 2026-04-12_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
