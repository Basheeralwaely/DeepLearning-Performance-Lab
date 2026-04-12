---
phase: 02-profiling-diagnostics
plan: 01
subsystem: profiling
tags: [pytorch-profiler, memory-profiling, gpu-memory, oom-recovery, gradient-checkpointing, chrome-trace]
dependency_graph:
  requires: [utils/models.py, utils/benchmark.py, utils/__init__.py, utils/device.py]
  provides: [profiling/pytorch_profiler.py, profiling/memory_profiling.py, .gitignore]
  affects: [profiler_output/]
tech_stack:
  added: [torch.profiler, torch.utils.checkpoint]
  patterns: [profiler-schedule, memory-lifecycle-tracking, oom-recovery-pattern, gradient-checkpointing]
key_files:
  created:
    - .gitignore
    - profiling/pytorch_profiler.py
    - profiling/memory_profiling.py
  modified: []
decisions:
  - Used batch_size=32 for gradient checkpointing comparison (reduced from 64) to fit RTX 2070 memory
  - Caught RuntimeError alongside OutOfMemoryError for CUBLAS initialization failures under memory pressure
metrics:
  duration: "~10 minutes"
  completed: "2026-04-12"
  tasks_completed: 2
  tasks_total: 2
  files_created: 3
  files_modified: 0
requirements:
  - PROF-01
  - PROF-02
---

# Phase 02 Plan 01: PyTorch Profiler & Memory Profiling Tutorials Summary

Two profiling tutorials teaching GPU bottleneck identification via torch.profiler (with Chrome trace export and console tables) and GPU memory lifecycle tracking with OOM recovery and gradient checkpointing comparison.

## What Was Built

### Task 1: .gitignore and PyTorch Profiler Tutorial (PROF-01)

**Commit:** `4e52adb`

Created `.gitignore` with entries for `__pycache__/`, `profiler_output/`, IDE files, and Python build artifacts.

Created `profiling/pytorch_profiler.py` with 6 sections:
1. **Setup** -- SimpleCNN model, SGD optimizer, CrossEntropyLoss
2. **Profile a Training Step** -- torch.profiler with schedule (wait/warmup/active/repeat), record_function context, Chrome trace export
3. **Analyze Console Output** -- Three profiler tables sorted by cuda_time_total, self_cpu_memory_usage, and grouped by input shape
4. **Inference vs Training Benchmark** -- @benchmark decorated comparison using print_benchmark_table
5. **Chrome Trace Instructions** -- How to open profiler_output/training_trace.json in chrome://tracing
6. **Key Takeaways** -- Schedule phases, sort_by options, memory differences, when to use traces vs tables

### Task 2: GPU Memory Profiling Tutorial (PROF-02)

**Commit:** `eaf0ab1`

Created `profiling/memory_profiling.py` with 6 sections:
1. **Setup** -- Device info, total GPU memory display
2. **Memory Lifecycle** -- Tracks allocated/reserved/peak MB at each training stage (model creation, input allocation, forward, backward, optimizer step, zero_grad), plus memory_summary table
3. **Batch Size Scaling** -- Sweeps batch sizes [8, 16, 32, 64, 128] with 224x224 inputs, recording peak memory at each size, with OOM/CUBLAS error handling
4. **OOM Recovery** -- Demonstrates try/except pattern with batch_size=256, cache clearing, and retry at half batch size
5. **Gradient Checkpointing** -- Compares peak memory with and without torch.utils.checkpoint on the features extractor using use_reentrant=False
6. **Key Takeaways** -- Memory stages, batch size as primary knob, OOM recovery pattern, checkpointing tradeoff, which monitoring tools to use when

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed torch.cuda.get_device_properties attribute name**
- **Found during:** Task 2
- **Issue:** Plan specified `total_mem` but the correct PyTorch attribute is `total_memory`
- **Fix:** Changed `total_mem` to `total_memory`
- **Files modified:** profiling/memory_profiling.py

**2. [Rule 1 - Bug] Added RuntimeError handling alongside OutOfMemoryError**
- **Found during:** Task 2
- **Issue:** CUBLAS operations can throw RuntimeError (CUBLAS_STATUS_NOT_INITIALIZED) under memory pressure instead of OutOfMemoryError
- **Fix:** Changed `except torch.cuda.OutOfMemoryError` to `except (torch.cuda.OutOfMemoryError, RuntimeError)` in sections 3, 4, and 5
- **Files modified:** profiling/memory_profiling.py

**3. [Rule 1 - Bug] Reduced checkpointing comparison batch size from 64 to 32**
- **Found during:** Task 2
- **Issue:** batch_size=64 with 224x224 inputs exceeds RTX 2070 memory during backward pass
- **Fix:** Reduced ckpt_batch_size to 32 and wrapped section 5 in try/except for safety
- **Files modified:** profiling/memory_profiling.py

**4. [Rule 2 - Missing critical functionality] Added CUBLAS warmup and GPU synchronization**
- **Found during:** Task 2
- **Issue:** Deleting model/tensors without synchronizing GPU first caused CUBLAS handle corruption in subsequent operations
- **Fix:** Added `torch.cuda.synchronize()` before cleanup and CUBLAS warmup matmul before section 3
- **Files modified:** profiling/memory_profiling.py

## Verification Results

- Both tutorials parse without syntax errors (verified via ast.parse)
- pytorch_profiler.py runs to completion, produces profiler tables with cuda_time_total sorting, exports Chrome trace to profiler_output/training_trace.json
- memory_profiling.py runs to completion, shows Allocated/Reserved/Peak MB at each training stage, handles OOM gracefully, compares checkpointing impact
- .gitignore contains profiler_output/ and __pycache__/ entries

## Known Stubs

None -- all sections produce real profiling data and measurements.

## Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | .gitignore and PyTorch Profiler tutorial | `4e52adb` | .gitignore, profiling/pytorch_profiler.py |
| 2 | GPU memory profiling tutorial | `eaf0ab1` | profiling/memory_profiling.py |

## Self-Check: PASSED

- All 3 created files exist on disk
- Both commits (4e52adb, eaf0ab1) found in git log
- SUMMARY.md created at correct path
