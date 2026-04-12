---
phase: 02-profiling-diagnostics
plan: 02
subsystem: profiling
tags: [dataloader, torch-compile, diagnostics, performance-tuning, graph-breaks]
dependency_graph:
  requires: [utils/models.py, utils/benchmark.py, utils/__init__.py]
  provides: [profiling/dataloader_tuning.py, profiling/torch_compile_diagnostics.py]
  affects: []
tech_stack:
  added: [torch._dynamo, torch.compile, multiprocessing, torch.utils.data.DataLoader]
  patterns: [synthetic-dataset-with-io-delay, num-workers-sweep, compile-mode-comparison, graph-break-detection]
key_files:
  created:
    - profiling/dataloader_tuning.py
    - profiling/torch_compile_diagnostics.py
  modified: []
decisions:
  - Used synthetic I/O dataset with configurable delay for reproducible DataLoader benchmarks (avoids requiring real dataset on disk)
  - Measured compilation overhead separately from runtime by timing first forward pass independently
  - Used ModelWithGraphBreaks with data-dependent control flow to demonstrate dynamo graph break detection
metrics:
  duration: 4m 12s
  completed: "2026-04-12T21:33:46Z"
  tasks: 2/2
  files_created: 2
  total_lines: 781
---

# Phase 02 Plan 02: DataLoader Tuning & torch.compile Diagnostics Summary

DataLoader throughput tuning with synthetic I/O-delayed dataset sweeping num_workers/pin_memory/prefetch_factor, and torch.compile diagnostics with mode comparison and graph break detection via dynamo.explain.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | DataLoader tuning tutorial (PROF-03) | 8fb05ca | profiling/dataloader_tuning.py |
| 2 | torch.compile diagnostics tutorial (PROF-04) | 6d30371 | profiling/torch_compile_diagnostics.py |

## What Was Built

### profiling/dataloader_tuning.py (396 lines)

- **SyntheticIODataset**: Custom Dataset with configurable `time.sleep()` I/O delay per sample for reproducible bottleneck measurement
- **num_workers sweep**: Tests 4 configurations [0, 2, 4, 8] measuring batches/sec throughput
- **pin_memory comparison**: True vs False with benchmark table showing transfer speed impact
- **prefetch_factor comparison**: Values [1, 2, 4] with throughput measurement
- **End-to-end training pipeline**: Full training step (forward + backward) comparing unoptimized (workers=0) vs tuned DataLoader using SimpleCNN
- Produces benchmark tables via `print_benchmark_table()` and `compare_results()`

### profiling/torch_compile_diagnostics.py (385 lines)

- **Eager baseline**: Uncompiled model measurement as reference point
- **Compilation overhead**: First-pass timing separated from runtime for each of 3 modes (default, reduce-overhead, max-autotune)
- **Mode comparison**: All 3 compile modes benchmarked against eager with speedup calculation
- **Graph break detection**: `dynamo.explain()` on clean SimpleCNN showing 0 breaks (fully traceable)
- **Intentional graph breaks**: `ModelWithGraphBreaks` with `if x.sum() > 0` data-dependent control flow, showing 1 graph break detected with reason
- **Clean vs broken comparison**: Side-by-side compile speedup showing impact of graph breaks

## Verification Results

Both tutorials:
- Parse without syntax errors (ast.parse verified)
- Run to completion producing expected output
- DataLoader tutorial shows throughput for all 4 num_workers configs with "workers=" rows in benchmark table
- torch.compile tutorial shows all 3 compile modes vs eager, detects 0 graph breaks in clean model and 1 in broken model

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

1. **Synthetic I/O dataset approach**: Used `time.sleep(0.01)` per sample rather than real disk I/O. This makes benchmarks reproducible across machines without requiring a real dataset, and the I/O delay is controllable.

2. **Separate compilation timing**: First forward pass timed independently using `time.perf_counter()` to isolate JIT compilation overhead from steady-state runtime.

3. **ModelWithGraphBreaks design**: Used `if x.sum() > 0` as the graph break trigger -- this is the simplest and most common real-world pattern that causes graph breaks (data-dependent branching).
