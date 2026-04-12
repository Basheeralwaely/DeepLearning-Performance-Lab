---
status: partial
phase: 02-profiling-diagnostics
source: [02-VERIFICATION.md]
started: 2026-04-12T00:00:00Z
updated: 2026-04-12T00:00:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. PyTorch Profiler console tables and trace export
expected: Run `python profiling/pytorch_profiler.py`, confirm three profiler tables appear and `profiler_output/training_trace.json` is created
result: [pending]

### 2. GPU memory lifecycle and OOM recovery
expected: Run `python profiling/memory_profiling.py`, confirm Allocated/Peak MB logs appear at each training stage and OOM recovery executes
result: [pending]

### 3. DataLoader throughput sweep
expected: Run `python profiling/dataloader_tuning.py`, confirm benchmark table shows visible throughput differences across 4 worker configs
result: [pending]

### 4. torch.compile mode comparison
expected: Run `python profiling/torch_compile_diagnostics.py`, confirm all 3 compile modes benchmarked and graph break count is 0 for SimpleCNN, >0 for ModelWithGraphBreaks
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps
