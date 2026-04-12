---
phase: 03-mixed-precision-training
plan: 02
subsystem: bf16-fp16-fp8-tutorials
tags: [mixed-precision, bf16, fp16, fp8, transformer-engine, comparison]
dependency_graph:
  requires: [03-01]
  provides: [bf16_vs_fp16_tutorial, fp8_transformer_engine_tutorial]
  affects: []
tech_stack:
  added: [torch.bfloat16, transformer_engine.pytorch, DelayedScaling, fp8_autocast]
  patterns: [tier-adaptive-gpu-detection, conditional-import-fallback, gradscaler-comparison]
key_files:
  created:
    - mixed_precision/bf16_vs_fp16.py
    - mixed_precision/fp8_transformer_engine.py
  modified: []
decisions:
  - BF16 tutorial uses SimpleCNN (same as amp_training.py) for direct comparison
  - FP8 tutorial uses SimpleViT per D-01 with TE TransformerLayer for FP8 path
  - Three-tier fallback (FP8 -> BF16 -> FP16) ensures tutorial runs on any GPU
metrics:
  duration_seconds: 163
  completed: "2026-04-12T23:30:21Z"
  tasks_completed: 2
  tasks_total: 2
---

# Phase 03 Plan 02: BF16 vs FP16 + FP8 Transformer Engine Tutorials Summary

BF16/FP16 numerical stability comparison with GradScaler demonstration, plus FP8 tutorial with three-tier GPU detection and conditional Transformer Engine import with graceful BF16/FP16 fallbacks.

## Task Results

| Task | Name | Commit | Files | Status |
|------|------|--------|-------|--------|
| 1 | Create BF16 vs FP16 comparison tutorial (PREC-02) | 17151e7 | mixed_precision/bf16_vs_fp16.py | Done |
| 2 | Create FP8 Transformer Engine tutorial (PREC-03) | f6c3e5e | mixed_precision/fp8_transformer_engine.py | Done |

## Decisions Made

1. **BF16 tutorial uses SimpleCNN**: Matches amp_training.py model for direct apples-to-apples comparison of FP16 vs BF16 on the same architecture.
2. **FP8 tutorial uses SimpleViT per D-01**: ViT architecture is the natural fit for Transformer Engine FP8 and dimensions are TE-compatible (divisible by 16).
3. **Three-tier GPU fallback**: get_gpu_tier() detects SM capability and selects FP8 (SM 8.9+), BF16 (SM 8.0+), or FP16 (SM 7.x), ensuring every user gets a runnable tutorial.
4. **Conditional TE import**: try_import_transformer_engine() gracefully handles missing Transformer Engine, logging install instructions and falling back to BF16/FP16.

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

- `python mixed_precision/bf16_vs_fp16.py` -- exits 0, shows numerical stability demo + benchmark table
- `python mixed_precision/fp8_transformer_engine.py` -- exits 0 on RTX 2070 (SM 7.5), runs FP16 fallback path
- bf16_vs_fp16.py: 249 lines (requirement: >150)
- fp8_transformer_engine.py: 357 lines (requirement: >180)
- Both tutorials use modern torch.amp API (no deprecated torch.cuda.amp)
- BF16 section has NO GradScaler (critical teaching point verified)
- FP8 tutorial uses SimpleViT and get_gpu_capability from utils

## Known Stubs

None.

## Self-Check: PASSED
