---
phase: 03-mixed-precision-training
reviewed: 2026-04-12T12:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - mixed_precision/amp_training.py
  - mixed_precision/bf16_vs_fp16.py
  - mixed_precision/fp8_transformer_engine.py
  - utils/device.py
  - utils/__init__.py
  - utils/models.py
findings:
  critical: 3
  warning: 1
  info: 1
  total: 5
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-04-12T12:00:00Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

The mixed precision tutorials are well-structured, thoroughly documented, and demonstrate clear pedagogical value. The utility modules (device.py, models.py, __init__.py) are clean and well-typed.

The main concern is a recurring pattern across all three tutorials: `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')` are hardcoded with the `'cuda'` device string, which will raise a RuntimeError when running on CPU. Each tutorial includes explicit CPU-fallback warnings (suggesting the code will still run on CPU), but the hardcoded CUDA device strings prevent that from actually working.

## Critical Issues

### CR-01: GradScaler('cuda') crashes on CPU in amp_training.py

**File:** `mixed_precision/amp_training.py:130`
**Issue:** `torch.amp.GradScaler('cuda')` is created unconditionally. On CPU, this will raise a RuntimeError. Lines 58-60 warn that CPU will not show speedup but imply the code will still run. The scaler and autocast on line 136 both hardcode `'cuda'`.
**Fix:**
```python
# Line 130: Guard scaler creation
scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

# Line 133-144: Conditionally use autocast and scaler
@benchmark
def amp_training():
    for _ in range(NUM_ITERATIONS):
        optimizer.zero_grad()
        with torch.amp.autocast(device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
```

### CR-02: GradScaler('cuda') and autocast('cuda') crash on CPU in bf16_vs_fp16.py

**File:** `mixed_precision/bf16_vs_fp16.py:122,144,150,180`
**Issue:** Same hardcoded `'cuda'` pattern across multiple locations. Line 122 uses `torch.amp.autocast('cuda')` in warmup. Line 144 creates `GradScaler('cuda')`. Lines 150 and 180 use `autocast('cuda')`. All crash on CPU despite the warning on lines 61-62 implying CPU execution is supported.
**Fix:** Use `device.type` instead of `'cuda'` string literal. Guard GradScaler creation behind a `device.type == "cuda"` check. For autocast, use `enabled=(device.type == "cuda")` parameter to make it a no-op on CPU:
```python
with torch.amp.autocast(device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
```

### CR-03: GradScaler('cuda') and autocast('cuda') crash on CPU in fp8_transformer_engine.py

**File:** `mixed_precision/fp8_transformer_engine.py:272,293,299`
**Issue:** Same hardcoded `'cuda'` pattern. The bf16 path (line 272) and fp16 path (lines 293, 299) both hardcode `'cuda'`. On CPU the `get_gpu_tier()` function returns `"fp16"` tier, so the fp16 code path will execute and immediately crash on `GradScaler('cuda')`.
**Fix:** Same pattern as CR-01 and CR-02 -- use `device.type` and guard scaler creation.

## Warnings

### WR-01: Potential division by zero when computing throughput

**File:** `mixed_precision/fp8_transformer_engine.py:320`
**Issue:** `opt_throughput = NUM_ITERATIONS * BATCH_SIZE / optimized["time_seconds"]` executes before the zero-check on line 324. If `time_seconds` were ever 0, this line would raise `ZeroDivisionError`. The same unguarded pattern appears in `bf16_vs_fp16.py:161` and `bf16_vs_fp16.py:191`. While extremely unlikely in practice (benchmarks always take measurable time), the inconsistency with the existing guard on line 324 suggests the intent was to be defensive.
**Fix:**
```python
if optimized["time_seconds"] > 0:
    opt_throughput = NUM_ITERATIONS * BATCH_SIZE / optimized["time_seconds"]
    logger.info(f"{tier.upper()} throughput:            {opt_throughput:.0f} samples/sec")
```

## Info

### IN-01: Warmup uses model state that is then discarded

**File:** `mixed_precision/amp_training.py:80-88`
**Issue:** The warmup loop trains a model instance, then lines 101-102 create a fresh `SimpleCNN` and optimizer for the FP32 benchmark. The warmup is effective for CUDA kernel compilation but the trained weights are thrown away. This is intentional (fair comparison) but worth a brief inline comment to prevent confusion. The same pattern in `bf16_vs_fp16.py:117-128` and `fp8_transformer_engine.py:137-149` would also benefit from a clarifying comment.
**Fix:** Add a one-line comment before model re-initialization:
```python
# Re-initialize model with fresh weights for fair comparison (warmup only primes CUDA kernels)
model = SimpleCNN().to(device)
```

---

_Reviewed: 2026-04-12T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
