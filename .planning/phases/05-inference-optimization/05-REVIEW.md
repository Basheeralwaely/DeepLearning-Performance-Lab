---
phase: 05-inference-optimization
reviewed: 2026-04-13T12:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - inference/onnx_inference.py
  - inference/tensorrt_inference.py
  - inference/torchscript_inference.py
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-04-13T12:00:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Three inference optimization tutorials were reviewed: ONNX Runtime, TensorRT, and TorchScript. The code is well-structured, thoroughly documented, and follows the project conventions (standalone `.py` files with rich logging). No security issues or critical bugs were found. The main concerns are around missing CUDA synchronization before timing measurements and a potential shape mismatch in the TensorRT tutorial when the ONNX model is exported with batch size 1 but inference runs with batch size 64 without dynamic axes.

## Warnings

### WR-01: ONNX export uses fixed batch_size=1 but inference uses batch_size=64 without dynamic axes (TensorRT)

**File:** `inference/tensorrt_inference.py:138-148`
**Issue:** The ONNX model is exported with `dummy_input = torch.randn(1, 3, 32, 32)` and `dynamic_axes=None` (line 148). However, the TensorRT engine is then set up to run inference with `BATCH_SIZE=64` (line 247). While TensorRT's `set_input_shape` can sometimes handle this, the ONNX model itself has the batch dimension baked in as 1. This works because TensorRT re-interprets the shapes during engine build, but it is fragile and could cause silent shape errors with different TensorRT versions.
**Fix:** Either export with the target batch size, or enable dynamic axes for the batch dimension:
```python
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
```

### WR-02: Same batch-size mismatch in ONNX inference tutorial

**File:** `inference/onnx_inference.py:167-177`
**Issue:** The ONNX model is exported with `dummy_input = torch.randn(1, 3, 32, 32)` (batch=1) but ORT inference uses `input_data = np.random.randn(BATCH_SIZE, 3, 32, 32)` where `BATCH_SIZE=64` (line 258). ONNX Runtime is flexible enough to handle mismatched batch dimensions for most models, but this mismatch between the exported shape and runtime shape is a latent correctness risk. If ORT strict mode or certain providers enforce shape checks, this could fail.
**Fix:** Add `dynamic_axes` to the export call:
```python
torch.onnx.export(
    model_cpu,
    dummy_input,
    onnx_path,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
```

### WR-03: Missing CUDA synchronization before GPU timing in onnx_inference.py

**File:** `inference/onnx_inference.py:134-142`
**Issue:** The `pytorch_gpu_inference` function at line 134 calls `torch.cuda.synchronize()` inside the benchmarked function body, but the `@benchmark` decorator (from utils) already calls `synchronize()` after the function returns. The real concern is the opposite direction: `torch.cuda.synchronize()` is not called *before* the timed loop starts (inside the function), meaning prior async GPU ops from warmup could still be in-flight when `time.perf_counter()` starts in the decorator. However, reviewing the decorator, it does call `synchronize()` before `start = time.perf_counter()`, so this is actually handled. The inner synchronize is redundant but harmless. Downgrading severity -- the actual issue is that the `pytorch_cpu_inference` function at line 111 has no synchronization, but since it runs on CPU, this is correct. No action required on this specific point.

*Re-assessed:* This is actually fine. The decorator handles synchronization. Withdrawing this finding.

### WR-03 (revised): Unused `device` variable in onnx_inference.py

**File:** `inference/onnx_inference.py:76`
**Issue:** `device = get_device()` is called but the `device` variable is never used. The tutorial deliberately runs on CPU for fair comparison with ORT, so `get_device()` is called only for its side effects (printing device info). However, `print_device_info()` is called separately on line 77, so the `get_device()` return value is truly unused.
**Fix:** Either remove the assignment or use `_` to indicate the value is intentionally discarded:
```python
_ = get_device()  # called for detection side effects
print_device_info()
```

### WR-04: TensorRT output tensor hardcoded to shape (BATCH_SIZE, 10)

**File:** `inference/tensorrt_inference.py:251`
**Issue:** The output buffer `torch.empty(BATCH_SIZE, 10, device="cuda")` hardcodes the number of output classes to 10. While this matches `SimpleCNN`'s default `num_classes=10`, it creates a tight coupling. If the model's `num_classes` is ever changed, this line would silently produce incorrect results or memory corruption (TensorRT would write beyond the allocated buffer).
**Fix:** Query the output shape from the engine instead of hardcoding:
```python
output_shape = engine.get_tensor_shape(output_name)
# Replace fixed dim with batch size (dim 0 may be -1 for dynamic)
output_shape_resolved = (BATCH_SIZE,) + tuple(output_shape[1:])
output_tensor = torch.empty(output_shape_resolved, device="cuda")
```

## Info

### IN-01: Unused import `get_sample_batch` in onnx_inference.py

**File:** `inference/onnx_inference.py:38`
**Issue:** `get_sample_batch` is imported but never called. The tutorial manually creates input tensors instead.
**Fix:** Remove `get_sample_batch` from the import list.

### IN-02: Unused import `get_device` in onnx_inference.py (partial)

**File:** `inference/onnx_inference.py:40`
**Issue:** As noted in WR-03, `get_device()` is called but the return value is unused. The import itself is used, but the function call's return value is discarded.
**Fix:** Use `get_device()` without assignment or use `_`.

### IN-03: Redundant `.cuda()` call on already-CUDA tensor

**File:** `inference/tensorrt_inference.py:258`
**Issue:** `trt_input = inputs.contiguous().cuda()` -- the `inputs` tensor is already on the CUDA device (created via `get_sample_batch(device=device)` where `device` is CUDA). The `.cuda()` call is redundant. This is harmless (PyTorch returns the same tensor if already on CUDA) but slightly misleading.
**Fix:**
```python
trt_input = inputs.contiguous()
```

---

_Reviewed: 2026-04-13T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
