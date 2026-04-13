---
phase: 06-distributed-training
reviewed: 2026-04-13T12:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - distributed_training/ddp_training.py
  - distributed_training/fsdp_training.py
  - distributed_training/model_parallelism.py
  - distributed_training/deepspeed_zero.py
findings:
  critical: 1
  warning: 2
  info: 2
  total: 5
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-04-13T12:00:00Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Four distributed training tutorial files were reviewed: DDP, FSDP, model parallelism, and DeepSpeed ZeRO. The code is generally well-structured with thorough docstrings, proper cleanup in `finally` blocks, and good educational logging. However, one critical deadlock bug was found in `ddp_training.py` where baseline training runs only on rank 0 but calls a function containing `dist.barrier()`, which will hang on multi-GPU systems. Two warnings address a minor gradient computation issue and an inconsistent helper function API across files.

## Critical Issues

### CR-01: Deadlock in DDP baseline training on multi-GPU systems

**File:** `distributed_training/ddp_training.py:309-330`
**Issue:** The baseline training (Section 4) is guarded by `if rank == 0:` (line 309), so only rank 0 calls `train_loop()`. However, `train_loop()` contains two `dist.barrier()` calls (lines 130 and 157). On a multi-GPU system (world_size > 1), rank 0 will block at the barrier inside `train_loop` while all other ranks skip to the barrier at line 330. Since these are different collective operations, the process group will deadlock -- rank 0 waits for peers inside `train_loop`, while peers wait for rank 0 at line 330. Neither side will ever proceed.

This only manifests on multi-GPU systems (world_size > 1). Single-GPU runs are unaffected.

**Fix:** Either run baseline on all ranks (like `fsdp_training.py` does), or create a separate baseline training function without barriers:

```python
# Option A: Run baseline on all ranks (preferred, matches fsdp_training.py pattern)
print_rank0("\n--- Section 4: Baseline Training (no DDP) ---\n", rank)

if rank == 0:
    logger.info("Training baseline model without DDP wrapping...")

baseline_model = SimpleViT(
    dim=256, depth=4, heads=8, mlp_dim=512
).to(device)
baseline_result = train_loop(
    baseline_model,
    device,
    rank,
    NUM_ITERATIONS,
    WARMUP_ITERATIONS,
    BATCH_SIZE,
)

if rank == 0:
    logger.info(
        f"Baseline: {baseline_result['time_seconds']:.4f}s, "
        f"Memory: {baseline_result['memory_mb']}"
    )

# barrier at line 330 can then be removed since train_loop already synchronizes
```

## Warnings

### WR-01: Gradient norm computation mixes device and host operations

**File:** `distributed_training/ddp_training.py:195`
**Issue:** The expression `param.grad.data.norm(2).item() ** 2` calls `.item()` to convert a CUDA tensor to a Python float before squaring, then adds the Python float back to a CUDA tensor (`local_norm`). While this works (PyTorch auto-converts scalars), it forces a CUDA-to-CPU synchronization on every parameter, which can serialize GPU operations. More importantly, the accumulation loses GPU precision by round-tripping through Python floats.

**Fix:**
```python
# Keep computation on device -- no .item() needed
local_norm = torch.tensor(0.0, device=device)
for param in ddp_model.parameters():
    if param.grad is not None:
        local_norm += param.grad.data.norm(2) ** 2
local_norm = local_norm.sqrt()
```

### WR-02: Inconsistent `log_rank0` function signature across files

**File:** `distributed_training/deepspeed_zero.py:136` vs `distributed_training/ddp_training.py:75` and `distributed_training/fsdp_training.py:84`
**Issue:** `deepspeed_zero.py` defines `log_rank0(logger, msg, rank)` with logger as the first argument, while `ddp_training.py` and `fsdp_training.py` define `log_rank0(msg, rank, logger)` with logger as the last argument. This inconsistency creates a maintenance hazard -- copy-pasting logging calls between files will silently pass the wrong arguments (all three params accept any type), producing incorrect behavior rather than a clear error.

**Fix:** Standardize the signature across all files. The `(msg, rank, logger)` order is used in 2 of 3 files, so update `deepspeed_zero.py` to match:
```python
def log_rank0(msg: str, rank: int, logger) -> None:
```
And update all call sites in `deepspeed_zero.py` accordingly (swap first two arguments).

## Info

### IN-01: Code duplication of setup/cleanup/log_rank0/print_rank0 across all distributed files

**File:** `distributed_training/ddp_training.py:50-95`, `distributed_training/fsdp_training.py:59-104`, `distributed_training/deepspeed_zero.py:109-156`
**Issue:** The functions `setup()`, `cleanup()`, `log_rank0()`, and `print_rank0()` are nearly identical across all three distributed training files. This duplication means bug fixes (like CR-01) or signature changes (like WR-02) must be applied in multiple places.

**Fix:** Consider extracting these into a shared `distributed_training/dist_utils.py` module. However, this trades off against the project's "standalone tutorial" constraint in CLAUDE.md, so this is a judgment call for the maintainer.

### IN-02: Magic number 4 in layer count log message

**File:** `distributed_training/model_parallelism.py:321`
**Issue:** The line `logger.info(f"Splitting {4} transformer layers across {len(devices)} device(s)")` hardcodes `4` instead of deriving it from the model configuration. If the `depth` parameter changes, this log message will be incorrect.

**Fix:**
```python
depth = 4
# ... later ...
logger.info(f"Splitting {depth} transformer layers across {len(devices)} device(s)")
```

---

_Reviewed: 2026-04-13T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
