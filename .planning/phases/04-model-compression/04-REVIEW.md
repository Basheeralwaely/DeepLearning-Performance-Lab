---
phase: 04-model-compression
reviewed: 2026-04-13T12:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - pruning/structured_unstructured_pruning.py
  - compression/knowledge_distillation.py
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 04: Code Review Report

**Reviewed:** 2026-04-13T12:00:00Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Reviewed two new tutorial files for the model compression phase: a pruning tutorial covering structured vs unstructured approaches, and a knowledge distillation tutorial implementing Hinton-style teacher-student compression. Both files are well-structured, have excellent inline documentation, and follow the project conventions (standalone .py files with rich logging output). No security issues or critical bugs found. Three warnings relate to fragile model construction, hardcoded dimension assumptions, and a missing division guard. Two info items cover minor code quality issues.

## Warnings

### WR-01: Fragile Model Construction via Monkey-Patched forward Method

**File:** `pruning/structured_unstructured_pruning.py:195-208`
**Issue:** `build_pruned_model` creates a bare `nn.Module()` and attaches a `forward` method at runtime using `types.MethodType`. This bypasses PyTorch's module registration patterns and will fail with `torch.jit.script`, `torch.compile`, `torch.export`, or any tool that inspects the class definition for `forward`. It also makes the model harder to serialize/deserialize correctly.
**Fix:** Define a proper subclass instead of monkey-patching:
```python
class PrunedCNN(nn.Module):
    def __init__(self, features, classifier):
        super().__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Then in build_pruned_model:
new_model = PrunedCNN(new_features, copy.deepcopy(temp_model.classifier))
return new_model.to(device)
```

### WR-02: Hardcoded flat_dim Assumes INPUT_SIZE=32

**File:** `compression/knowledge_distillation.py:90,123`
**Issue:** Both `TeacherCNN` and `StudentCNN` hardcode `flat_dim` based on the assumption that input is 32x32 (e.g., `256 * 4 * 4`). The constant `INPUT_SIZE = 32` exists at module level but is not used in the model constructors. If `INPUT_SIZE` is changed, the models will crash at runtime with a tensor shape mismatch in the classifier's first Linear layer, with no clear error message pointing to the root cause.
**Fix:** Compute `flat_dim` dynamically from `INPUT_SIZE`, or accept `input_size` as a constructor parameter (matching the pattern in `utils/models.py`'s `SimpleCNN`):
```python
class TeacherCNN(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super().__init__()
        # ... features unchanged ...
        reduced_size = input_size // 8  # 3x MaxPool2d(2)
        flat_dim = 256 * reduced_size * reduced_size
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, NUM_CLASSES),
        )
```

### WR-03: Potential ZeroDivisionError in Sparsity Calculation

**File:** `pruning/structured_unstructured_pruning.py:328`
**Issue:** `zero_weights / total_weights * 100` has no guard against `total_weights == 0`. While this is unlikely with `SimpleCNN`, the function operates on a deep-copied model after pruning, and defensive coding would prevent a confusing crash if the model structure changes.
**Fix:** Add a guard:
```python
actual_sparsity = (zero_weights / total_weights * 100) if total_weights > 0 else 0.0
```

## Info

### IN-01: Import Statement Inside Function Body

**File:** `pruning/structured_unstructured_pruning.py:206`
**Issue:** `import types` is placed inside `build_pruned_model()` rather than at the top of the file with other imports. This is a minor style issue but deviates from Python convention (PEP 8) and makes the dependency less discoverable. Note: this import becomes unnecessary if WR-01 is addressed.
**Fix:** Move `import types` to the top-level imports block, or remove entirely if WR-01's fix is adopted.

### IN-02: Misleading Variable Name for Speed Comparison

**File:** `compression/knowledge_distillation.py:448-452`
**Issue:** The variable `speed_ratio` actually computes a percentage (distilled time / teacher time * 100), not a ratio. The log message uses it correctly ("of teacher's inference time"), but the variable name could confuse future maintainers.
**Fix:** Rename to `speed_pct` or `inference_time_pct` for clarity:
```python
inference_time_pct = (
    distill_bench["time_seconds"] / teacher_bench["time_seconds"] * 100
    if teacher_bench["time_seconds"] > 0
    else 0
)
```

---

_Reviewed: 2026-04-13T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
