# Phase 04: Model Compression - Research

**Researched:** 2026-04-13
**Domain:** PyTorch model pruning and knowledge distillation
**Confidence:** HIGH

## Summary

Phase 4 covers two standalone tutorials: (1) a pruning tutorial demonstrating both unstructured (L1 magnitude via `torch.nn.utils.prune`) and structured (physical channel removal) pruning with an iterative prune-then-fine-tune workflow, and (2) a knowledge distillation tutorial using Hinton's soft-label approach with temperature-scaled KL divergence loss to transfer knowledge from a larger SimpleCNN teacher to a smaller SimpleCNN student.

Both techniques use only PyTorch stdlib -- no additional dependencies are needed. The existing `SimpleCNN` (620K params, conv channels 32->64->128) serves as the pruning subject and as the baseline for teacher/student sizing. The pruning tutorial must explicitly demonstrate that unstructured pruning does NOT speed up inference on standard GPUs (just creates zeros in dense tensors), while structured pruning physically removes channels for real speedup. The distillation tutorial must compare three outcomes: teacher, student-from-scratch, and distilled-student.

**Primary recommendation:** Use `torch.nn.utils.prune` for the unstructured section, manually reconstruct smaller layers for the structured section, and use `F.kl_div` + `F.cross_entropy` for the distillation loss. All APIs verified working on PyTorch 2.8.0.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Tutorial covers BOTH unstructured (L1 magnitude on individual weights) and structured (filter/channel removal) pruning in a single tutorial. Unstructured section uses `torch.nn.utils.prune`; structured section physically removes channels and adjusts subsequent layers.
- **D-02:** Includes iterative prune-then-fine-tune loop: train -> prune -> fine-tune -> measure. Shows the practical workflow, not just the pruning API. Multiple sparsity levels demonstrated.
- **D-03:** Explicitly logs that unstructured pruning does NOT produce inference speedup on standard GPUs (needs sparse hardware), while structured pruning shows real speedup.
- **D-04:** Uses `SimpleCNN` from `utils/models.py` as the pruning subject -- keep focus on the pruning technique, not the model.
- **D-05:** Teacher and student are SAME architecture family -- both SimpleCNN variants. Teacher is a larger SimpleCNN (more filters/layers), student is a smaller SimpleCNN. Need to add these size variants to `utils/models.py` or define them within the tutorial.
- **D-06:** Loss formulation is classic Hinton distillation: weighted combination of KL-divergence between teacher/student soft logits (with temperature scaling) and standard cross-entropy on ground truth labels. Both loss components demonstrated.
- **D-07:** Tutorial trains teacher first, then distills into student, comparing student-with-distillation vs student-trained-from-scratch.
- **D-08:** Pruning tutorial goes in `pruning/` folder. Distillation tutorial goes in `compression/` folder. Both folders already scaffolded.
- **D-09:** Both tutorials measure and compare: model size (total parameter count + saved model file size on disk) and inference speed (latency/throughput per batch). Use existing `benchmark` decorator and `print_benchmark_table` from `utils/benchmark.py`.
- **D-10:** Accuracy tracking and per-layer sparsity analysis are explicitly excluded -- focus stays on performance metrics (size and speed), consistent with Phase 3 D-05.

### Claude's Discretion
- SimpleCNN sizing for teacher and student variants (filter counts, layer depth)
- Specific sparsity levels to demonstrate in the pruning tutorial
- Number of fine-tuning epochs per pruning iteration
- Temperature and alpha hyperparameters for distillation loss
- Tutorial file naming within pruning/ and compression/ (following Phase 1 D-02: topic-based, no numbering)
- Whether teacher/student size variants go in utils/models.py or are defined inline in the tutorial
- Training data setup (synthetic vs CIFAR-style -- keep it standalone)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| COMP-01 | User can learn pruning techniques (structured/unstructured, magnitude-based) | `torch.nn.utils.prune` API verified for L1 unstructured + `ln_structured`; physical channel removal pattern documented; iterative prune-fine-tune workflow researched |
| COMP-02 | User can learn knowledge distillation for model compression | KL divergence + CE loss formulation verified; teacher/student SimpleCNN sizing researched; three-way comparison pattern documented |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.8.0+cu126 | All pruning, training, distillation | Already installed, only dependency needed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.nn.utils.prune | (stdlib) | Unstructured and structured pruning masks | Pruning tutorial unstructured section |
| torch.nn.functional | (stdlib) | kl_div, cross_entropy, softmax, log_softmax | Distillation loss computation |
| tempfile + os | (stdlib) | Measuring saved model file size on disk | Both tutorials for size comparison |

[VERIFIED: PyTorch 2.8.0+cu126 installed and CUDA available on system]

**No new dependencies required.** Both tutorials use only PyTorch stdlib and Python stdlib.

## Architecture Patterns

### Recommended Project Structure
```
pruning/
  structured_unstructured_pruning.py    # COMP-01: pruning tutorial
compression/
  knowledge_distillation.py             # COMP-02: distillation tutorial
utils/
  models.py                             # May need teacher/student variants added
```

### Pattern 1: Iterative Prune-Then-Fine-Tune Loop (COMP-01)
**What:** Train baseline -> apply pruning at sparsity level -> fine-tune a few epochs -> measure -> repeat at higher sparsity
**When to use:** The pruning tutorial, to show the practical workflow
**Example:**
```python
# Source: verified against torch.nn.utils.prune API on PyTorch 2.8.0
import torch.nn.utils.prune as prune

# Unstructured pruning: creates mask, zeros out weights, but tensor stays dense
for sparsity in [0.2, 0.5, 0.8]:
    prune.l1_unstructured(module, name='weight', amount=sparsity)
    # Fine-tune for N epochs...
    # Measure inference time (will NOT be faster -- same dense tensor)
    prune.remove(module, 'weight')  # Make pruning permanent
```

### Pattern 2: Physical Channel Removal for Structured Pruning (COMP-01)
**What:** Use `ln_structured` to identify which channels to prune, then physically reconstruct smaller layers
**When to use:** Structured pruning section -- this is what produces real inference speedup
**Example:**
```python
# Source: verified on PyTorch 2.8.0
# Step 1: Apply structured pruning to get mask
prune.ln_structured(conv, name='weight', amount=0.5, n=1, dim=0)
mask = conv.weight_mask

# Step 2: Identify surviving channels
channel_norms = mask.view(mask.shape[0], -1).sum(dim=1)
keep_indices = (channel_norms > 0).nonzero(as_tuple=True)[0]

# Step 3: Build new smaller conv layer
new_conv = nn.Conv2d(in_channels, len(keep_indices), kernel_size, padding=padding)
new_conv.weight.data = conv.weight_orig.data[keep_indices]
new_conv.bias.data = conv.bias.data[keep_indices]

# Step 4: Adjust NEXT layer's input channels
next_conv = nn.Conv2d(len(keep_indices), out_channels, ...)
next_conv.weight.data = original_next.weight.data[:, keep_indices]
```

### Pattern 3: Hinton Distillation Loss (COMP-02)
**What:** Temperature-scaled KL divergence on soft targets + cross-entropy on hard labels
**When to use:** Knowledge distillation tutorial
**Example:**
```python
# Source: Hinton et al. 2015 formulation, verified on PyTorch 2.8.0
import torch.nn.functional as F

T = 4.0    # temperature
alpha = 0.7  # weight for soft target loss

# Soft target loss (KL divergence)
soft_loss = F.kl_div(
    F.log_softmax(student_logits / T, dim=1),
    F.softmax(teacher_logits / T, dim=1),
    reduction='batchmean'
) * (T * T)

# Hard target loss (standard cross-entropy)
hard_loss = F.cross_entropy(student_logits, labels)

# Combined
loss = alpha * soft_loss + (1 - alpha) * hard_loss
```

### Pattern 4: Model Size Measurement
**What:** Compare parameter count and saved file size on disk
**When to use:** Both tutorials for before/after size comparison
**Example:**
```python
# Source: verified on PyTorch 2.8.0
import tempfile, os

# Parameter count
total_params = sum(p.numel() for p in model.parameters())

# File size on disk
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    torch.save(model.state_dict(), f.name)
    size_mb = os.path.getsize(f.name) / (1024 * 1024)
    os.unlink(f.name)
```

### Anti-Patterns to Avoid
- **Claiming unstructured pruning speeds up inference:** Unstructured pruning creates zeros in dense tensors. Standard CUDA kernels still do full dense matrix ops. Only structured pruning (physically smaller layers) produces real speedup. The tutorial must log this explicitly (D-03).
- **Pruning without fine-tuning:** Single-shot high-sparsity pruning devastates model quality. Always show iterative prune-then-fine-tune (D-02).
- **Using `prune.remove()` before fine-tuning:** `prune.remove()` makes the mask permanent and removes the hook. Fine-tune FIRST (mask keeps weights at zero during gradient updates), THEN call `remove()`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Unstructured weight masking | Custom zero-out logic | `torch.nn.utils.prune.l1_unstructured` | Handles mask lifecycle, forward hooks, composability |
| Structured channel ranking | Manual L1 norm computation + sorting | `torch.nn.utils.prune.ln_structured` | Correct per-channel norm, composable with other methods |
| KL divergence computation | Manual softmax + log + element-wise multiply | `F.kl_div` with `F.log_softmax` / `F.softmax` | Numerically stable, correct gradient computation |
| Timing + memory measurement | Manual `time.time()` + CUDA sync | Existing `benchmark` decorator from `utils/benchmark.py` | Already handles CUDA sync, peak memory, consistent format |
| Comparison tables | Custom print formatting | Existing `print_benchmark_table` from `utils/benchmark.py` | Consistent with all other tutorials |

## Common Pitfalls

### Pitfall 1: Expecting Inference Speedup from Unstructured Pruning
**What goes wrong:** Engineer applies 90% unstructured pruning, sees no speedup, thinks pruning is useless
**Why it happens:** Dense tensor operations on GPU process all elements regardless of zeros. Sparse hardware (NVIDIA A100 structured sparsity, or sparse GEMM kernels) is needed.
**How to avoid:** Tutorial must explicitly log: "Unstructured pruning: model has X% zeros but inference time is unchanged because standard GPU kernels operate on dense tensors"
**Warning signs:** Inference time within noise range of unpruned model

### Pitfall 2: Forgetting to Adjust Downstream Layers in Structured Pruning
**What goes wrong:** Remove output channels from conv1, but conv2 still expects the original number of input channels. Shape mismatch error.
**Why it happens:** Structured pruning changes tensor dimensions -- every subsequent layer that consumes the pruned output must be rebuilt.
**How to avoid:** When removing output channels from layer N, also slice input channels of layer N+1. For SimpleCNN: pruning features.0 (32 out) requires adjusting features.3 (32 in).
**Warning signs:** RuntimeError about mismatched tensor sizes during forward pass

### Pitfall 3: Wrong KL Divergence Argument Order
**What goes wrong:** `F.kl_div(target, input)` instead of `F.kl_div(input, target)` -- PyTorch convention is `kl_div(log_input, target)`.
**Why it happens:** Mathematical KL(P||Q) convention differs from PyTorch API signature.
**How to avoid:** Always: `F.kl_div(F.log_softmax(student/T), F.softmax(teacher/T), reduction='batchmean')`
**Warning signs:** Negative loss values, loss not decreasing during training

### Pitfall 4: Missing Temperature Squared Scaling
**What goes wrong:** KL divergence loss is too small relative to CE loss because softmax with temperature produces lower-magnitude gradients.
**Why it happens:** Hinton et al. show that gradients through soft targets scale as 1/T^2, so you must multiply the KL loss by T^2 to compensate.
**How to avoid:** Always include `* (T * T)` after the KL divergence term.
**Warning signs:** Distilled student performs nearly identical to student-from-scratch (soft targets having no effect)

### Pitfall 5: Training Data -- Keep It Standalone
**What goes wrong:** Tutorial requires downloading CIFAR-10 or other dataset, fails in air-gapped environments.
**Why it happens:** Using real datasets makes tutorials non-standalone.
**How to avoid:** Use synthetic data (random tensors with random labels). The goal is demonstrating the technique's effect on model size and speed, not achieving SOTA accuracy. Consistent with Phase 1-3 approach using `get_sample_batch`.
**Warning signs:** ImportError or download failures when running tutorial

## Code Examples

### SimpleCNN Architecture Reference (Pruning Subject)
```python
# Source: utils/models.py (verified in codebase)
# Default SimpleCNN: 620,362 parameters
# features.0: Conv2d(3, 32, 3, padding=1)   ->     864 + 32 params
# features.3: Conv2d(32, 64, 3, padding=1)   ->  18,432 + 64 params
# features.6: Conv2d(64, 128, 3, padding=1)  ->  73,728 + 128 params
# classifier.0: Linear(2048, 256)             -> 524,288 + 256 params
# classifier.2: Linear(256, 10)              ->   2,560 + 10 params
# MaxPool2d(2) after each conv block: 32->16->8->4 spatial
```

### Teacher/Student SimpleCNN Sizing Recommendation
```python
# Recommendation for distillation tutorial (Claude's discretion per D-05)
# Teacher: ~2.4M params (wider channels: 64->128->256)
# Default SimpleCNN: ~620K params (channels: 32->64->128)
# Student: ~160K params (narrower channels: 16->32->64)
#
# This gives roughly 15x size ratio teacher-to-student,
# making compression benefit visually obvious in benchmark output.
#
# Define inline in tutorial (not in utils/models.py) because:
# - Teacher/student are tutorial-specific concepts
# - Avoids polluting shared utils with one-off variants
# - Keeps the tutorial fully self-contained and readable
```

### Sparsity Levels Recommendation
```python
# Recommendation for pruning tutorial (Claude's discretion)
SPARSITY_LEVELS = [0.2, 0.5, 0.7, 0.9]
# 20% - mild pruning, negligible impact
# 50% - moderate, small impact
# 70% - aggressive, noticeable impact
# 90% - extreme, significant degradation
# Shows clear progression and demonstrates the tradeoff
```

### Distillation Hyperparameters Recommendation
```python
# Recommendation (Claude's discretion)
TEMPERATURE = 4.0   # Standard choice from Hinton et al.
ALPHA = 0.7         # Weight soft targets heavily (they carry more information)
# T=4 produces soft enough probability distributions
# alpha=0.7 means 70% soft-target loss, 30% hard-target loss
```

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Teacher (64->128->256 channels) will have ~2.4M params | Code Examples | Low -- easy to adjust, just filter counts |
| A2 | Student (16->32->64 channels) will have ~160K params | Code Examples | Low -- easy to adjust |
| A3 | T=4.0, alpha=0.7 are good defaults for distillation demo | Code Examples | Low -- well-established defaults from Hinton et al. |
| A4 | Synthetic data is sufficient to demonstrate compression metrics | Pitfalls | Low -- consistent with Phases 1-3 approach |

## Open Questions (RESOLVED)

1. **Teacher/student variant placement** — RESOLVED: Define inline in the distillation tutorial. Teacher and student are tutorial-specific concepts, not reusable utilities. Keeps utils/models.py clean. (Implemented in 04-02-PLAN.md)

2. **Structured pruning scope for SimpleCNN** — RESOLVED: Demonstrate structured pruning on features.0 (32 channels -> 16 channels), propagating the change to features.3 input. This is the simplest chain to illustrate. Mention that the same approach applies to deeper layers. (Implemented in 04-01-PLAN.md)

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch | Both tutorials | Yes | 2.8.0+cu126 | -- |
| CUDA GPU | Inference speed benchmarks | Yes | CUDA available | CPU fallback (tutorials should work on CPU too) |
| torch.nn.utils.prune | Pruning tutorial | Yes | stdlib in PyTorch 2.8.0 | -- |

No missing dependencies. All required APIs are PyTorch stdlib.

## Project Constraints (from CLAUDE.md)

- **Format:** .py files only -- no notebooks, no markdown-only docs
- **Logging:** Every tutorial must produce rich console output showing what's happening and why
- **Standalone:** Each tutorial must be independently runnable (`python tutorial_name.py`)
- **Framework:** PyTorch + NVIDIA tooling
- **Benchmarks:** Before/after performance comparisons as formatted tables
- **GSD Workflow:** Use GSD commands for all file changes

## Sources

### Primary (HIGH confidence)
- PyTorch 2.8.0 `torch.nn.utils.prune` API -- tested directly on installed version
- PyTorch 2.8.0 `F.kl_div`, `F.cross_entropy` -- tested directly on installed version
- Existing codebase: `utils/models.py`, `utils/benchmark.py`, `utils/__init__.py` -- read from repo
- SimpleCNN parameter counts -- computed directly from model instantiation

### Secondary (MEDIUM confidence)
- Hinton distillation loss formulation (T^2 scaling, alpha weighting) [ASSUMED: well-known from Hinton et al. 2015 "Distilling the Knowledge in Neural Networks"]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all APIs verified on installed PyTorch 2.8.0
- Architecture: HIGH - patterns tested with working code, codebase conventions observed from Phases 1-3
- Pitfalls: HIGH - common issues verified through direct API testing (mask lifecycle, argument order, T^2 scaling)

**Research date:** 2026-04-13
**Valid until:** 2026-05-13 (stable PyTorch APIs, unlikely to change)
