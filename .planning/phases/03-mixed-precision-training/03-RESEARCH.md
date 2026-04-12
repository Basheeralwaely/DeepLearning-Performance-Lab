# Phase 3: Mixed Precision Training - Research

**Researched:** 2026-04-12
**Domain:** PyTorch mixed precision training (AMP, BF16/FP16, FP8 with Transformer Engine)
**Confidence:** HIGH

## Summary

This phase creates three standalone Python tutorials teaching mixed precision training techniques with measurable speedup and memory benchmarks. The core PyTorch AMP API (`torch.amp.autocast`, `torch.amp.GradScaler`) is stable and well-documented in PyTorch 2.10. BF16 vs FP16 numerical differences are straightforward to demonstrate with clear real-world implications (overflow, underflow, precision loss). The FP8 tutorial using NVIDIA Transformer Engine requires careful conditional logic since FP8 needs compute capability 8.9+ (Ada/Hopper/Blackwell GPUs), and the development machine has an RTX 2070 (Turing, SM 7.5).

The key challenge is the FP8 tutorial: Transformer Engine is not installed and cannot run FP8 on the current hardware. The tutorial must be written with conditional GPU capability detection, providing FP8 code for Hopper+, BF16 fallback for Ampere, and FP16 fallback for older GPUs. The tutorial code must be structurally correct and educational even when FP8 is unavailable.

**Primary recommendation:** Use the modern `torch.amp` API (not the deprecated `torch.cuda.amp`), structure each tutorial with warmup + baseline + optimized pattern matching the reference tutorial, and build the FP8 tutorial with triple-tier hardware detection using `torch.cuda.get_device_capability()`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** FP8 tutorial uses a small Vision Transformer (ViT) model built with Transformer Engine layers -- patch embeddings + transformer encoder blocks, sized for tutorial demos. Add this model to `utils/models.py` alongside SimpleCNN/SimpleMLP.
- **D-02:** FP8 tutorial uses conditional sections based on GPU capability detection: FP8 code runs on Hopper+ (H100/H200), BF16 fallback on Ampere GPUs, FP16 fallback on older hardware. Every user can run the tutorial and learn something regardless of hardware.
- **D-03:** AMP tutorial (PREC-01) and BF16 vs FP16 tutorial (PREC-02) use existing SimpleCNN/SimpleMLP from `utils/models.py`.
- **D-04:** Before/after benchmarks measure training throughput (samples/sec) and peak GPU memory usage. Use existing `benchmark` decorator and `print_benchmark_table` from `utils/benchmark.py`.
- **D-05:** Loss convergence comparison is explicitly excluded -- focus on speed and memory savings.

### Claude's Discretion
- Training loop structure (how many iterations/epochs to run for meaningful benchmarks)
- ViT model sizing (number of layers, hidden dim, patch size) -- small enough for tutorial, large enough to show FP8 benefits
- Tutorial file naming within `mixed_precision/` (following Phase 1 D-02: topic-based, no numbering)
- How to structure the GPU capability detection and conditional sections in the FP8 tutorial
- Whether AMP and BF16/FP16 tutorials share a training loop helper or each define their own

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PREC-01 | User can learn AMP with autocast and GradScaler | Modern `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')` APIs verified on PyTorch 2.10. Old `torch.cuda.amp` API deprecated with FutureWarning. |
| PREC-02 | User can learn BF16 vs FP16 tradeoffs with comparison benchmarks | Both dtypes work on current hardware. BF16 has wider dynamic range (no overflow at 70000, no underflow at 1e-8) but lower precision (7 vs 10 mantissa bits). BF16 does NOT need GradScaler. |
| PREC-03 | User can learn FP8 training with Transformer Engine | TE v2.13.0 available. FP8 needs SM 8.9+. Current dev machine is SM 7.5 (Turing). Tutorial must use conditional hardware detection with BF16/FP16 fallbacks. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.10.0+cu128 | AMP autocast, GradScaler, BF16/FP16 training | Already installed, primary framework | [VERIFIED: local Python environment] |
| transformer-engine[pytorch] | 2.13.0 | FP8 training layers and autocast | NVIDIA's official FP8 training library for Transformers | [VERIFIED: PyPI] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| utils (project) | local | benchmark, models, device detection | All tutorials -- existing project utilities | [VERIFIED: codebase] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Transformer Engine | MS-AMP / manual FP8 casting | TE is the standard for FP8 training; others are experimental or incomplete |
| torch.amp.autocast | Manual dtype casting | autocast is the standard approach; manual is error-prone and not recommended |

**Installation:**
```bash
# Transformer Engine for FP8 tutorial (optional -- tutorial works without it)
pip install --no-build-isolation transformer_engine[pytorch]
```

## Architecture Patterns

### Recommended Project Structure
```
mixed_precision/
    amp_training.py           # PREC-01: AMP with autocast + GradScaler
    bf16_vs_fp16.py           # PREC-02: BF16 vs FP16 comparison
    fp8_transformer_engine.py # PREC-03: FP8 with Transformer Engine
utils/
    models.py                 # Add SimpleViT class here
    device.py                 # Add get_gpu_capability() helper here
```

### Pattern 1: Modern AMP Training Loop
**What:** Standard mixed precision training with autocast and GradScaler
**When to use:** PREC-01 tutorial
**Example:**
```python
# Source: PyTorch 2.10 -- verified via local API inspection
scaler = torch.amp.GradScaler('cuda')

for inputs, labels in dataloader:
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
[VERIFIED: torch.amp.autocast and torch.amp.GradScaler confirmed available in PyTorch 2.10 with correct signatures]

### Pattern 2: BF16 Training (No GradScaler Needed)
**What:** BF16 autocast without loss scaling
**When to use:** PREC-02 tutorial BF16 section
**Example:**
```python
# Source: PyTorch docs -- BF16 has 8 exponent bits, same range as FP32
# No GradScaler needed because BF16's dynamic range prevents underflow
for inputs, labels in dataloader:
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    loss.backward()  # No scaler needed!
    optimizer.step()
```
[VERIFIED: BF16 autocast works on current GPU, confirmed no underflow at 1e-8]

### Pattern 3: FP8 with Transformer Engine
**What:** FP8 forward pass with TE autocast and DelayedScaling recipe
**When to use:** PREC-03 tutorial on Hopper+ hardware
**Example:**
```python
# Source: https://github.com/NVIDIA/TransformerEngine README
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

fp8_recipe = DelayedScaling(
    fp8_format=Format.HYBRID,
    amax_history_len=16,
    amax_compute_algo="max",
)

model = te.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads)
model.to(dtype=torch.float16).cuda()

# fp8_autocast wraps ONLY the forward pass
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    output = model(input_tensor, attention_mask=None)

loss = criterion(output, target)
loss.backward()  # Must be OUTSIDE fp8_autocast
optimizer.step()
```
[CITED: https://github.com/NVIDIA/TransformerEngine]

### Pattern 4: GPU Capability Detection for Conditional Sections
**What:** Three-tier hardware detection for adaptive tutorials
**When to use:** PREC-03 FP8 tutorial
**Example:**
```python
# Source: Verified on local PyTorch 2.10
def get_gpu_tier():
    """Detect GPU capability tier for mixed precision support."""
    if not torch.cuda.is_available():
        return "cpu", "No CUDA GPU available"
    major, minor = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    if major > 8 or (major == 8 and minor >= 9):
        return "fp8", f"{gpu_name} (SM {major}.{minor}) -- FP8 supported"
    elif major >= 8:
        return "bf16", f"{gpu_name} (SM {major}.{minor}) -- BF16 native, FP8 not supported"
    else:
        return "fp16", f"{gpu_name} (SM {major}.{minor}) -- FP16 native, BF16 emulated"
```
[VERIFIED: torch.cuda.get_device_capability returns (7, 5) on RTX 2070]

### Anti-Patterns to Avoid
- **Using deprecated `torch.cuda.amp.autocast()`:** Produces FutureWarning in PyTorch 2.10. Use `torch.amp.autocast('cuda')` instead. [VERIFIED: FutureWarning confirmed locally]
- **Using GradScaler with BF16:** BF16 has the same exponent range as FP32, so loss scaling is unnecessary and wasteful. Only use GradScaler with FP16.
- **Putting backward pass inside `te.fp8_autocast()`:** TE documentation explicitly states fp8_autocast must exit before backward pass.
- **Not warming up CUDA before benchmarks:** First CUDA operations compile kernels and allocate memory, skewing timing results. Always include a warmup phase.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Loss scaling for FP16 | Custom loss scale tracking | `torch.amp.GradScaler` | Dynamic scaling handles overflow/underflow edge cases correctly |
| FP8 quantization | Manual FP8 casting | `transformer_engine.pytorch` | FP8 requires per-tensor scaling factors with delayed scaling recipes |
| Benchmark timing | `time.time()` with no sync | `utils.benchmark` decorator | Must call `torch.cuda.synchronize()` before timing GPU ops |
| GPU capability detection | Hardcoded GPU name checks | `torch.cuda.get_device_capability()` | Numeric compute capability is reliable; GPU names vary |

## Common Pitfalls

### Pitfall 1: Forgetting CUDA Synchronization in Benchmarks
**What goes wrong:** GPU operations are asynchronous; timing without `synchronize()` measures kernel launch time, not execution time.
**Why it happens:** CPU returns immediately after dispatching GPU work.
**How to avoid:** The existing `benchmark` decorator already handles this correctly. Always use it.
**Warning signs:** Suspiciously fast benchmark results (< 1ms for large operations).

### Pitfall 2: Using GradScaler with BF16
**What goes wrong:** Unnecessary computation and potentially confusing output.
**Why it happens:** Tutorials commonly show GradScaler with AMP without distinguishing FP16 from BF16.
**How to avoid:** PREC-02 must explicitly teach that BF16 does NOT need GradScaler because its 8 exponent bits give it the same dynamic range as FP32.
**Warning signs:** GradScaler scale factor staying at 65536 and never adjusting (because no overflow/underflow occurs).

### Pitfall 3: Deprecated torch.cuda.amp API
**What goes wrong:** FutureWarning floods output, code will break in future PyTorch versions.
**Why it happens:** Many online tutorials and blog posts still use the old API.
**How to avoid:** Use `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')`.
**Warning signs:** FutureWarning messages in tutorial output. [VERIFIED: confirmed locally]

### Pitfall 4: FP8 Tutorial Failing on Non-Hopper GPUs
**What goes wrong:** ImportError or runtime error when transformer-engine or FP8 ops not available.
**Why it happens:** FP8 requires SM 8.9+ hardware and transformer-engine installed.
**How to avoid:** Three-tier detection: try importing TE, check GPU capability, fall back gracefully. Tutorial must be educational even in fallback mode.
**Warning signs:** Tutorial crashes instead of printing helpful "your GPU doesn't support FP8" message.

### Pitfall 5: Benchmark Too Short to Be Meaningful
**What goes wrong:** Noise dominates results; speedup appears inconsistent.
**Why it happens:** A single forward+backward pass takes milliseconds; variance is high.
**How to avoid:** Run 50-100 training iterations minimum for benchmarks. Use warmup of 10+ iterations before timing. The exact count is Claude's discretion per CONTEXT.md.
**Warning signs:** Reported speedup varies wildly between runs.

### Pitfall 6: SimpleViT Too Small to Show FP8 Benefits
**What goes wrong:** FP8 shows no speedup or even slowdown vs FP16/BF16.
**Why it happens:** FP8's overhead (scaling factor computation) dominates for tiny models. Tensor dimensions must be divisible by 16 for TE.
**How to avoid:** Size the ViT large enough: hidden_dim >= 256, multiple transformer layers, patch size that creates enough tokens. Both dimensions of every Linear must be divisible by 16 (TE requirement).
**Warning signs:** FP8 benchmark is slower than BF16 baseline. [CITED: TE docs -- tensor dimensions must be divisible by 16]

## Code Examples

### SimpleViT Model for utils/models.py (D-01)
```python
# Recommended structure for the ViT model added to utils/models.py
# This is a STANDARD PyTorch ViT -- the FP8 tutorial will optionally
# replace layers with TE equivalents based on hardware detection.

class SimpleViT(nn.Module):
    """A minimal Vision Transformer for mixed precision tutorial demos.
    
    Args:
        image_size: Input image spatial dimension (square).
        patch_size: Size of each patch (must divide image_size evenly).
        num_classes: Number of output classes.
        dim: Hidden dimension (should be divisible by 16 for TE compatibility).
        depth: Number of transformer encoder layers.
        heads: Number of attention heads.
        mlp_dim: FFN hidden dimension (should be divisible by 16).
    """
    # Patch embedding -> positional embedding -> N x TransformerBlock -> class token -> head
    # Keep all dimensions divisible by 16 for TE FP8 compatibility
```
[ASSUMED: Specific ViT sizing recommendations are based on general transformer architecture knowledge. Exact dimensions to be determined during implementation.]

### GPU Capability Helper for utils/device.py
```python
# Source: Verified API on PyTorch 2.10 local environment
def get_gpu_capability() -> tuple[int, int] | None:
    """Return (major, minor) compute capability, or None if no CUDA GPU."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)
```
[VERIFIED: torch.cuda.get_device_capability(0) returns (7, 5) on RTX 2070]

### Numerical Stability Demo for PREC-02
```python
# Verified locally on PyTorch 2.10
# FP16 overflow: 70000.0 -> inf (max FP16 is 65504)
# BF16 overflow: 70000.0 -> 70144.0 (approximate but no overflow)
# FP16 underflow: 1e-8 -> 0.0 (below FP16 min normal ~6e-8)
# BF16 underflow: 1e-8 -> 1e-8 (BF16 min ~1e-38, same as FP32)
```
[VERIFIED: All values confirmed via local Python execution]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.cuda.amp.autocast()` | `torch.amp.autocast('cuda')` | PyTorch 2.x | Old API deprecated with FutureWarning |
| `torch.cuda.amp.GradScaler()` | `torch.amp.GradScaler('cuda')` | PyTorch 2.x | Old API deprecated with FutureWarning |
| `te.fp8_autocast()` | `te.autocast()` (newer TE versions) | TE 2.x | Unified autocast API, fp8_autocast still works |
| FP8 = Hopper only | FP8 = Hopper + Ada + Blackwell | TE 1.x+ | Ada Lovelace (SM 8.9) also supports FP8 |

**Deprecated/outdated:**
- `torch.cuda.amp.autocast`: Deprecated in favor of `torch.amp.autocast('cuda')` [VERIFIED: FutureWarning on PyTorch 2.10]
- `torch.cuda.amp.GradScaler`: Deprecated in favor of `torch.amp.GradScaler('cuda')` [VERIFIED: FutureWarning on PyTorch 2.10]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | SimpleViT hidden_dim >= 256 and depth >= 4 will show FP8 benefits on Hopper | Code Examples | Model may be too small or too large; adjustable at implementation time |
| A2 | 50-100 training iterations sufficient for stable benchmarks | Pitfalls | May need more iterations on fast GPUs; can tune during implementation |
| A3 | `te.fp8_autocast` still works in TE 2.13 (alongside newer `te.autocast`) | Architecture Patterns | If deprecated, use `te.autocast` instead -- low risk |
| A4 | Transformer Engine can be installed on Turing GPU (builds but FP8 ops disabled at runtime) | Standard Stack | If TE refuses to install on SM 7.5, FP8 tutorial must use try/except ImportError pattern |

## Open Questions

1. **Can Transformer Engine install on SM 7.5 (Turing)?**
   - What we know: TE supports Ampere+ for optimized ops, FP8 needs SM 8.9+. The docs say it supports "Ampere" for FP16/BF16 optimizations.
   - What's unclear: Whether `pip install transformer_engine[pytorch]` will succeed on a Turing-only machine or fail at build time.
   - Recommendation: The FP8 tutorial should guard TE imports with try/except and work purely as educational material when TE is unavailable. GPU capability detection runs before any TE import.

2. **Exact ViT sizing for the FP8 tutorial**
   - What we know: Must be large enough to show FP8 benefits, all Linear dimensions divisible by 16 (TE requirement). Must fit in 8GB GPU memory for dev testing.
   - What's unclear: Optimal hidden_dim/depth/heads tradeoff for tutorial purposes.
   - Recommendation: Start with dim=256, depth=4, heads=8, mlp_dim=512, image_size=32, patch_size=4 (64 tokens). Adjust during implementation if benchmarks are too fast.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch + CUDA | All tutorials | Yes | 2.10.0+cu128 | -- |
| CUDA toolkit | All tutorials | Yes | 12.8 | -- |
| NVIDIA GPU | All tutorials | Yes | RTX 2070 (SM 7.5, 7.6GB) | CPU mode (limited) |
| transformer-engine | PREC-03 (FP8) | No | -- | Tutorial uses try/except import, falls back to BF16/FP16 demo |
| BF16 hardware acceleration | PREC-02 | No (Turing = emulated) | -- | BF16 works in software; benchmark will show FP16 faster on this GPU |
| FP8 hardware | PREC-03 | No (need SM 8.9+) | -- | Tutorial conditional sections show BF16/FP16 fallback |

**Missing dependencies with no fallback:**
- None -- all tutorials have fallback paths

**Missing dependencies with fallback:**
- `transformer-engine`: FP8 tutorial includes try/except ImportError and GPU capability detection. Fallback runs BF16 or FP16 training loop instead.
- Native BF16 acceleration: RTX 2070 runs BF16 in software emulation. PREC-02 tutorial should note this -- BF16 may appear slower than FP16 on Turing, which is itself an educational data point about hardware requirements.

## Project Constraints (from CLAUDE.md)

- **Format**: .py files only -- no notebooks, no markdown
- **Logging**: Every tutorial must produce rich console output
- **Standalone**: Each tutorial independently runnable
- **Framework**: PyTorch + NVIDIA tooling
- **GSD Workflow**: All file changes through GSD workflow

## Sources

### Primary (HIGH confidence)
- Local PyTorch 2.10.0+cu128 environment -- verified API signatures, deprecation warnings, dtype behavior, GPU capability detection
- Local GPU inspection -- RTX 2070, SM 7.5, 7.6GB VRAM, BF16 emulated
- Existing codebase -- utils/benchmark.py, utils/models.py, utils/device.py, profiling/reference_tutorial.py

### Secondary (MEDIUM confidence)
- [NVIDIA Transformer Engine GitHub](https://github.com/NVIDIA/TransformerEngine) -- README with quickstart code, hardware requirements, installation
- [Transformer Engine PyPI](https://pypi.org/project/transformer-engine/) -- v2.13.0, Python 3.10+, CUDA 12.1+
- [TE FP8 Primer Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) -- FP8 autocast patterns, DelayedScaling recipe

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- PyTorch AMP verified locally, TE verified via PyPI and official docs
- Architecture: HIGH -- Tutorial patterns established by Phase 1/2, AMP API verified locally
- Pitfalls: HIGH -- Deprecation warnings confirmed locally, numerical behavior verified, TE constraints documented
- FP8 specifics: MEDIUM -- Cannot test FP8 on current hardware; API patterns from official docs

**Research date:** 2026-04-12
**Valid until:** 2026-05-12 (stable domain, PyTorch AMP API unlikely to change)
