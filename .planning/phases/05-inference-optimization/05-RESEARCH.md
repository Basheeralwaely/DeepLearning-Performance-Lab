# Phase 5: Inference Optimization - Research

**Researched:** 2026-04-13
**Domain:** PyTorch model export and inference acceleration (TensorRT, ONNX Runtime, TorchScript)
**Confidence:** HIGH

## Summary

Phase 5 creates three standalone tutorials demonstrating inference optimization via model export and compilation. The environment has all required dependencies installed and verified: PyTorch 2.8.0+cu126, TensorRT 10.12.0, ONNX 1.19.0, and ONNX Runtime 1.23.2. Full pipeline testing confirms that SimpleCNN successfully exports to ONNX, builds TensorRT FP16 engines, traces/scripts via TorchScript, and runs through ONNX Runtime.

Key environmental constraint: ONNX Runtime only has CPUExecutionProvider available (CUDAExecutionProvider is not functional despite onnxruntime-gpu being installed). The ONNX tutorial must benchmark ORT CPU vs PyTorch GPU and PyTorch CPU, making the comparison fair by running both on CPU when comparing ORT. The TensorRT tutorial path (PyTorch -> ONNX -> TRT engine) is fully functional on GPU with FP16 mode.

**Primary recommendation:** Follow the established tutorial pattern from prior phases. Use TensorRT 10.x API (set_tensor_address + execute_async_v3), legacy torch.onnx.export with opset_version=17 (dynamo export requires onnxscript which is not installed), and standard torch.jit.trace/torch.jit.script for TorchScript.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** All three tutorials use `SimpleCNN` from `utils/models.py` as the model subject
- **D-02:** TensorRT: Export PyTorch -> ONNX -> TensorRT engine -> benchmark. Show FP16 engine mode. No INT8 calibration, no dynamic shapes, no workspace tuning
- **D-03:** Graceful dependency handling for TensorRT -- try-import, log instructions if missing, skip TRT sections
- **D-04:** ONNX: Export to ONNX, run through ONNX Runtime with graph optimizations, benchmark vs PyTorch
- **D-05:** Graceful dependency handling for ONNX Runtime -- try-import, log instructions if missing
- **D-06:** TorchScript: Both tracing and scripting side-by-side with pros/cons and benchmarks
- **D-07:** No external dependencies for TorchScript -- built into PyTorch
- **D-08:** All tutorials follow graceful skip pattern from Phase 3 FP8 tutorial
- **D-09:** Three separate standalone .py files, one technique per tutorial
- **D-10:** Benchmark with existing `benchmark` decorator and `print_benchmark_table` from `utils/benchmark.py`. Measure latency/throughput vs PyTorch baseline. No accuracy tracking.

### Claude's Discretion
- Specific benchmark methodology (warmup runs, iterations, batch sizes)
- Tutorial file naming within `inference/` (topic-based, no numbering)
- ONNX Runtime optimization levels and graph optimization details
- TensorRT builder configuration details (max workspace size, etc.)
- How to structure graceful skip sections
- Training data setup (synthetic data, keep standalone)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INFER-01 | User can learn TensorRT model export, optimization, and benchmarking | Full TRT 10.x pipeline verified: ONNX export -> OnnxParser -> Builder with FP16 -> engine deserialization -> execute_async_v3. All APIs tested on this environment. |
| INFER-02 | User can learn ONNX export and runtime optimization | ONNX export (opset 17) and ORT inference verified. Only CPUExecutionProvider available -- tutorial should benchmark ORT CPU optimizations vs PyTorch CPU baseline for fair comparison, plus show GPU provider detection pattern. |
| INFER-03 | User can learn TorchScript JIT compilation (tracing vs scripting) | Both torch.jit.trace and torch.jit.script work with SimpleCNN. No external deps needed. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Format**: .py files only -- no notebooks, no markdown-only docs
- **Logging**: Every tutorial must produce rich console output showing what's happening and why
- **Standalone**: Each tutorial must be independently runnable
- **Framework**: PyTorch + NVIDIA tooling (TensorRT, CUDA)
- **GSD Workflow**: All changes through GSD workflow

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.8.0+cu126 | Base framework, model definition, ONNX export, TorchScript | Already installed, project foundation [VERIFIED: python import] |
| tensorrt | 10.12.0.36 | TensorRT engine build and inference | Industry standard for NVIDIA GPU inference acceleration [VERIFIED: python import] |
| onnx | 1.19.0 | ONNX model format, intermediate for TRT pipeline | Standard interchange format [VERIFIED: python import] |
| onnxruntime | 1.23.2 | ONNX model inference with graph optimizations | Microsoft's optimized inference runtime [VERIFIED: python import] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | (installed) | Array conversion for ORT input/output | ORT expects numpy arrays for CPU inference [VERIFIED: available] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Legacy torch.onnx.export | torch.onnx.export(dynamo=True) | Dynamo export is the future but requires `onnxscript` (not installed). Legacy works fine and is stable. [VERIFIED: dynamo fails with ModuleNotFoundError] |
| TensorRT Python API | torch_tensorrt | Higher-level API but adds dependency; raw TRT API is more educational [ASSUMED] |

## Architecture Patterns

### Recommended Project Structure
```
inference/
    tensorrt_inference.py     # INFER-01: TensorRT export + FP16 optimization
    onnx_inference.py         # INFER-02: ONNX export + ORT optimization
    torchscript_inference.py  # INFER-03: TorchScript tracing vs scripting
```

### Pattern 1: Tutorial Structure (from reference_tutorial.py)
**What:** Standardized tutorial layout matching all prior phases
**When to use:** Every tutorial
**Example:**
```python
"""
[Title]
=======
[Multi-line docstring explaining the technique]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import (
    setup_logging, benchmark, compare_results, print_benchmark_table,
    SimpleCNN, get_sample_batch, get_device, print_device_info,
)

logger = setup_logging("tutorial_name")

# Constants
NUM_ITERATIONS = 100
BATCH_SIZE = 64
WARMUP_ITERATIONS = 10

def main():
    print("\n" + "=" * 60)
    print("  Tutorial Title")
    print("=" * 60 + "\n")
    
    device = get_device()
    print_device_info()
    # ... sections with print() headers, logger for details
    # ... benchmark comparison table at end

if __name__ == "__main__":
    main()
```
[VERIFIED: profiling/reference_tutorial.py pattern]

### Pattern 2: Graceful Dependency Skip
**What:** Try-import optional libraries, skip sections if missing
**When to use:** TensorRT and ONNX Runtime tutorials
**Example:**
```python
try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

# Later in main():
if not HAS_TENSORRT:
    logger.warning("TensorRT not installed. Install with: pip install tensorrt")
    logger.info("Skipping TensorRT engine build -- showing export pipeline only")
    return  # or skip to next section
```
[VERIFIED: mixed_precision/fp8_transformer_engine.py uses this pattern]

### Pattern 3: TensorRT 10.x Inference Pipeline
**What:** Modern TRT 10 API using set_tensor_address and execute_async_v3
**When to use:** INFER-01 TensorRT tutorial
**Example:**
```python
# Build engine
logger_trt = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger_trt)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger_trt)
with open(onnx_path, "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)

serialized = builder.build_serialized_network(network, config)
runtime = trt.Runtime(logger_trt)
engine = runtime.deserialize_cuda_engine(serialized)

# Run inference
context = engine.create_execution_context()
input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

input_tensor = torch.randn(1, 3, 32, 32, device="cuda")
output_tensor = torch.empty(1, 10, device="cuda")

context.set_tensor_address(input_name, input_tensor.data_ptr())
context.set_tensor_address(output_name, output_tensor.data_ptr())

stream = torch.cuda.Stream()
context.execute_async_v3(stream.cuda_stream)
stream.synchronize()
```
[VERIFIED: tested on this environment with TensorRT 10.12.0]

### Pattern 4: ONNX Runtime with Optimization Levels
**What:** ORT session with configurable graph optimization
**When to use:** INFER-02 ONNX tutorial
**Example:**
```python
import onnxruntime as ort

# Compare optimization levels
for level_name, level in [
    ("Disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
    ("Basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
    ("Extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
    ("All", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
]:
    so = ort.SessionOptions()
    so.graph_optimization_level = level
    session = ort.InferenceSession(onnx_path, so, providers=providers)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_data})
```
[VERIFIED: all four optimization levels work on this environment]

### Anti-Patterns to Avoid
- **Using deprecated TRT APIs:** Do NOT use `execute_v2()` or `allocate_buffers()` patterns from older TRT tutorials. TRT 10.x uses `set_tensor_address()` + `execute_async_v3()`. [VERIFIED: tested]
- **Assuming CUDA EP in ORT:** Do NOT assume CUDAExecutionProvider is available. Always check `ort.get_available_providers()` and fall back to CPUExecutionProvider. [VERIFIED: CUDA EP not functional on this system]
- **Using dynamo ONNX export:** Do NOT use `torch.onnx.export(dynamo=True)` -- requires `onnxscript` which is not installed. Use legacy export with `opset_version=17`. [VERIFIED: dynamo export fails]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Benchmark timing | Custom timing loops | `benchmark` decorator from `utils/benchmark.py` | Handles CUDA sync, memory tracking, consistent format [VERIFIED: codebase] |
| Comparison tables | Custom print formatting | `compare_results` / `print_benchmark_table` from `utils/benchmark.py` | Consistent output across all tutorials [VERIFIED: codebase] |
| Model creation | New model architectures | `SimpleCNN` from `utils/models.py` | Decision D-01, consistent with phases 3-4 [VERIFIED: codebase] |
| Device detection | Custom CUDA checks | `get_device` / `print_device_info` from `utils/device.py` | Handles CPU fallback, consistent logging [VERIFIED: codebase] |
| ONNX export | Custom export wrapper | `torch.onnx.export()` with opset_version=17 | Standard PyTorch API, well-tested [VERIFIED: works on this env] |

## Common Pitfalls

### Pitfall 1: TensorRT ONNX Tensor Naming
**What goes wrong:** ONNX export generates opaque tensor names (e.g., "input.1", "30") that differ between export runs if input_names/output_names are not specified.
**Why it happens:** torch.onnx.export auto-generates names from the computation graph.
**How to avoid:** Always pass `input_names=["input"]` and `output_names=["output"]` to `torch.onnx.export()` for consistent, readable tensor names in the TRT engine.
**Warning signs:** TRT engine has names like "input.1" or numeric names.
[VERIFIED: observed "input.1" and "30" in test output]

### Pitfall 2: Missing CUDA Synchronization in Benchmarks
**What goes wrong:** GPU benchmarks report incorrect (too fast) times because CUDA operations are asynchronous.
**Why it happens:** `time.perf_counter()` measures CPU time, not GPU completion time.
**How to avoid:** The `benchmark` decorator already handles `torch.cuda.synchronize()`. For TRT benchmarks using custom streams, call `stream.synchronize()` before stopping the timer.
**Warning signs:** Suspiciously fast inference times, inconsistent measurements.
[VERIFIED: benchmark.py already handles this]

### Pitfall 3: ONNX Runtime Provider Mismatch
**What goes wrong:** Tutorial crashes or silently falls back when requesting CUDAExecutionProvider that isn't available.
**Why it happens:** `onnxruntime` (CPU) and `onnxruntime-gpu` can conflict. On this system, only CPUExecutionProvider is functional.
**How to avoid:** Always check `ort.get_available_providers()` and select from available providers. Log which provider is active.
**Warning signs:** UserWarning about specified provider not in available provider names.
[VERIFIED: CUDAExecutionProvider not available on this system despite onnxruntime-gpu installed]

### Pitfall 4: TorchScript Tracing with Control Flow
**What goes wrong:** Traced model bakes in one code path, ignoring if/else branches.
**Why it happens:** Tracing records operations executed with the specific input, not the model's full logic.
**How to avoid:** This is intentionally demonstrated in the tutorial. Show a model with branching, trace it, and demonstrate the failure. Then show torch.jit.script as the solution.
**Warning signs:** TracerWarning about if/else statements.
[ASSUMED: standard TorchScript behavior, well-documented]

### Pitfall 5: TensorRT Engine Build Time
**What goes wrong:** TRT engine build takes a long time (tens of seconds for even simple models), making the tutorial feel slow.
**Why it happens:** TRT optimizes kernel selection, layer fusion, memory planning during build.
**How to avoid:** Log that build time is expected and one-time. Keep the model small (SimpleCNN). Optionally save/load the serialized engine.
**Warning signs:** Tutorial appears to hang during engine build.
[VERIFIED: observed multi-second build during testing]

## Code Examples

### ONNX Export with Explicit Names
```python
# Source: verified on this environment
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,  # Fixed batch size per D-02
)
```
[VERIFIED: tested with SimpleCNN, produces clean tensor names]

### ORT Provider Detection and Fallback
```python
# Source: verified on this environment
providers = []
available = ort.get_available_providers()
if "CUDAExecutionProvider" in available:
    providers.append("CUDAExecutionProvider")
    logger.info("Using CUDA execution provider")
providers.append("CPUExecutionProvider")
logger.info(f"Available providers: {available}")
logger.info(f"Selected providers: {providers}")

session = ort.InferenceSession(onnx_path, so, providers=providers)
logger.info(f"Active provider: {session.get_providers()}")
```
[VERIFIED: tested on this system]

### TorchScript Tracing vs Scripting
```python
# Source: verified on this environment
model = SimpleCNN().eval()
dummy = torch.randn(1, 3, 32, 32)

# Tracing -- records operations for a specific input
traced_model = torch.jit.trace(model, dummy)

# Scripting -- analyzes Python code, handles control flow
scripted_model = torch.jit.script(model)
```
[VERIFIED: both work with SimpleCNN]

### Benchmark Methodology Recommendation
```python
WARMUP_ITERATIONS = 10   # Discard initial runs (JIT warmup, cache effects)
NUM_ITERATIONS = 100     # Enough for stable timing
BATCH_SIZE = 64          # Moderate batch, not too large for tutorial speed

@benchmark
def run_inference(model, inputs, num_iters):
    with torch.inference_mode():
        for _ in range(num_iters):
            _ = model(inputs)
    return None
```
[ASSUMED: methodology based on prior phase patterns]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TRT execute_v2() with buffer arrays | set_tensor_address() + execute_async_v3() | TensorRT 10.x | Must use new API; old examples on web are outdated [VERIFIED: TRT 10.12] |
| torch.onnx.export (TorchScript-based) | torch.onnx.export(dynamo=True) | PyTorch 2.8 deprecation warning | Legacy still works, dynamo is future. Use legacy since onnxscript not installed [VERIFIED: deprecation warning observed] |
| Binding-based TRT memory management | PyTorch tensor data_ptr() for TRT addresses | TRT 8+ | Simpler integration, no need for pycuda [VERIFIED: tested] |

**Deprecated/outdated:**
- `execute_v2()` in TensorRT: replaced by `execute_async_v3()` with tensor address API
- TorchScript-based ONNX export: deprecated in PyTorch 2.8, will be replaced by dynamo export in 2.9 (but still functional)

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | torch_tensorrt adds unnecessary dependency vs raw TRT API for educational purpose | Alternatives Considered | Low -- torch_tensorrt could simplify code but raw API is more instructive |
| A2 | TorchScript tracing fails silently on control flow (standard behavior) | Pitfall 4 | Low -- well-documented PyTorch behavior |
| A3 | Warmup 10 + 100 iterations is sufficient for stable benchmarks | Code Examples | Low -- could adjust if measurements are noisy |

## Open Questions

1. **ONNX Runtime GPU Provider**
   - What we know: CUDAExecutionProvider is not functional despite onnxruntime-gpu being installed (version conflict with onnxruntime CPU package)
   - What's unclear: Whether this is fixable by reinstalling onnxruntime-gpu only
   - Recommendation: Tutorial should handle both cases gracefully. Benchmark ORT CPU vs PyTorch CPU for fair comparison, and show GPU provider detection code that logs what's available. This is actually good for the tutorial -- demonstrates real-world provider handling.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PyTorch | All tutorials | Yes | 2.8.0+cu126 | -- |
| CUDA | TRT + GPU benchmarks | Yes | 12.6 | CPU fallback in tutorials |
| TensorRT | INFER-01 | Yes | 10.12.0.36 | Graceful skip (D-03) |
| ONNX | INFER-01, INFER-02 | Yes | 1.19.0 | -- |
| ONNX Runtime | INFER-02 | Yes | 1.23.2 (CPU only) | CPUExecutionProvider works |
| ORT CUDAExecutionProvider | INFER-02 (optional) | No | -- | CPUExecutionProvider (benchmark CPU vs CPU) |
| TorchScript | INFER-03 | Yes | Built into PyTorch | -- |

**Missing dependencies with no fallback:**
- None -- all critical dependencies available

**Missing dependencies with fallback:**
- ORT CUDAExecutionProvider: Not functional. Fallback to CPUExecutionProvider with fair CPU-vs-CPU comparison.

## Sources

### Primary (HIGH confidence)
- Environment verification via Python imports: PyTorch 2.8.0, TensorRT 10.12.0, ONNX 1.19.0, ORT 1.23.2
- Full pipeline testing: ONNX export, TRT engine build, TRT inference, ORT inference, TorchScript trace/script -- all verified on this machine
- Codebase inspection: utils/benchmark.py, utils/models.py, utils/__init__.py, profiling/reference_tutorial.py, mixed_precision/fp8_transformer_engine.py

### Secondary (MEDIUM confidence)
- TensorRT 10.x API patterns (set_tensor_address, execute_async_v3) verified by successful execution

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries verified installed and functional via import + test
- Architecture: HIGH - patterns verified from existing codebase tutorials
- Pitfalls: HIGH - all except A2 verified through direct testing on this environment

**Research date:** 2026-04-13
**Valid until:** 2026-05-13 (stable domain, installed versions pinned)
