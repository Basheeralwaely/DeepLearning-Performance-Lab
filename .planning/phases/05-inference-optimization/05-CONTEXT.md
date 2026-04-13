# Phase 5: Inference Optimization - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Three standalone tutorials teaching ML engineers to accelerate model inference using export and compilation techniques. Covers TensorRT engine optimization with FP16 mode (INFER-01), ONNX export and ONNX Runtime optimization (INFER-02), and TorchScript JIT compilation with both tracing and scripting approaches (INFER-03). Each tutorial benchmarks optimized inference against vanilla PyTorch baseline.

</domain>

<decisions>
## Implementation Decisions

### Model Subject
- **D-01:** All three tutorials use `SimpleCNN` from `utils/models.py` as the model subject. Consistent with Phases 3-4 — keeps focus on the export/compilation technique, not the model.

### TensorRT Tutorial (INFER-01)
- **D-02:** Core flow only: Export PyTorch → ONNX → TensorRT engine → benchmark. Show FP16 engine mode as the main optimization. No INT8 calibration (deferred to v2 QUANT-01 scope), no dynamic shapes, no workspace tuning.
- **D-03:** Graceful dependency handling — try-import `tensorrt` at top. If missing, log what's needed (install instructions) and skip TensorRT-specific sections. Tutorial still demonstrates the export pipeline concept.

### ONNX Tutorial (INFER-02)
- **D-04:** Export PyTorch model to ONNX format, run through ONNX Runtime with graph optimizations, benchmark against vanilla PyTorch inference.
- **D-05:** Graceful dependency handling — try-import `onnxruntime` at top. If missing, log install instructions and skip ONNX Runtime sections.

### TorchScript Tutorial (INFER-03)
- **D-06:** Covers BOTH tracing and scripting side-by-side with pros/cons. Tracing for the common case (fixed shape, no control flow), scripting for models with branching logic. Both approaches benchmarked.
- **D-07:** No external dependencies required — TorchScript is built into PyTorch. This tutorial is guaranteed runnable for all users.

### Dependency & Hardware Fallback
- **D-08:** All tutorials follow the graceful skip pattern from Phase 3's FP8 tutorial: try-import at top, if missing log what's needed and skip that section. Ensures every tutorial is educational even without optional deps.
- **D-09:** Three separate standalone .py files — one technique per tutorial. Consistent with project convention. No combined fallback tutorial.

### Benchmark Metrics
- **D-10:** Inherited from prior phases (Phase 3 D-05, Phase 4 D-09/D-10): measure inference latency and throughput against vanilla PyTorch baseline. Use existing `benchmark` decorator and `print_benchmark_table` from `utils/benchmark.py`. No accuracy tracking.

### Claude's Discretion
- Specific benchmark methodology (warmup runs, number of iterations, batch sizes)
- Tutorial file naming within `inference/` (following Phase 1 D-02: topic-based, no numbering)
- ONNX Runtime optimization levels and graph optimization details
- TensorRT builder configuration details (max workspace size, etc.)
- How to structure the graceful skip sections (decorator, context manager, or inline try/except)
- Training data setup for any pre-export model preparation (synthetic data, keep it standalone)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

No external specs — requirements fully captured in decisions above and in REQUIREMENTS.md (INFER-01, INFER-02, INFER-03).

### Existing Code (Phases 1-4 output)
- `utils/__init__.py` — Public API for shared utilities
- `utils/benchmark.py` — Benchmark decorator and comparison table utilities
- `utils/models.py` — SimpleCNN, SimpleMLP, SimpleViT, get_sample_batch (SimpleCNN is the inference subject)
- `utils/device.py` — get_device, print_device_info, get_gpu_capability
- `utils/logging_config.py` — setup_logging
- `profiling/reference_tutorial.py` — Convention reference: docstring format, import pattern, logging style, benchmark table output

### Prior Phase Patterns
- `mixed_precision/fp8_transformer_engine.py` — Reference for graceful dependency skip pattern (try-import with fallback)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `benchmark` decorator: Measures wall-clock time and GPU memory — use for latency/throughput comparisons
- `compare_results` / `print_benchmark_table`: Formatted comparison tables — use for optimized vs baseline output
- `SimpleCNN`: 3-conv + 2-FC layer CNN — inference subject for all three tutorials
- `get_sample_batch`: Generates input tensors with correct device placement — use for inference input
- `get_device` / `print_device_info`: Device detection and info logging

### Established Patterns
- Tutorial structure: module docstring → imports → constants → main() → section headers with print() → logging for details → benchmark table at end
- Import pattern: `sys.path.insert(0, ...)` then import from utils
- All tutorials standalone-runnable: `python tutorial_name.py`
- Graceful dep handling: try-import at top, conditional sections based on availability (Phase 3 FP8 tutorial)

### Integration Points
- All tutorials go in `inference/` folder (already scaffolded with .gitkeep)
- No new utils expected — existing benchmark and model utilities cover the needs
- TensorRT tutorial needs ONNX as intermediate format (torch → ONNX → TensorRT)

</code_context>

<specifics>
## Specific Ideas

- TensorRT tutorial should make the export pipeline clear: PyTorch → ONNX → TensorRT engine, showing each step with logging so users understand the conversion chain
- TorchScript tutorial should clearly demonstrate when tracing fails (control flow) and when scripting is needed, making the tradeoff practical rather than theoretical

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-inference-optimization*
*Context gathered: 2026-04-13*
