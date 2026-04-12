# Phase 3: Mixed Precision Training - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Three standalone tutorials teaching ML engineers to train models faster using reduced numerical precision. Covers AMP with autocast/GradScaler (PREC-01), BF16 vs FP16 tradeoffs with comparison benchmarks (PREC-02), and FP8 training with Transformer Engine on supported hardware (PREC-03). Each tutorial demonstrates a precision technique with measurable throughput and memory improvements.

</domain>

<decisions>
## Implementation Decisions

### FP8 / Transformer Engine Scope
- **D-01:** FP8 tutorial uses a small Vision Transformer (ViT) model built with Transformer Engine layers — patch embeddings + transformer encoder blocks, sized for tutorial demos (not production scale). Inspired by architectures like Wan2.2. Add this model to `utils/models.py` alongside SimpleCNN/SimpleMLP.
- **D-02:** FP8 tutorial uses conditional sections based on GPU capability detection: FP8 code runs on Hopper+ (H100/H200), BF16 fallback on Ampere GPUs, FP16 fallback on older hardware. Every user can run the tutorial and learn something regardless of hardware.
- **D-03:** AMP tutorial (PREC-01) and BF16 vs FP16 tutorial (PREC-02) use existing SimpleCNN/SimpleMLP from `utils/models.py` — keep focus on the precision technique, not the model.

### Benchmark Methodology
- **D-04:** Before/after benchmarks measure two metrics: training throughput (samples/sec) and peak GPU memory usage. Use existing `benchmark` decorator and `print_benchmark_table` from `utils/benchmark.py`.
- **D-05:** Loss convergence comparison is explicitly excluded — these are performance tutorials, not training quality tutorials. Focus stays on speed and memory savings.

### Claude's Discretion
- Training loop structure (how many iterations/epochs to run for meaningful benchmarks)
- ViT model sizing (number of layers, hidden dim, patch size) — small enough for tutorial, large enough to show FP8 benefits
- Tutorial file naming within `mixed_precision/` (following Phase 1 D-02: topic-based, no numbering)
- How to structure the GPU capability detection and conditional sections in the FP8 tutorial
- Whether AMP and BF16/FP16 tutorials share a training loop helper or each define their own

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

No external specs — requirements fully captured in decisions above and in REQUIREMENTS.md (PREC-01 through PREC-03).

### Existing Code (Phase 1 output)
- `utils/__init__.py` — Public API for shared utilities
- `utils/benchmark.py` — Benchmark decorator and comparison table utilities
- `utils/models.py` — SimpleCNN, SimpleMLP, get_sample_batch (will be extended with SimpleViT for FP8 tutorial)
- `utils/device.py` — get_device, print_device_info (GPU capability detection for conditional sections)
- `utils/logging_config.py` — setup_logging
- `profiling/reference_tutorial.py` — Convention reference: docstring format, import pattern, logging style, benchmark table output

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `benchmark` decorator: Measures wall-clock time and GPU memory — use for throughput and memory comparisons
- `compare_results` / `print_benchmark_table`: Formatted comparison tables — use for final before/after output
- `SimpleCNN` / `SimpleMLP`: Lightweight models for AMP and BF16/FP16 tutorials
- `get_device` / `print_device_info`: Device detection — extend for GPU compute capability checks (Hopper/Ampere/older)
- `get_sample_batch`: Generates input tensors with correct device placement

### Established Patterns
- Tutorial structure: module docstring -> imports -> constants -> main() -> section headers with print() -> logging for details -> benchmark table at end
- Import pattern: `sys.path.insert(0, ...)` then import from utils
- All tutorials standalone-runnable: `python tutorial_name.py`

### Integration Points
- New tutorials go in `mixed_precision/` folder (already scaffolded with .gitkeep)
- `utils/models.py` needs a new SimpleViT class for the FP8 tutorial
- `utils/device.py` may need GPU compute capability detection for conditional FP8/BF16/FP16 sections
- May need to add `transformer-engine` as an optional dependency for PREC-03

</code_context>

<specifics>
## Specific Ideas

- FP8 tutorial model should be a Vision Transformer (ViT) style architecture, inspired by models like Wan2.2 — transformer-based, not CNN-based, to naturally leverage Transformer Engine's FP8 support
- Conditional hardware sections ensure the tutorial is educational regardless of GPU generation — users see what FP8 does even if they can only run BF16 or FP16

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-mixed-precision-training*
*Context gathered: 2026-04-12*
