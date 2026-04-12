# Phase 2: Profiling & Diagnostics - Context

**Gathered:** 2026-04-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Four standalone tutorials teaching ML engineers how to measure and understand performance bottlenecks using PyTorch's profiling tools. Each tutorial focuses on the DIAGNOSTIC aspect — identifying problems and measuring impact — not on applying optimizations (that's Phases 3-6). Tutorials: PyTorch Profiler (PROF-01), GPU memory profiling (PROF-02), DataLoader tuning (PROF-03), torch.compile diagnostics (PROF-04).

</domain>

<decisions>
## Implementation Decisions

### Model Complexity for Demos
- **D-01:** Use existing SimpleCNN/SimpleMLP from `utils/models.py` across all profiling tutorials. Keep focus on the profiling TOOL, not the model.
- **D-02:** For memory profiling (PROF-02), scale up batch size and input resolution with SimpleCNN to trigger real memory pressure and OOM conditions. Do not use artificial memory fraction limits.

### DataLoader Workload
- **D-03:** DataLoader tuning tutorial (PROF-03) uses synthetic data with fake I/O delay in `__getitem__` — no dataset downloads needed. The I/O bottleneck is controllable and reproducible.
- **D-04:** Sweep 4 num_workers configurations. Claude determines the specific values based on what produces the clearest educational comparison.

### Tutorial Scope Boundaries
- **D-05:** Gradient checkpointing in PROF-02 is a profiling demo only — show memory before/after with checkpointing enabled. Do NOT deep-dive into the checkpointing API, custom checkpoint functions, or segment-level control.
- **D-06:** torch.compile in PROF-04 focuses on diagnostics: mode comparisons (default/reduce-overhead/max-autotune), detecting graph breaks, and understanding compilation overhead. Production deployment patterns belong in Phase 5.

### Profiler Output & Artifacts
- **D-07:** PyTorch Profiler tutorial (PROF-01) produces both console table output (key_averages) AND a Chrome trace JSON file. Users get immediate console insight plus a visual trace for chrome://tracing.
- **D-08:** Trace files saved to `./profiler_output/` directory in repo root. Add to `.gitignore`. All profiling artifacts go there.

### Claude's Discretion
- Specific num_workers values for the DataLoader sweep (4 values that show clear throughput differences)
- How to structure the fake I/O delay in the synthetic dataset (sleep duration, transform complexity)
- Profiler configuration details (schedule, activities, record_shapes, etc.)
- Tutorial file naming within `profiling/` folder (following D-02 from Phase 1: topic-based, no numbering)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

No external specs — requirements fully captured in decisions above and in REQUIREMENTS.md (PROF-01 through PROF-04).

### Existing Code (Phase 1 output)
- `utils/__init__.py` — Public API for shared utilities
- `utils/benchmark.py` — Benchmark decorator and comparison table utilities
- `utils/models.py` — SimpleCNN, SimpleMLP, get_sample_batch
- `utils/device.py` — get_device, print_device_info
- `utils/logging_config.py` — setup_logging
- `profiling/reference_tutorial.py` — Convention reference: docstring format, import pattern, logging style, benchmark table output

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `benchmark` decorator: Measures wall-clock time and GPU memory per function call — use for before/after comparisons
- `compare_results` / `print_benchmark_table`: Formatted comparison tables — use for final output in each tutorial
- `SimpleCNN` / `SimpleMLP`: Lightweight models for demos — scale via batch size / input size for memory pressure
- `get_sample_batch`: Generates input tensors with correct device placement
- `get_device` / `print_device_info`: Device detection and hardware info display

### Established Patterns
- Tutorial structure: module docstring -> imports -> constants -> main() -> section headers with print() -> logging for details -> benchmark table at end
- Import pattern: `sys.path.insert(0, ...)` then import from utils
- All tutorials standalone-runnable: `python tutorial_name.py`

### Integration Points
- New tutorials go in `profiling/` folder alongside `reference_tutorial.py`
- `profiler_output/` directory needs to be created and added to `.gitignore`
- May need to extend `utils/models.py` if SimpleCNN/SimpleMLP need configuration options for larger inputs

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. Key constraint: these are diagnostic/measurement tutorials, not optimization tutorials. The user should learn to IDENTIFY bottlenecks, not fix them (fixing comes in later phases).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-profiling-diagnostics*
*Context gathered: 2026-04-12*
