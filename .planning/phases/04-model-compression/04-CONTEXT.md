# Phase 4: Model Compression - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Two standalone tutorials teaching ML engineers to reduce model size while preserving accuracy. Covers pruning techniques — both unstructured and structured — with iterative prune-then-fine-tune workflow (COMP-01), and knowledge distillation using classic Hinton soft-label approach with a same-family teacher-student pair (COMP-02). Each tutorial demonstrates measurable compression with model size and inference speed comparisons.

</domain>

<decisions>
## Implementation Decisions

### Pruning Tutorial (COMP-01)
- **D-01:** Tutorial covers BOTH unstructured (L1 magnitude on individual weights) and structured (filter/channel removal) pruning in a single tutorial. Unstructured section uses `torch.nn.utils.prune`; structured section physically removes channels and adjusts subsequent layers.
- **D-02:** Includes iterative prune-then-fine-tune loop: train → prune → fine-tune → measure. Shows the practical workflow, not just the pruning API. Multiple sparsity levels demonstrated.
- **D-03:** Explicitly logs that unstructured pruning does NOT produce inference speedup on standard GPUs (needs sparse hardware), while structured pruning shows real speedup.
- **D-04:** Uses `SimpleCNN` from `utils/models.py` as the pruning subject — keep focus on the pruning technique, not the model.

### Distillation Tutorial (COMP-02)
- **D-05:** Teacher and student are SAME architecture family — both SimpleCNN variants. Teacher is a larger SimpleCNN (more filters/layers), student is a smaller SimpleCNN. Need to add these size variants to `utils/models.py` or define them within the tutorial.
- **D-06:** Loss formulation is classic Hinton distillation: weighted combination of KL-divergence between teacher/student soft logits (with temperature scaling) and standard cross-entropy on ground truth labels. Both loss components demonstrated.
- **D-07:** Tutorial trains teacher first, then distills into student, comparing student-with-distillation vs student-trained-from-scratch.

### Folder Placement
- **D-08:** Pruning tutorial goes in `pruning/` folder. Distillation tutorial goes in `compression/` folder. Matches Phase 1 D-01 folder structure. Both folders already scaffolded.

### Benchmark Metrics
- **D-09:** Both tutorials measure and compare: model size (total parameter count + saved model file size on disk) and inference speed (latency/throughput per batch). Use existing `benchmark` decorator and `print_benchmark_table` from `utils/benchmark.py`.
- **D-10:** Accuracy tracking and per-layer sparsity analysis are explicitly excluded — focus stays on performance metrics (size and speed), consistent with Phase 3 D-05.

### Claude's Discretion
- SimpleCNN sizing for teacher and student variants (filter counts, layer depth)
- Specific sparsity levels to demonstrate in the pruning tutorial
- Number of fine-tuning epochs per pruning iteration
- Temperature and alpha hyperparameters for distillation loss
- Tutorial file naming within pruning/ and compression/ (following Phase 1 D-02: topic-based, no numbering)
- Whether teacher/student size variants go in utils/models.py or are defined inline in the tutorial
- Training data setup (synthetic vs CIFAR-style — keep it standalone)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

No external specs — requirements fully captured in decisions above and in REQUIREMENTS.md (COMP-01, COMP-02).

### Existing Code (Phases 1-3 output)
- `utils/__init__.py` — Public API for shared utilities
- `utils/benchmark.py` — Benchmark decorator and comparison table utilities
- `utils/models.py` — SimpleCNN, SimpleMLP, SimpleViT, get_sample_batch (SimpleCNN is the pruning/distillation subject)
- `utils/device.py` — get_device, print_device_info
- `utils/logging_config.py` — setup_logging
- `profiling/reference_tutorial.py` — Convention reference: docstring format, import pattern, logging style, benchmark table output

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `benchmark` decorator: Measures wall-clock time and GPU memory — use for inference speed comparisons
- `compare_results` / `print_benchmark_table`: Formatted comparison tables — use for final before/after output
- `SimpleCNN`: 3-conv + 2-FC layer CNN — primary subject for both pruning and distillation tutorials
- `get_sample_batch`: Generates input tensors with correct device placement
- `get_device` / `print_device_info`: Device detection and info logging

### Established Patterns
- Tutorial structure: module docstring → imports → constants → main() → section headers with print() → logging for details → benchmark table at end
- Import pattern: `sys.path.insert(0, ...)` then import from utils
- All tutorials standalone-runnable: `python tutorial_name.py`
- Before/after benchmarks as formatted comparison tables (Phase 1 D-05)

### Integration Points
- Pruning tutorial goes in `pruning/` folder (already scaffolded, empty)
- Distillation tutorial goes in `compression/` folder (already scaffolded, empty)
- May need larger/smaller SimpleCNN variants in `utils/models.py` for teacher/student pair
- `torch.nn.utils.prune` is stdlib PyTorch — no new dependencies needed

</code_context>

<specifics>
## Specific Ideas

- Pruning tutorial should make the unstructured vs structured distinction very clear — log explicitly that unstructured pruning creates zeros but doesn't reduce actual computation on standard GPUs, while structured pruning physically removes channels for real speedup
- Distillation tutorial should compare three outcomes: teacher accuracy/speed, student-from-scratch accuracy/speed, distilled-student accuracy/speed — making the value of distillation visually obvious in the benchmark table

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-model-compression*
*Context gathered: 2026-04-13*
