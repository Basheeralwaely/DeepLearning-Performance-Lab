# Requirements: DeepLearning Performance Lab

**Defined:** 2026-04-12
**Core Value:** Every tutorial must be runnable, well-logged, and teach one specific performance technique with measurable before/after results.

## v1 Requirements

### Distributed Training

- [ ] **DIST-01**: User can learn DDP setup with multi-GPU launch, process groups, and gradient sync
- [ ] **DIST-02**: User can learn FSDP for sharding large models across GPUs
- [ ] **DIST-03**: User can learn model parallelism by splitting layers across GPUs
- [ ] **DIST-04**: User can learn DeepSpeed ZeRO stages and offloading

### Mixed Precision

- [ ] **PREC-01**: User can learn AMP with autocast and GradScaler
- [ ] **PREC-02**: User can learn BF16 vs FP16 tradeoffs with comparison benchmarks
- [ ] **PREC-03**: User can learn FP8 training with Transformer Engine

### Inference Optimization

- [ ] **INFER-01**: User can learn TensorRT model export, optimization, and benchmarking
- [ ] **INFER-02**: User can learn ONNX export and runtime optimization
- [ ] **INFER-03**: User can learn TorchScript JIT compilation (tracing vs scripting)

### Model Compression

- [ ] **COMP-01**: User can learn pruning techniques (structured/unstructured, magnitude-based)
- [ ] **COMP-02**: User can learn knowledge distillation for model compression

### Profiling & Training Tricks

- [ ] **PROF-01**: User can learn PyTorch Profiler with trace export and bottleneck analysis
- [ ] **PROF-02**: User can learn GPU memory profiling, OOM debugging, and gradient checkpointing
- [ ] **PROF-03**: User can learn DataLoader tuning (prefetch, num_workers, pinned memory)
- [ ] **PROF-04**: User can learn torch.compile modes, graph breaks, and speedup measurement

### Repository Structure

- [ ] **REPO-01**: Each tutorial is a standalone .py with rich logging and inline explanations
- [ ] **REPO-02**: Repo organized by technique (one folder per technique)
- [ ] **REPO-03**: Each tutorial includes before/after performance benchmarks

## v2 Requirements

### Quantization

- **QUANT-01**: Post-training quantization (INT8, calibration datasets)
- **QUANT-02**: Quantization-aware training (fake quantize, fine-tuning)

### Inference Advanced

- **INFER-04**: Dynamic batching strategies for throughput optimization

### Mixed Precision Advanced

- **PREC-04**: Loss scaling deep dive (dynamic vs static, debugging underflow)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Jupyter notebooks | .py files only — richer logging, cluster-friendly, VCS-friendly |
| Beginner ML fundamentals | Target audience is ML engineers with existing knowledge |
| Model architecture tutorials | Focus is purely on performance techniques |
| Cloud deployment / MLOps | Focus is local/cluster training and inference |
| Framework-agnostic coverage | PyTorch-centric with NVIDIA tooling |
| Mobile/edge deployment | Desktop/server GPU focus for v1 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DIST-01 | — | Pending |
| DIST-02 | — | Pending |
| DIST-03 | — | Pending |
| DIST-04 | — | Pending |
| PREC-01 | — | Pending |
| PREC-02 | — | Pending |
| PREC-03 | — | Pending |
| INFER-01 | — | Pending |
| INFER-02 | — | Pending |
| INFER-03 | — | Pending |
| COMP-01 | — | Pending |
| COMP-02 | — | Pending |
| PROF-01 | — | Pending |
| PROF-02 | — | Pending |
| PROF-03 | — | Pending |
| PROF-04 | — | Pending |
| REPO-01 | — | Pending |
| REPO-02 | — | Pending |
| REPO-03 | — | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 0
- Unmapped: 19 ⚠️

---
*Requirements defined: 2026-04-12*
*Last updated: 2026-04-12 after initial definition*
