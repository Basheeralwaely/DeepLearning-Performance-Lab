# Roadmap: DeepLearning Performance Lab

## Overview

This roadmap delivers a complete set of performance-focused PyTorch tutorials, starting with repository structure and profiling (measure first), then progressing through training optimizations (mixed precision, compression), inference acceleration, and finally distributed multi-GPU training. Each phase produces standalone, runnable tutorials with rich logging and before/after benchmarks.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Repository Foundation** - Scaffold repo structure, shared utilities, and establish tutorial conventions
- [ ] **Phase 2: Profiling & Diagnostics** - Tutorials for measuring and understanding performance bottlenecks
- [ ] **Phase 3: Mixed Precision Training** - Tutorials for FP16, BF16, FP8, and automatic mixed precision
- [ ] **Phase 4: Model Compression** - Tutorials for pruning and knowledge distillation
- [ ] **Phase 5: Inference Optimization** - Tutorials for TensorRT, ONNX, and TorchScript acceleration
- [ ] **Phase 6: Distributed Training** - Tutorials for DDP, FSDP, model parallelism, and DeepSpeed

## Phase Details

### Phase 1: Repository Foundation
**Goal**: Repository structure is established with conventions that every tutorial will follow
**Depends on**: Nothing (first phase)
**Requirements**: REPO-01, REPO-02, REPO-03
**Success Criteria** (what must be TRUE):
  1. Repository has one folder per technique category with clear naming
  2. A reference tutorial exists demonstrating the standalone .py format with rich logging, docstrings, and inline comments
  3. The reference tutorial includes before/after benchmark output showing measurable performance comparison
  4. Any shared utilities (logging setup, benchmark helpers, simple model definitions) are importable by tutorials
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md — Folder structure + shared utils package
- [x] 01-02-PLAN.md — Reference tutorial + README navigation

### Phase 2: Profiling & Diagnostics
**Goal**: Users can identify and measure performance bottlenecks before applying optimizations
**Depends on**: Phase 1
**Requirements**: PROF-01, PROF-02, PROF-03, PROF-04
**Success Criteria** (what must be TRUE):
  1. User can run PyTorch Profiler tutorial and get a trace file with CPU/GPU activity and bottleneck analysis logged
  2. User can run memory profiling tutorial and see GPU memory breakdown, OOM strategies, and gradient checkpointing impact
  3. User can run DataLoader tuning tutorial and see throughput differences across num_workers, prefetch, and pinned memory configs
  4. User can run torch.compile tutorial and see mode comparisons, graph break detection, and speedup measurements
**Plans**: 2 plans

Plans:
- [ ] 02-01-PLAN.md — PyTorch Profiler trace export + GPU memory profiling with OOM and checkpointing
- [ ] 02-02-PLAN.md — DataLoader tuning sweep + torch.compile diagnostics with graph break detection

### Phase 3: Mixed Precision Training
**Goal**: Users can train models faster using reduced precision with understanding of tradeoffs
**Depends on**: Phase 1
**Requirements**: PREC-01, PREC-02, PREC-03
**Success Criteria** (what must be TRUE):
  1. User can run AMP tutorial and see autocast/GradScaler in action with before/after training speed comparison
  2. User can run BF16 vs FP16 tutorial and see numerical stability differences, hardware requirements, and benchmark comparisons
  3. User can run FP8 tutorial with Transformer Engine and see training speedup on supported hardware
**Plans**: 2 plans

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Model Compression
**Goal**: Users can reduce model size while preserving accuracy using pruning and distillation
**Depends on**: Phase 1
**Requirements**: COMP-01, COMP-02
**Success Criteria** (what must be TRUE):
  1. User can run pruning tutorial and see structured/unstructured pruning applied with sparsity levels, accuracy impact, and inference speedup logged
  2. User can run distillation tutorial and see a smaller student model trained from a teacher with accuracy and speed comparisons
**Plans**: 2 plans

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

### Phase 5: Inference Optimization
**Goal**: Users can accelerate model inference using export and compilation techniques
**Depends on**: Phase 1
**Requirements**: INFER-01, INFER-02, INFER-03
**Success Criteria** (what must be TRUE):
  1. User can run TensorRT tutorial and see model export, engine optimization, and latency/throughput benchmarks vs vanilla PyTorch
  2. User can run ONNX tutorial and see model export, runtime optimization, and inference speed comparison
  3. User can run TorchScript tutorial and see tracing vs scripting approaches with JIT compilation speedup measured
**Plans**: 2 plans

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

### Phase 6: Distributed Training
**Goal**: Users can scale training across multiple GPUs using various parallelism strategies
**Depends on**: Phase 1
**Requirements**: DIST-01, DIST-02, DIST-03, DIST-04
**Success Criteria** (what must be TRUE):
  1. User can run DDP tutorial with multi-GPU launch and see process group setup, gradient synchronization, and scaling efficiency logged
  2. User can run FSDP tutorial and see model sharding across GPUs with memory savings and throughput reported
  3. User can run model parallelism tutorial and see layers split across GPUs with pipeline execution and memory distribution logged
  4. User can run DeepSpeed tutorial and see ZeRO stages, CPU offloading, and memory/speed tradeoffs benchmarked
**Plans**: 2 plans

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD
- [ ] 06-03: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Repository Foundation | 0/2 | Not started | - |
| 2. Profiling & Diagnostics | 0/2 | Not started | - |
| 3. Mixed Precision Training | 0/2 | Not started | - |
| 4. Model Compression | 0/2 | Not started | - |
| 5. Inference Optimization | 0/2 | Not started | - |
| 6. Distributed Training | 0/3 | Not started | - |
