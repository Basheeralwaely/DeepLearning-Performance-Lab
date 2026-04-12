# DeepLearning Performance Lab

## What This Is

A performance-focused AI engineering repository featuring hands-on Python tutorials that cover training tricks, fast inference techniques, TensorRT acceleration, pruning, quantization, and distributed multi-GPU training. Each tutorial is a standalone `.py` file with rich logging output and detailed inline explanations, targeting ML engineers who want to make their models faster.

## Core Value

Every tutorial must be runnable, well-logged, and teach one specific performance technique with measurable before/after results.

## Requirements

### Validated

- [x] Each tutorial is a standalone .py file with rich logging and inline explanations — Validated in Phase 1: Repository Foundation
- [x] Repo organized by technique (one folder per technique) — Validated in Phase 1: Repository Foundation
- [x] Measurable performance comparisons (before/after benchmarks in each tutorial) — Validated in Phase 1: Repository Foundation
- [x] Profiling tutorials (PyTorch profiler, bottleneck analysis, memory tracking) — Validated in Phase 2: Profiling & Diagnostics
- [x] Mixed precision training tutorials (FP16/BF16, AMP, loss scaling) — Validated in Phase 3: Mixed Precision Training

### Active

- [ ] Multi-GPU distributed training tutorials (DDP, FSDP, model/data parallelism)
- [ ] TensorRT acceleration tutorials (model export, optimization, benchmarking)
- [ ] Pruning tutorials (structured/unstructured pruning, magnitude-based, iterative)
- [ ] Quantization tutorials (post-training quantization, quantization-aware training, INT8)
- [ ] Inference optimization tutorials (batching, TorchScript, ONNX export)

### Out of Scope

- Jupyter notebooks — .py files only
- Framework-agnostic coverage — PyTorch-centric (with NVIDIA tooling where relevant)
- Beginner ML fundamentals — assumes ML engineering background
- Model architecture tutorials — focus is purely on performance techniques
- Cloud deployment / MLOps — focus is local/cluster training and inference

## Context

- Repository name: DeepLearning-Performance-Lab
- Target audience: ML engineers already familiar with training models
- Tutorial format: Python scripts with extensive logging (print/logging), detailed comments, and docstrings
- Primary framework: PyTorch with NVIDIA ecosystem (TensorRT, CUDA, NCCL)
- Each tutorial should demonstrate a technique on a simple model so the focus stays on the performance trick, not the model itself

## Constraints

- **Format**: .py files only — no notebooks, no markdown-only docs
- **Logging**: Every tutorial must produce rich console output showing what's happening and why
- **Standalone**: Each tutorial should be independently runnable
- **Framework**: PyTorch + NVIDIA tooling (TensorRT, CUDA)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Organize by technique | ML engineers search for specific tricks, not difficulty levels | Validated (Phase 1) |
| .py over notebooks | Richer logging, easier to run on clusters, version-control friendly | Validated (Phase 1) |
| PyTorch-centric | Dominant framework for performance-focused ML engineering | Validated (Phase 1) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-12 after Phase 3 completion*
