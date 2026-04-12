# Phase 3: Mixed Precision Training - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-12
**Phase:** 03-mixed-precision-training
**Areas discussed:** FP8 / Transformer Engine scope, Benchmark methodology

---

## FP8 / Transformer Engine Scope

### Question 1: What model should the FP8/Transformer Engine tutorial use?

| Option | Description | Selected |
|--------|-------------|----------|
| Small transformer (Recommended) | Introduce a lightweight transformer built with TE layers. Add to utils/models.py. | |
| SimpleMLP with te.Linear | Swap nn.Linear in SimpleMLP with te.Linear. Simpler but less realistic. | |
| Both approaches | Show te.Linear swap first, then full TE transformer model. | |
| Other (user input) | Vision Transformer style model like Wan2.2 | ✓ |

**User's choice:** Vision Transformer (ViT) style model, inspired by architectures like Wan2.2
**Notes:** User wants a ViT with patch embeddings + transformer encoder blocks, built with Transformer Engine layers. Added to utils/models.py.

### Question 2: Should the FP8 tutorial ONLY cover Hopper+ or also show fallback?

| Option | Description | Selected |
|--------|-------------|----------|
| Hopper-only with clear messaging | Run on FP8-capable GPUs only. Clear check and exit on older hardware. | |
| Conditional sections | Detect GPU capability. FP8 on Hopper+, BF16 on Ampere, FP16 on older. | ✓ |
| You decide | Claude determines the best fallback approach. | |

**User's choice:** Conditional sections
**Notes:** Everyone can run the tutorial regardless of GPU generation.

---

## Benchmark Methodology

### Question 1: What should the mixed precision before/after benchmarks measure?

| Option | Description | Selected |
|--------|-------------|----------|
| Training throughput (samples/sec) | Direct speed metric — run N iterations and divide. | ✓ |
| Peak GPU memory | Max memory during training. Already supported by benchmark decorator. | ✓ |
| Loss convergence comparison | Train a few epochs, show loss curves are similar. | |
| All of the above | Comprehensive: throughput + memory + convergence. | |

**User's choice:** Training throughput (samples/sec) and Peak GPU memory
**Notes:** Skip loss convergence — keep focus on performance metrics, not training quality.

---

## Claude's Discretion

- Training loop structure and iteration count
- ViT model sizing (layers, hidden dim, patch size)
- Tutorial file naming
- GPU capability detection implementation
- Training loop helper sharing across tutorials

## Deferred Ideas

None — discussion stayed within phase scope
