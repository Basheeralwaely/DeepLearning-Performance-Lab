# Phase 5: Inference Optimization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-13
**Phase:** 05-inference-optimization
**Areas discussed:** Model subject & scope per tutorial, Dependency & hardware fallback

---

## Model subject & scope per tutorial

### Model selection

| Option | Description | Selected |
|--------|-------------|----------|
| SimpleCNN for all three | Consistent with Phases 3-4. Keeps focus on the technique. | ✓ |
| SimpleCNN + SimpleViT | Use SimpleCNN for TorchScript, SimpleViT for TensorRT and ONNX. | |
| SimpleCNN for all, with ViT bonus section | Primary demos use SimpleCNN, TensorRT includes optional ViT section. | |

**User's choice:** SimpleCNN for all three
**Notes:** Consistency with prior phases was the deciding factor.

### TensorRT depth

| Option | Description | Selected |
|--------|-------------|----------|
| Core flow only | Export PyTorch → ONNX → TensorRT engine → benchmark. FP16 engine mode. | ✓ |
| Core + INT8 calibration | Add INT8 engine build with calibration dataset. Overlaps v2 QUANT-01. | |
| Full options sweep | FP32, FP16, INT8 + dynamic shapes + workspace tuning. | |

**User's choice:** Core flow only
**Notes:** INT8 calibration deferred to v2 quantization scope.

### TorchScript scope

| Option | Description | Selected |
|--------|-------------|----------|
| Both tracing and scripting | Side-by-side with pros/cons. Matches INFER-03 requirement. | ✓ |
| Tracing only | Simpler tutorial, mention scripting as alternative. | |

**User's choice:** Both tracing and scripting
**Notes:** INFER-03 explicitly mentions both approaches.

---

## Dependency & hardware fallback

### Missing dependency handling

| Option | Description | Selected |
|--------|-------------|----------|
| Graceful skip with explanation | Try-import, log what's needed, skip section. Consistent with Phase 3 FP8 pattern. | ✓ |
| Hard requirement | Fail immediately with install instructions. | |
| Mock/simulate mode | Show pre-recorded output if dep missing. | |

**User's choice:** Graceful skip with explanation
**Notes:** Follows the established pattern from Phase 3's FP8 tutorial.

### Tutorial structure

| Option | Description | Selected |
|--------|-------------|----------|
| Separate tutorials, each standalone | Three independent .py files. One technique per file. | ✓ |
| Combined with TorchScript as baseline | Single tutorial, TorchScript always runs, others optional. | |

**User's choice:** Separate tutorials, each standalone
**Notes:** Consistent with project convention of one technique per tutorial file.

---

## Claude's Discretion

- Benchmark methodology details (warmup, iterations, batch sizes)
- Tutorial file naming within inference/
- ONNX Runtime optimization levels
- TensorRT builder configuration
- Graceful skip implementation pattern
- Training data setup for pre-export model preparation

## Deferred Ideas

None — discussion stayed within phase scope.
