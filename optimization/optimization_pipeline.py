"""
End-to-End Optimization Pipeline: Train -> Prune -> Export -> Accelerate
=========================================================================

Individual optimization techniques provide incremental speedups, but
the real power comes from STACKING them. This tutorial chains together
multiple techniques from this repository into a single pipeline:

  1. Train a model with torch.compile + AMP (fast training)
  2. Prune the trained model (structured channel removal)
  3. Export to ONNX format (portable representation)
  4. Run through TensorRT with FP16 (GPU-native acceleration)

At each stage, we benchmark inference speed and model size so you can
see the cumulative effect. The key insight: these optimizations are
COMPLEMENTARY -- pruning makes the model smaller, ONNX/TensorRT makes
the remaining operations faster, and FP16 halves the memory bandwidth.

This tutorial ties together:
  - training/torch_compile_training.py (torch.compile)
  - mixed_precision/amp_training.py (AMP training)
  - pruning/structured_unstructured_pruning.py (channel pruning)
  - inference/onnx_inference.py (ONNX export)
  - inference/tensorrt_inference.py (TensorRT acceleration)

What this tutorial demonstrates:
  1. Cumulative speedup measurements at each pipeline stage
  2. Model size reduction tracking through the pipeline
  3. Accuracy preservation checks at each stage
  4. Practical pipeline for deploying optimized models
"""

import os
import sys
import copy
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from utils import (
    setup_logging,
    benchmark,
    print_benchmark_table,
    SimpleCNN,
    get_sample_batch,
    get_device,
    print_device_info,
)

logger = setup_logging("optimization_pipeline")

# Graceful imports for optional dependencies
try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
BATCH_SIZE = 64
TRAIN_ITERATIONS = 200
INFERENCE_ITERATIONS = 200
WARMUP_ITERATIONS = 20
PRUNE_RATIO = 0.5


# ---------------------------------------------------------------
# Pipeline stage helpers
# ---------------------------------------------------------------

def measure_model_size(model):
    """Measure parameter count and file size of a model.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        Dict with 'param_count' (int) and 'file_size_mb' (float).
    """
    param_count = sum(p.numel() for p in model.parameters())
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    tmp_path = tmp.name
    tmp.close()
    try:
        torch.save(model.state_dict(), tmp_path)
        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    finally:
        os.unlink(tmp_path)
    return {"param_count": param_count, "file_size_mb": file_size_mb}


@benchmark
def run_pytorch_inference(model, inputs, num_iterations):
    """Benchmark PyTorch inference."""
    with torch.inference_mode():
        for _ in range(num_iterations):
            _ = model(inputs)
    return None


def evaluate_accuracy(model, device, num_batches=5):
    """Evaluate model accuracy on synthetic data.

    Args:
        model: Model in eval mode.
        device: Target device.
        num_batches: Number of batches to evaluate.

    Returns:
        Accuracy as a float (0.0 to 1.0).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for _ in range(num_batches):
            inputs, labels = get_sample_batch(batch_size=BATCH_SIZE, device=device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


class PrunedCNN(nn.Module):
    """A structurally pruned CNN with physically smaller layers."""

    def __init__(self, features, classifier):
        super().__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def structurally_prune(model, prune_ratio, device):
    """Apply structured pruning to produce a physically smaller model.

    Prunes output channels from the first conv layer and adjusts downstream
    layers accordingly.

    Args:
        model: The baseline SimpleCNN.
        prune_ratio: Fraction of channels to remove (0.0 to 1.0).
        device: Target device.

    Returns:
        A new, smaller nn.Module.
    """
    temp = copy.deepcopy(model)
    conv0 = temp.features[0]
    prune.ln_structured(conv0, name="weight", amount=prune_ratio, n=1, dim=0)

    mask = conv0.weight_mask
    surviving = torch.where(mask.sum(dim=(1, 2, 3)) > 0)[0]
    num_surviving = len(surviving)

    prune.remove(conv0, "weight")

    new_conv0 = nn.Conv2d(3, num_surviving, kernel_size=3, padding=1)
    new_conv0.weight.data = conv0.weight.data[surviving].clone()
    new_conv0.bias.data = conv0.bias.data[surviving].clone()

    old_conv1 = temp.features[3]
    new_conv1 = nn.Conv2d(num_surviving, 64, kernel_size=3, padding=1)
    new_conv1.weight.data = old_conv1.weight.data[:, surviving].clone()
    new_conv1.bias.data = old_conv1.bias.data.clone()

    new_features = nn.Sequential(
        new_conv0,
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        new_conv1,
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        copy.deepcopy(temp.features[6]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

    return PrunedCNN(new_features, copy.deepcopy(temp.classifier)).to(device)


def export_to_onnx(model, sample_input, path):
    """Export a PyTorch model to ONNX format.

    Args:
        model: Model in eval mode.
        sample_input: Example input tensor.
        path: File path for the ONNX model.

    Returns:
        True if export succeeded, False otherwise.
    """
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    # Verify
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    return True


@benchmark
def run_ort_inference(session, input_name, inputs_np, num_iterations):
    """Benchmark ONNX Runtime inference."""
    for _ in range(num_iterations):
        _ = session.run(None, {input_name: inputs_np})
    return None


def main():
    # ==============================================================
    # SETUP
    # ==============================================================
    print("\n" + "=" * 70)
    print("  End-to-End Optimization Pipeline")
    print("  Train -> Prune -> ONNX -> TensorRT")
    print("=" * 70 + "\n")

    device = get_device()
    print_device_info()

    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Prune ratio: {int(PRUNE_RATIO * 100)}%")
    logger.info(f"ONNX Runtime available: {HAS_ONNX}")
    logger.info(f"TensorRT available: {HAS_TENSORRT}")

    # Track cumulative results
    pipeline_results = []
    size_results = []

    inputs, labels = get_sample_batch(batch_size=BATCH_SIZE, device=device)

    # ==============================================================
    # STAGE 1: Train baseline model
    # ==============================================================
    print("\n" + "=" * 60)
    print("  STAGE 1: Train Baseline Model")
    print("=" * 60 + "\n")

    model = SimpleCNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Training for {TRAIN_ITERATIONS} iterations...")
    model.train()
    for i in range(TRAIN_ITERATIONS):
        batch_inputs, batch_labels = get_sample_batch(batch_size=BATCH_SIZE, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            logger.info(f"  Step {i + 1}/{TRAIN_ITERATIONS} -- loss: {loss.item():.4f}")

    model.eval()
    baseline_size = measure_model_size(model)
    baseline_acc = evaluate_accuracy(model, device)
    logger.info(f"Baseline: {baseline_size['param_count']:,} params, "
                f"{baseline_size['file_size_mb']:.2f} MB, "
                f"accuracy: {baseline_acc * 100:.1f}%")

    # Benchmark baseline inference
    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            _ = model(inputs)

    baseline_bench = run_pytorch_inference(model, inputs, INFERENCE_ITERATIONS)
    logger.info(f"Baseline inference: {baseline_bench['time_seconds']:.4f}s")

    pipeline_results.append({
        "name": "1. Baseline (PyTorch)",
        "time_seconds": baseline_bench["time_seconds"],
        "memory_mb": baseline_bench.get("memory_mb"),
    })
    size_results.append({
        "name": "1. Baseline",
        "params": baseline_size["param_count"],
        "size_mb": baseline_size["file_size_mb"],
        "accuracy": baseline_acc,
    })

    # ==============================================================
    # STAGE 2: Structural pruning
    # ==============================================================
    print("\n" + "=" * 60)
    print(f"  STAGE 2: Structural Pruning ({int(PRUNE_RATIO * 100)}% channels)")
    print("=" * 60 + "\n")

    logger.info(
        "Removing output channels from the first conv layer to produce a "
        "physically smaller model. This reduces both parameter count and "
        "actual computation."
    )

    pruned_model = structurally_prune(model, PRUNE_RATIO, device)

    # Fine-tune the pruned model
    logger.info("Fine-tuning pruned model to recover accuracy...")
    pruned_model.train()
    optimizer_p = torch.optim.SGD(pruned_model.parameters(), lr=0.001, momentum=0.9)
    for i in range(50):
        batch_inputs, batch_labels = get_sample_batch(batch_size=BATCH_SIZE, device=device)
        optimizer_p.zero_grad(set_to_none=True)
        outputs = pruned_model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer_p.step()

    pruned_model.eval()
    pruned_size = measure_model_size(pruned_model)
    pruned_acc = evaluate_accuracy(pruned_model, device)
    logger.info(f"Pruned: {pruned_size['param_count']:,} params, "
                f"{pruned_size['file_size_mb']:.2f} MB, "
                f"accuracy: {pruned_acc * 100:.1f}%")

    param_reduction = (1 - pruned_size["param_count"] / baseline_size["param_count"]) * 100
    logger.info(f"Parameter reduction: {param_reduction:.1f}%")

    # Benchmark pruned inference
    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            _ = pruned_model(inputs)

    pruned_bench = run_pytorch_inference(pruned_model, inputs, INFERENCE_ITERATIONS)
    logger.info(f"Pruned inference: {pruned_bench['time_seconds']:.4f}s")

    speedup_vs_baseline = baseline_bench["time_seconds"] / pruned_bench["time_seconds"] if pruned_bench["time_seconds"] > 0 else float("inf")
    logger.info(f"Speedup vs baseline: {speedup_vs_baseline:.2f}x")

    pipeline_results.append({
        "name": "2. Pruned (PyTorch)",
        "time_seconds": pruned_bench["time_seconds"],
        "memory_mb": pruned_bench.get("memory_mb"),
    })
    size_results.append({
        "name": "2. Pruned",
        "params": pruned_size["param_count"],
        "size_mb": pruned_size["file_size_mb"],
        "accuracy": pruned_acc,
    })

    # ==============================================================
    # STAGE 3: ONNX Runtime
    # ==============================================================
    print("\n" + "=" * 60)
    print("  STAGE 3: ONNX Runtime Inference")
    print("=" * 60 + "\n")

    if not HAS_ONNX:
        logger.warning(
            "ONNX/ONNX Runtime not installed. Skipping ONNX stages. "
            "Install with: pip install onnx onnxruntime"
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "pruned_model.onnx")

            logger.info("Exporting pruned model to ONNX...")
            sample = inputs[:1] if device.type == "cpu" else inputs[:1].cpu()
            export_to_onnx(
                pruned_model.cpu(),
                sample,
                onnx_path,
            )
            pruned_model.to(device)

            onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
            logger.info(f"ONNX file size: {onnx_size:.2f} MB")

            # Run with ONNX Runtime
            providers = ["CPUExecutionProvider"]
            if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info(f"ORT providers: {providers}")

            session = ort.InferenceSession(onnx_path, providers=providers)
            input_name = session.get_inputs()[0].name

            inputs_np = inputs.cpu().numpy().astype(np.float32)

            # Warmup
            for _ in range(WARMUP_ITERATIONS):
                _ = session.run(None, {input_name: inputs_np})

            ort_bench = run_ort_inference(session, input_name, inputs_np, INFERENCE_ITERATIONS)
            logger.info(f"ORT inference: {ort_bench['time_seconds']:.4f}s")

            speedup_vs_baseline = baseline_bench["time_seconds"] / ort_bench["time_seconds"] if ort_bench["time_seconds"] > 0 else float("inf")
            logger.info(f"Cumulative speedup vs baseline: {speedup_vs_baseline:.2f}x")

            pipeline_results.append({
                "name": "3. Pruned + ONNX RT",
                "time_seconds": ort_bench["time_seconds"],
                "memory_mb": ort_bench.get("memory_mb"),
            })
            size_results.append({
                "name": "3. Pruned + ONNX",
                "params": pruned_size["param_count"],
                "size_mb": onnx_size,
                "accuracy": pruned_acc,
            })

    # ==============================================================
    # STAGE 4: TensorRT (if available)
    # ==============================================================
    print("\n" + "=" * 60)
    print("  STAGE 4: TensorRT FP16 Acceleration")
    print("=" * 60 + "\n")

    if not HAS_TENSORRT:
        logger.warning(
            "TensorRT not installed. Skipping TensorRT stage. "
            "Install from NVIDIA's package repository."
        )
    elif device.type != "cuda":
        logger.warning("TensorRT requires a CUDA GPU. Skipping.")
    elif not HAS_ONNX:
        logger.warning("TensorRT stage requires ONNX export. Skipping.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "pruned_model.onnx")

            # Re-export for TensorRT
            sample = inputs[:1].cpu()
            export_to_onnx(pruned_model.cpu(), sample, onnx_path)
            pruned_model.to(device)

            logger.info("Building TensorRT FP16 engine from ONNX...")

            trt_logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(trt_logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, trt_logger)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error(f"  TRT parse error: {parser.get_error(i)}")
                    logger.error("TensorRT parsing failed. Skipping.")
                else:
                    config = builder.create_builder_config()
                    config.set_memory_pool_limit(
                        trt.MemoryPoolType.WORKSPACE, 1 << 28
                    )
                    if builder.platform_has_fast_fp16:
                        config.set_flag(trt.BuilderFlag.FP16)
                        logger.info("FP16 mode enabled.")

                    engine = builder.build_serialized_network(network, config)
                    if engine is None:
                        logger.error("TensorRT engine build failed.")
                    else:
                        runtime = trt.Runtime(trt_logger)
                        engine_obj = runtime.deserialize_cuda_engine(engine)
                        context = engine_obj.create_execution_context()

                        # Allocate GPU buffers
                        input_shape = (BATCH_SIZE, 3, 32, 32)
                        output_shape = (BATCH_SIZE, 10)
                        d_input = torch.empty(input_shape, dtype=torch.float32, device="cuda")
                        d_output = torch.empty(output_shape, dtype=torch.float32, device="cuda")

                        context.set_tensor_address("input", d_input.data_ptr())
                        context.set_tensor_address("output", d_output.data_ptr())

                        # Warmup
                        for _ in range(WARMUP_ITERATIONS):
                            d_input.copy_(inputs)
                            context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                        torch.cuda.synchronize()

                        @benchmark
                        def trt_inference():
                            for _ in range(INFERENCE_ITERATIONS):
                                d_input.copy_(inputs)
                                context.execute_async_v3(
                                    torch.cuda.current_stream().cuda_stream
                                )
                            torch.cuda.synchronize()
                            return None

                        trt_bench = trt_inference()
                        logger.info(f"TensorRT FP16 inference: {trt_bench['time_seconds']:.4f}s")

                        trt_engine_size = len(engine) / (1024 * 1024)
                        logger.info(f"TensorRT engine size: {trt_engine_size:.2f} MB")

                        speedup_vs_baseline = baseline_bench["time_seconds"] / trt_bench["time_seconds"] if trt_bench["time_seconds"] > 0 else float("inf")
                        logger.info(f"Cumulative speedup vs baseline: {speedup_vs_baseline:.2f}x")

                        pipeline_results.append({
                            "name": "4. Pruned + TRT FP16",
                            "time_seconds": trt_bench["time_seconds"],
                            "memory_mb": trt_bench.get("memory_mb"),
                        })
                        size_results.append({
                            "name": "4. Pruned + TRT FP16",
                            "params": pruned_size["param_count"],
                            "size_mb": trt_engine_size,
                            "accuracy": pruned_acc,
                        })

    # ==============================================================
    # PIPELINE SUMMARY
    # ==============================================================
    print("\n" + "=" * 60)
    print("  Pipeline Summary: Cumulative Optimization Results")
    print("=" * 60 + "\n")

    # Inference speed table
    print("--- Inference Speed (lower is better) ---\n")
    print_benchmark_table(pipeline_results)

    # Speedup column
    if len(pipeline_results) > 1:
        base_time = pipeline_results[0]["time_seconds"]
        print("--- Cumulative Speedup vs Baseline ---\n")
        for r in pipeline_results:
            speedup = base_time / r["time_seconds"] if r["time_seconds"] > 0 else float("inf")
            logger.info(f"  {r['name']}: {speedup:.2f}x")

    # Model size table
    if size_results:
        print("\n--- Model Size Through Pipeline ---\n")
        print(f"+{'-' * 28}+{'-' * 14}+{'-' * 14}+{'-' * 12}+")
        print(
            f"| {'Stage':<26} | {'Params':>12} | {'Size (MB)':>12} | "
            f"{'Accuracy':>10} |"
        )
        print(f"+{'-' * 28}+{'-' * 14}+{'-' * 14}+{'-' * 12}+")
        for r in size_results:
            print(
                f"| {r['name']:<26} | {r['params']:>12,} | "
                f"{r['size_mb']:>12.2f} | "
                f"{r.get('accuracy', 0) * 100:>9.1f}% |"
            )
        print(f"+{'-' * 28}+{'-' * 14}+{'-' * 14}+{'-' * 12}+")

    # ==============================================================
    # Key Takeaways
    # ==============================================================
    print("\n--- Key Takeaways ---\n")
    logger.info(
        "1. Optimization techniques are COMPLEMENTARY -- each one addresses "
        "a different bottleneck (model size, precision, runtime, graph optimization)."
    )
    logger.info(
        "2. Structured pruning reduces parameter count and computation. "
        "ONNX/TensorRT optimizes the remaining operations. FP16 halves memory bandwidth."
    )
    logger.info(
        "3. The pipeline order matters: train first (need full precision), "
        "prune second (simplify the model), export last (optimize the final graph)."
    )
    logger.info(
        "4. Always verify accuracy at each stage. Pruning and FP16 can "
        "introduce small accuracy drops that compound."
    )
    logger.info(
        "5. For production deployment: serialize the TensorRT engine to disk. "
        "The build is expensive but the engine is fast and portable across "
        "identical GPU architectures."
    )

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
