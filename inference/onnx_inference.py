"""
ONNX Runtime Inference Optimization
=====================================

ONNX (Open Neural Network Exchange) is an open format for representing machine
learning models. ONNX Runtime (ORT) is Microsoft's high-performance inference
engine that applies graph-level optimizations to ONNX models, including:
  - Constant folding (pre-compute static subgraphs)
  - Redundant node elimination (remove identity ops)
  - Operator fusion (combine Conv+BN+ReLU into single kernel)
  - Memory planning (optimize tensor allocation)

ORT supports multiple execution providers (CPU, CUDA, TensorRT, etc.) and
allows fine-grained control over optimization levels.

What this tutorial demonstrates:
  1. PyTorch baseline inference on CPU (fair comparison baseline)
  2. ONNX model export with opset 17 and verification
  3. ORT execution provider detection and selection
  4. Graph optimization level comparison (Disabled, Basic, Extended, All)
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import (
    setup_logging,
    benchmark,
    print_benchmark_table,
    SimpleCNN,
    get_sample_batch,
    get_device,
    print_device_info,
)

logger = setup_logging("onnx_inference")

# Graceful ONNX Runtime import
try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False

# Graceful ONNX import
try:
    import onnx

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_ITERATIONS = 100
BATCH_SIZE = 64
WARMUP_ITERATIONS = 10


def main():
    # ==================================================================
    # Setup
    # ==================================================================
    print("\n" + "=" * 60)
    print("  ONNX Runtime Inference Optimization")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Benchmark iterations: {NUM_ITERATIONS}")
    logger.info(f"Warmup iterations: {WARMUP_ITERATIONS}")
    logger.info(f"ONNX Runtime available: {HAS_ORT}")
    logger.info(f"ONNX available: {HAS_ONNX}")

    has_cuda = torch.cuda.is_available()

    # ==================================================================
    # Section 1: PyTorch Baseline Inference (CPU)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Section 1: PyTorch Baseline Inference (CPU)")
    print("=" * 60 + "\n")

    logger.info("Creating model on CPU for fair comparison with ORT CPU provider")
    model_cpu = SimpleCNN().eval()
    total_params = sum(p.numel() for p in model_cpu.parameters())
    logger.info(f"Model: SimpleCNN with {total_params:,} parameters")

    # CPU input for fair comparison
    cpu_input = torch.randn(BATCH_SIZE, 3, 32, 32)
    logger.info(f"Input shape: {tuple(cpu_input.shape)} (CPU)")

    # Warmup on CPU
    logger.info("Running warmup passes...")
    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            _ = model_cpu(cpu_input)
    logger.info("CPU warmup complete.")

    @benchmark
    def pytorch_cpu_inference(model, inputs, num_iters):
        with torch.inference_mode():
            for _ in range(num_iters):
                output = model(inputs)
        return output

    cpu_baseline = pytorch_cpu_inference(model_cpu, cpu_input, NUM_ITERATIONS)
    logger.info(f"PyTorch CPU baseline: {cpu_baseline['time_seconds']:.4f}s "
                f"({NUM_ITERATIONS} iterations)")

    # Optionally benchmark on GPU for reference
    gpu_baseline = None
    if has_cuda:
        logger.info("CUDA available -- also running GPU baseline for reference")
        model_gpu = SimpleCNN().eval().to("cuda")
        gpu_input = cpu_input.to("cuda")

        # GPU warmup
        with torch.inference_mode():
            for _ in range(WARMUP_ITERATIONS):
                _ = model_gpu(gpu_input)
        torch.cuda.synchronize()

        @benchmark
        def pytorch_gpu_inference(model, inputs, num_iters):
            with torch.inference_mode():
                for _ in range(num_iters):
                    output = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return output

        gpu_baseline = pytorch_gpu_inference(model_gpu, gpu_input, NUM_ITERATIONS)
        logger.info(f"PyTorch GPU baseline: {gpu_baseline['time_seconds']:.4f}s "
                    f"({NUM_ITERATIONS} iterations)")
    else:
        logger.info("No CUDA GPU available -- GPU baseline skipped")

    # ==================================================================
    # Section 2: ONNX Model Export
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Section 2: ONNX Model Export")
    print("=" * 60 + "\n")

    if not HAS_ONNX:
        logger.warning("ONNX not installed. pip install onnx")
        logger.warning("Cannot proceed with ONNX export.")
        return

    # Create temp file for ONNX model
    onnx_tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx_path = onnx_tmp.name
    onnx_tmp.close()

    try:
        dummy_input = torch.randn(1, 3, 32, 32)
        logger.info(f"Exporting model to ONNX (opset 17) at: {onnx_path}")

        torch.onnx.export(
            model_cpu,
            dummy_input,
            onnx_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )

        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        logger.info(f"ONNX file size: {file_size_mb:.2f} MB")

        # Verify the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed (checker.check_model)")

        num_nodes = len(onnx_model.graph.node)
        opset = onnx_model.opset_import[0].version if onnx_model.opset_import else "unknown"
        logger.info(f"ONNX graph: {num_nodes} nodes, opset version {opset}")

        # List model inputs/outputs
        for inp in onnx_model.graph.input:
            logger.info(f"  Input: '{inp.name}' "
                        f"shape={[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
        for out in onnx_model.graph.output:
            logger.info(f"  Output: '{out.name}' "
                        f"shape={[d.dim_value for d in out.type.tensor_type.shape.dim]}")

        # ==============================================================
        # Section 3: ONNX Runtime Provider Detection
        # ==============================================================
        print("\n" + "=" * 60)
        print("  Section 3: ONNX Runtime Provider Detection")
        print("=" * 60 + "\n")

        if not HAS_ORT:
            logger.warning("onnxruntime not installed. pip install onnxruntime")
            logger.warning("Skipping ORT inference benchmark.")
            return

        available_providers = ort.get_available_providers()
        logger.info(f"Available ORT providers: {available_providers}")

        has_cuda_provider = "CUDAExecutionProvider" in available_providers
        has_trt_provider = "TensorrtExecutionProvider" in available_providers

        if has_cuda_provider:
            logger.info("CUDAExecutionProvider detected -- GPU acceleration available")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            logger.info("CUDAExecutionProvider not available")
            if has_trt_provider:
                logger.info("TensorrtExecutionProvider detected")
            providers = ["CPUExecutionProvider"]

        logger.info(f"Selected providers: {providers}")

        # Create a test session to verify provider activation
        test_session = ort.InferenceSession(onnx_path, providers=providers)
        active_providers = test_session.get_providers()
        logger.info(f"Active providers after session creation: {active_providers}")

        if not has_cuda_provider:
            logger.info("Note: Benchmarking ORT CPU vs PyTorch CPU for fair comparison")
            logger.info("Install onnxruntime-gpu for CUDA provider support")

        # ==============================================================
        # Section 4: Graph Optimization Level Comparison
        # ==============================================================
        print("\n" + "=" * 60)
        print("  Section 4: Graph Optimization Level Comparison")
        print("=" * 60 + "\n")

        logger.info("Comparing all four ORT graph optimization levels:")
        logger.info("  - DISABLED: No graph optimizations applied")
        logger.info("  - BASIC:    Constant folding, redundant node elimination")
        logger.info("  - EXTENDED: Complex node fusions (e.g., Conv+BN+ReLU)")
        logger.info("  - ALL:      All optimizations including layout transforms")

        levels = [
            ("ORT Disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
            ("ORT Basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
            ("ORT Extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
            ("ORT All", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
        ]

        # Prepare numpy input for ORT (ORT expects numpy arrays on CPU)
        input_data = np.random.randn(BATCH_SIZE, 3, 32, 32).astype(np.float32)
        input_name = test_session.get_inputs()[0].name
        logger.info(f"ORT input name: '{input_name}'")
        logger.info(f"ORT input shape: {input_data.shape}, dtype: {input_data.dtype}")

        @benchmark
        def ort_inference(session, input_name, input_data, num_iters):
            for _ in range(num_iters):
                output = session.run(None, {input_name: input_data})
            return output

        # Collect results for all optimization levels
        ort_results = []
        for level_name, opt_level in levels:
            logger.info(f"\nBenchmarking: {level_name} ({opt_level})")

            opts = ort.SessionOptions()
            opts.graph_optimization_level = opt_level

            session = ort.InferenceSession(
                onnx_path, sess_options=opts, providers=providers
            )

            # Warmup
            for _ in range(WARMUP_ITERATIONS):
                session.run(None, {input_name: input_data})

            # Benchmark
            result = ort_inference(session, input_name, input_data, NUM_ITERATIONS)
            logger.info(f"  {level_name}: {result['time_seconds']:.4f}s")

            ort_results.append({
                "name": level_name,
                "time_seconds": result["time_seconds"],
                "memory_mb": result.get("memory_mb"),
            })

        # Build comprehensive results table
        print("\n--- Benchmark Results ---\n")

        all_results = [
            {
                "name": "PyTorch CPU",
                "time_seconds": cpu_baseline["time_seconds"],
                "memory_mb": cpu_baseline.get("memory_mb"),
            },
        ]
        all_results.extend(ort_results)

        if gpu_baseline is not None:
            all_results.append({
                "name": "PyTorch GPU (reference)",
                "time_seconds": gpu_baseline["time_seconds"],
                "memory_mb": gpu_baseline.get("memory_mb"),
            })

        print_benchmark_table(all_results)

        # Find fastest ORT level
        fastest_ort = min(ort_results, key=lambda r: r["time_seconds"])
        cpu_time = cpu_baseline["time_seconds"]
        ort_time = fastest_ort["time_seconds"]
        speedup = cpu_time / ort_time if ort_time > 0 else float("inf")

        logger.info(f"Fastest ORT level: {fastest_ort['name']} ({ort_time:.4f}s)")
        logger.info(f"Speedup vs PyTorch CPU: {speedup:.2f}x")

        if gpu_baseline is not None:
            gpu_speedup = gpu_baseline["time_seconds"] / ort_time if ort_time > 0 else float("inf")
            logger.info(f"ORT CPU vs PyTorch GPU: {gpu_speedup:.2f}x "
                        f"({'faster' if gpu_speedup > 1 else 'slower'})")

        # Key takeaways
        print("\n--- Key Takeaways ---\n")
        logger.info("ONNX Runtime graph optimizations can significantly speed up CPU inference")
        logger.info("ORT_ENABLE_ALL applies the most aggressive optimizations")
        logger.info("For production: export once, optimize once, run many times")
        logger.info("For GPU acceleration: pip install onnxruntime-gpu")
        logger.info("ORT supports quantized models for further speedup (INT8/UINT8)")
        logger.info("See tensorrt_inference.py for GPU-native TensorRT acceleration")

    finally:
        # Cleanup temp ONNX file
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)
            logger.info(f"Cleaned up temp ONNX file: {onnx_path}")

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
