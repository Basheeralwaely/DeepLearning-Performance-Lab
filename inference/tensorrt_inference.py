"""
TensorRT Inference Acceleration
================================

TensorRT is NVIDIA's high-performance deep learning inference optimizer and
runtime. It takes a trained model (via ONNX intermediate representation) and
applies layer fusion, kernel auto-tuning, precision calibration, and memory
optimization to produce a highly optimized inference engine.

The pipeline demonstrated here:
  PyTorch model -> ONNX export -> TensorRT engine (FP16) -> Benchmark

FP16 (half-precision) mode halves the memory footprint and doubles throughput
on GPUs with Tensor Cores (Volta and newer), with minimal accuracy loss for
most models.

What this tutorial demonstrates:
  1. PyTorch baseline inference timing
  2. ONNX model export with opset 17
  3. TensorRT FP16 engine build from ONNX
  4. TensorRT inference benchmark vs PyTorch baseline
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import (
    setup_logging,
    benchmark,
    compare_results,
    print_benchmark_table,
    SimpleCNN,
    get_sample_batch,
    get_device,
    print_device_info,
)

logger = setup_logging("tensorrt_inference")

# Graceful TensorRT import -- not all environments have TRT installed
try:
    import tensorrt as trt

    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

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
INPUT_SHAPE = (BATCH_SIZE, 3, 32, 32)


def main():
    # ==================================================================
    # Section 1: PyTorch Baseline Inference
    # ==================================================================
    print("\n" + "=" * 60)
    print("  TensorRT Inference Acceleration")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Benchmark iterations: {NUM_ITERATIONS}")
    logger.info(f"Warmup iterations: {WARMUP_ITERATIONS}")
    logger.info(f"TensorRT available: {HAS_TENSORRT}")
    logger.info(f"ONNX available: {HAS_ONNX}")

    # Prepare model and data
    model = SimpleCNN().eval().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: SimpleCNN with {total_params:,} parameters")

    inputs, _ = get_sample_batch(batch_size=BATCH_SIZE, device=device)
    logger.info(f"Input shape: {tuple(inputs.shape)}")

    # Warmup -- compile CUDA kernels before timing
    print("\n--- Section 1: PyTorch Baseline Inference ---\n")
    logger.info("Running warmup passes to compile CUDA kernels...")
    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            _ = model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    logger.info("Warmup complete.")

    @benchmark
    def pytorch_inference(model, inputs, num_iters):
        with torch.inference_mode():
            for _ in range(num_iters):
                output = model(inputs)
        return output

    baseline = pytorch_inference(model, inputs, NUM_ITERATIONS)
    logger.info(f"PyTorch baseline: {baseline['time_seconds']:.4f}s "
                f"({NUM_ITERATIONS} iterations)")

    # Store PyTorch output for numerical verification later
    with torch.inference_mode():
        pytorch_out = model(inputs)

    # ==================================================================
    # Section 2: ONNX Export
    # ==================================================================
    print("\n" + "=" * 60)
    print("  Section 2: ONNX Export")
    print("=" * 60 + "\n")

    if not HAS_ONNX:
        logger.warning("ONNX not installed. pip install onnx")
        logger.warning("Cannot proceed with ONNX export or TensorRT conversion.")
        return

    # Use a temp file for the ONNX model -- cleaned up in finally block
    onnx_tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx_path = onnx_tmp.name
    onnx_tmp.close()

    try:
        # Export to ONNX -- single-batch dummy input for shape inference
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
        logger.info(f"Exporting model to ONNX (opset 17) at: {onnx_path}")

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,
        )

        file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        logger.info(f"ONNX file size: {file_size_mb:.2f} MB")

        # Verify the exported model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed (checker.check_model)")

        num_nodes = len(onnx_model.graph.node)
        logger.info(f"ONNX graph contains {num_nodes} nodes")

        # ==============================================================
        # Section 3: TensorRT Engine Build (FP16)
        # ==============================================================
        print("\n" + "=" * 60)
        print("  Section 3: TensorRT Engine Build (FP16)")
        print("=" * 60 + "\n")

        if not HAS_TENSORRT:
            logger.warning("TensorRT not installed. pip install tensorrt")
            logger.warning("Skipping TensorRT engine build and benchmark.")
            logger.info("PyTorch baseline results are still valid above.")

            # Print benchmark table with only PyTorch result
            results = [
                {
                    "name": "PyTorch (FP32)",
                    "time_seconds": baseline["time_seconds"],
                    "memory_mb": baseline.get("memory_mb"),
                },
            ]
            print_benchmark_table(results)
            return

        if not torch.cuda.is_available():
            logger.warning("TensorRT requires a CUDA GPU. No CUDA device found.")
            logger.warning("Skipping TensorRT engine build and benchmark.")
            return

        # Build TensorRT engine from ONNX using TRT 10.x API
        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        logger.info("Parsing ONNX model into TensorRT network...")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"TRT Parser Error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model with TensorRT")

        logger.info(f"TensorRT network: {network.num_layers} layers, "
                     f"{network.num_inputs} inputs, {network.num_outputs} outputs")

        # Configure FP16 mode
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 mode enabled -- Tensor Cores will be used where possible")
        logger.info("Building TensorRT FP16 engine... (this may take a moment)")

        # Build serialized engine
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            logger.error("TensorRT engine build failed (returned None)")
            return

        # Deserialize into runtime engine
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        logger.info(f"TensorRT engine built successfully")
        logger.info(f"Engine has {engine.num_io_tensors} IO tensors")

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            mode = engine.get_tensor_mode(name)
            logger.info(f"  Tensor '{name}': shape={shape}, dtype={dtype}, mode={mode}")

        # ==============================================================
        # Section 4: TensorRT Inference Benchmark
        # ==============================================================
        print("\n" + "=" * 60)
        print("  Section 4: TensorRT Inference Benchmark")
        print("=" * 60 + "\n")

        # Create execution context
        context = engine.create_execution_context()

        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)

        # Set input shape for the batch
        context.set_input_shape(input_name, (BATCH_SIZE, 3, 32, 32))
        logger.info(f"Input tensor: '{input_name}' shape={(BATCH_SIZE, 3, 32, 32)}")

        # Allocate output buffer on GPU
        output_tensor = torch.empty(BATCH_SIZE, 10, device="cuda")
        logger.info(f"Output tensor: '{output_name}' shape={(BATCH_SIZE, 10)}")

        # Create CUDA stream for async execution
        stream = torch.cuda.Stream()

        # Use contiguous input on CUDA
        trt_input = inputs.contiguous().cuda()

        @benchmark
        def trt_inference(context, input_tensor, output_tensor, input_name,
                          output_name, stream, num_iters):
            for _ in range(num_iters):
                context.set_tensor_address(input_name, input_tensor.data_ptr())
                context.set_tensor_address(output_name, output_tensor.data_ptr())
                context.execute_async_v3(stream.cuda_stream)
                stream.synchronize()
            return output_tensor.clone()

        # Warmup TRT
        logger.info(f"Running {WARMUP_ITERATIONS} TensorRT warmup iterations...")
        for _ in range(WARMUP_ITERATIONS):
            context.set_tensor_address(input_name, trt_input.data_ptr())
            context.set_tensor_address(output_name, output_tensor.data_ptr())
            context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()

        # Benchmark TRT
        trt_result = trt_inference(
            context, trt_input, output_tensor, input_name,
            output_name, stream, NUM_ITERATIONS
        )
        logger.info(f"TensorRT FP16: {trt_result['time_seconds']:.4f}s "
                     f"({NUM_ITERATIONS} iterations)")

        # Compare results
        compare_results(baseline, trt_result, "TensorRT FP16 vs PyTorch")

        # Build comprehensive benchmark table
        results = [
            {
                "name": "PyTorch (FP32)",
                "time_seconds": baseline["time_seconds"],
                "memory_mb": baseline.get("memory_mb"),
            },
            {
                "name": "TensorRT (FP16)",
                "time_seconds": trt_result["time_seconds"],
                "memory_mb": trt_result.get("memory_mb"),
            },
        ]
        print_benchmark_table(results)

        # Numerical verification -- check TRT output matches PyTorch
        print("\n--- Numerical Verification ---\n")
        trt_out = output_tensor
        pytorch_out_cuda = pytorch_out.cuda()
        match = torch.allclose(pytorch_out_cuda, trt_out, atol=1e-2)
        max_diff = (pytorch_out_cuda - trt_out).abs().max().item()
        logger.info(f"PyTorch vs TensorRT output match (atol=1e-2): {match}")
        logger.info(f"Maximum absolute difference: {max_diff:.6f}")
        if match:
            logger.info("Outputs are numerically consistent -- FP16 precision is acceptable")
        else:
            logger.warning("Outputs differ beyond tolerance -- inspect model for numerical sensitivity")

        # Key takeaways
        print("\n--- Key Takeaways ---\n")
        speedup = baseline["time_seconds"] / trt_result["time_seconds"] if trt_result["time_seconds"] > 0 else float("inf")
        logger.info(f"TensorRT FP16 speedup over PyTorch: {speedup:.2f}x")
        logger.info("TensorRT optimizes via: layer fusion, kernel auto-tuning, FP16 Tensor Cores")
        logger.info("Engine build is one-time cost -- serialize to disk for production use")
        logger.info("FP16 mode typically provides 2-4x speedup with <1% accuracy loss")
        logger.info("For INT8 quantization (further speedup), calibration data is required")

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
