"""
FP8 Training with NVIDIA Transformer Engine
=============================================

FP8 (8-bit floating point) training uses NVIDIA's Transformer Engine library
to run transformer forward passes in 8-bit precision, providing significant
speedup on Hopper (H100) and Ada Lovelace (RTX 4090) GPUs. FP8 uses per-tensor
dynamic scaling (DelayedScaling recipe) to maintain training quality.

Hardware requirements:
  - FP8:  SM 8.9+ (Ada Lovelace, Hopper, Blackwell)
  - BF16: SM 8.0+ (Ampere) -- used as fallback
  - FP16: SM 7.0+ (Volta, Turing) -- used as fallback

This tutorial adapts to your GPU automatically, running the highest-precision
format your hardware supports and explaining what FP8 would do on supported hardware.

What this tutorial demonstrates:
  1. GPU capability detection and precision tier selection
  2. SimpleViT model training as baseline
  3. FP8/BF16/FP16 optimized training (based on hardware)
  4. Benchmark comparison with throughput and memory metrics
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from utils import (
    setup_logging,
    benchmark,
    compare_results,
    print_benchmark_table,
    SimpleViT,
    get_sample_batch,
    get_device,
    print_device_info,
    get_gpu_capability,
)

logger = setup_logging("fp8_transformer_engine")

NUM_ITERATIONS = 50  # Fewer iterations -- ViT is heavier than CNN
BATCH_SIZE = 32      # Smaller batch -- ViT uses more memory
WARMUP_ITERATIONS = 10
IMAGE_SIZE = 32
PATCH_SIZE = 4


def get_gpu_tier() -> tuple[str, str]:
    """Detect GPU capability tier for mixed precision support.

    Returns:
        Tuple of (tier, description) where tier is 'fp8', 'bf16', or 'fp16'.
    """
    capability = get_gpu_capability()
    if capability is None:
        return "fp16", "No CUDA GPU -- CPU mode with FP16 emulation"
    major, minor = capability
    gpu_name = torch.cuda.get_device_name(0)
    if major > 8 or (major == 8 and minor >= 9):
        return "fp8", f"{gpu_name} (SM {major}.{minor}) -- FP8 supported"
    elif major >= 8:
        return "bf16", f"{gpu_name} (SM {major}.{minor}) -- BF16 native, FP8 not supported"
    else:
        return "fp16", f"{gpu_name} (SM {major}.{minor}) -- FP16 native, BF16 emulated"


def try_import_transformer_engine():
    """Try to import Transformer Engine. Returns module or None.

    Returns:
        Tuple of (te_module, Format, DelayedScaling) or (None, None, None).
    """
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import Format, DelayedScaling
        logger.info("Transformer Engine imported successfully")
        return te, Format, DelayedScaling
    except ImportError:
        logger.warning("Transformer Engine not installed -- FP8 code will be shown but not executed")
        logger.info("Install with: pip install --no-build-isolation transformer_engine[pytorch]")
        return None, None, None


def main():
    # ================================================================
    # Section 0: Setup + GPU Detection
    # ================================================================
    print("\n" + "=" * 60)
    print("  FP8 Training with Transformer Engine")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    tier, tier_desc = get_gpu_tier()
    print(f"\n  GPU Tier: {tier_desc}")
    print(f"  Running: {tier.upper()} precision path\n")

    if device.type != "cuda":
        logger.warning("Mixed precision requires a CUDA GPU for meaningful benchmarks.")
        logger.warning("Results will still run but performance differences will be minimal.")

    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Training iterations per benchmark: {NUM_ITERATIONS}")
    logger.info(f"Warmup iterations: {WARMUP_ITERATIONS}")

    # ================================================================
    # Section 1: SimpleViT Model Setup
    # ================================================================
    print("\n--- Section 1: SimpleViT Model ---\n")

    model = SimpleViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2

    logger.info(f"Model: SimpleViT with {total_params:,} parameters")
    logger.info(f"Architecture: dim=256, depth=4, heads=8, mlp_dim=512")
    logger.info(f"Input: {IMAGE_SIZE}x{IMAGE_SIZE} images, patch_size={PATCH_SIZE}")
    logger.info(f"Sequence: {num_patches} patches + 1 CLS token = {num_patches + 1} tokens")

    inputs, labels = get_sample_batch(
        batch_size=BATCH_SIZE, height=IMAGE_SIZE, width=IMAGE_SIZE, device=device
    )
    criterion = nn.CrossEntropyLoss()

    # ================================================================
    # Section 2: Warmup
    # ================================================================
    print("\n--- Warmup ---\n")

    logger.info("Running warmup passes to compile CUDA kernels...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(WARMUP_ITERATIONS):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    logger.info("Warmup complete. CUDA kernels compiled and memory pools initialized.")

    # ================================================================
    # Section 3: FP32 Baseline Training
    # ================================================================
    print("\n--- Section 2: FP32 Baseline Training ---\n")

    logger.info("All operations run in float32 -- full precision reference point.")

    model = SimpleViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    @benchmark
    def fp32_baseline():
        for _ in range(NUM_ITERATIONS):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    baseline = fp32_baseline()
    logger.info(f"FP32 baseline completed in {baseline['time_seconds']:.4f}s")
    baseline_throughput = NUM_ITERATIONS * BATCH_SIZE / baseline["time_seconds"]
    logger.info(f"FP32 throughput: {baseline_throughput:.0f} samples/sec")

    # ================================================================
    # Section 4: Optimized Training (tier-adaptive)
    # ================================================================
    optimized = None

    if tier == "fp8":
        print("\n--- Section 3: FP8 Training with Transformer Engine ---\n")
        te, Format, DelayedScaling = try_import_transformer_engine()

        if te is not None:
            logger.info("Building Transformer Engine model for FP8 training...")
            logger.info("FP8 uses per-tensor dynamic scaling via DelayedScaling recipe")
            logger.info("fp8_autocast wraps ONLY the forward pass -- backward must be outside")
            logger.info("Format.HYBRID uses E4M3 for forward, E5M2 for backward")

            # Build a TE-based transformer model
            # TE TransformerLayer replaces standard nn.TransformerEncoderLayer
            class TEViT(nn.Module):
                """SimpleViT using Transformer Engine layers for FP8 support."""

                def __init__(self):
                    super().__init__()
                    self.patch_embed = nn.Conv2d(3, 256, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
                    self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
                    self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, 256))
                    self.te_layers = nn.ModuleList([
                        te.TransformerLayer(
                            hidden_size=256,
                            ffn_hidden_size=512,
                            num_attention_heads=8,
                            layer_type="encoder",
                        )
                        for _ in range(4)
                    ])
                    self.norm = nn.LayerNorm(256)
                    self.head = nn.Linear(256, 10)

                def forward(self, x):
                    x = self.patch_embed(x).flatten(2).transpose(1, 2)
                    batch_size = x.size(0)
                    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
                    x = torch.cat([cls_tokens, x], dim=1)
                    x = x + self.pos_embed
                    for layer in self.te_layers:
                        x = layer(x)
                    cls_output = x[:, 0]
                    cls_output = self.norm(cls_output)
                    return self.head(cls_output)

            te_model = TEViT().to(device)
            te_optimizer = torch.optim.AdamW(te_model.parameters(), lr=1e-4)
            fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=16,
                amax_compute_algo="max",
            )

            @benchmark
            def fp8_training():
                for _ in range(NUM_ITERATIONS):
                    te_optimizer.zero_grad()
                    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                        outputs = te_model(inputs)
                    loss = criterion(outputs, labels)  # Loss outside fp8_autocast
                    loss.backward()  # Backward OUTSIDE fp8_autocast
                    te_optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()

            optimized = fp8_training()
            logger.info(f"FP8 training completed in {optimized['time_seconds']:.4f}s")
        else:
            # TE not installed -- show what would happen and fall back
            logger.info("=== FP8 Code (for reference -- requires Hopper+ GPU + Transformer Engine) ===")
            logger.info("  fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo='max')")
            logger.info("  with te.fp8_autocast(enabled=True, fp8_recipe=recipe):")
            logger.info("      output = te_model(input_tensor)")
            logger.info("  loss.backward()  # OUTSIDE fp8_autocast")
            logger.info("Falling back to BF16 for benchmark...")
            tier = "bf16"  # Fall through to BF16

    if tier == "bf16":
        print("\n--- Section 3: BF16 Training (Ampere Fallback) ---\n")

        logger.info("Your GPU supports BF16 natively but not FP8")
        logger.info("On Hopper/Ada GPUs, this section would run FP8 instead")
        logger.info("BF16 autocast: same range as FP32, no GradScaler needed")

        model = SimpleViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        @benchmark
        def bf16_training():
            for _ in range(NUM_ITERATIONS):
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()

        optimized = bf16_training()
        logger.info(f"BF16 training completed in {optimized['time_seconds']:.4f}s")

    elif tier == "fp16":
        print("\n--- Section 3: FP16 Training (Turing/Volta Fallback) ---\n")

        logger.info("Your GPU supports FP16 natively but not BF16 or FP8")
        logger.info("On Hopper/Ada GPUs, this section would run FP8 instead")
        logger.info("On Ampere GPUs, this section would run BF16 instead")
        logger.info("FP16 autocast with GradScaler for gradient underflow protection")

        model = SimpleViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler('cuda')

        @benchmark
        def fp16_training():
            for _ in range(NUM_ITERATIONS):
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()

        optimized = fp16_training()
        logger.info(f"FP16 + GradScaler training completed in {optimized['time_seconds']:.4f}s")

    # ================================================================
    # Section 5: Results
    # ================================================================
    print("\n--- Results ---\n")

    if optimized is not None:
        technique_label = tier.upper() + " Mixed Precision"
        compare_results(baseline, optimized, technique_label)

        opt_throughput = NUM_ITERATIONS * BATCH_SIZE / optimized["time_seconds"]
        logger.info(f"FP32 throughput:            {baseline_throughput:.0f} samples/sec")
        logger.info(f"{tier.upper()} throughput:            {opt_throughput:.0f} samples/sec")

        if optimized["time_seconds"] > 0:
            speedup = baseline["time_seconds"] / optimized["time_seconds"]
            logger.info(f"Speedup: {speedup:.2f}x")
    else:
        logger.warning("No optimized run completed -- cannot compare results")

    # ================================================================
    # Section 6: Key Takeaways
    # ================================================================
    print("\n--- Key Takeaways ---\n")

    logger.info("Mixed precision reduces memory and increases throughput by using lower-precision arithmetic")

    if tier == "fp8":
        logger.info("FP8 provides ~2x speedup over FP16/BF16 on Hopper GPUs for transformer workloads")
        logger.info("DelayedScaling recipe tracks per-tensor scaling factors across iterations")
        logger.info("Format.HYBRID: E4M3 (4-bit exponent, 3-bit mantissa) for forward, E5M2 for backward")
    elif tier == "bf16":
        logger.info("BF16 is the recommended fallback -- simpler than FP16 (no GradScaler)")
        logger.info("On Hopper/Ada GPUs, FP8 would provide additional ~2x speedup")
    else:
        logger.info("FP16 + GradScaler is the standard for pre-Ampere GPUs")
        logger.info("On Ampere GPUs, BF16 would be simpler (no GradScaler needed)")
        logger.info("On Hopper/Ada GPUs, FP8 would provide the best performance")

    logger.info("See amp_training.py for AMP basics and bf16_vs_fp16.py for BF16/FP16 deep dive")

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
