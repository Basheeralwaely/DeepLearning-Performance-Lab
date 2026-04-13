"""
Gradient Checkpointing: Trading Compute for Memory
====================================================

During a standard backward pass, PyTorch stores ALL intermediate activations
from the forward pass so that gradients can be computed. For deep models or
large batch sizes, these activations consume enormous GPU memory.

Gradient checkpointing (also called activation checkpointing) solves this by:
  1. NOT storing intermediate activations during the forward pass
  2. RE-COMPUTING them on-the-fly during the backward pass

The tradeoff:
  - Memory: Drops from O(N) to O(sqrt(N)) for N layers
  - Compute: ~30-40% slower per step (recomputes forward pass segments)
  - Net effect: You can train with LARGER batches or BIGGER models on the
    same GPU, often more than compensating for the per-step slowdown.

When to use:
  - You're hitting OOM errors during training
  - You want to increase batch size without upgrading GPUs
  - You're training deep models (ResNets, Transformers) where activations
    dominate memory usage

What this tutorial demonstrates:
  1. Baseline training with full activation storage
  2. Training with gradient checkpointing enabled
  3. Memory savings measurement (the primary benefit)
  4. Throughput comparison (showing the compute cost)
  5. Maximum batch size comparison (the practical payoff)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from utils import (
    setup_logging,
    benchmark,
    compare_results,
    print_benchmark_table,
    get_sample_batch,
    get_device,
    print_device_info,
)

logger = setup_logging("gradient_checkpointing")

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
BATCH_SIZE = 64
NUM_ITERATIONS = 50
WARMUP_ITERATIONS = 5


# ---------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------

class DeepCNN(nn.Module):
    """A deeper CNN to make activation memory costs more visible.

    Has 8 convolutional blocks -- enough depth that stored activations
    become a significant fraction of total GPU memory usage.

    Args:
        input_channels: Number of input channels (default: 3).
        num_classes: Number of output classes (default: 10).
    """

    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        # Build 8 conv blocks: each is Conv2d -> BatchNorm -> ReLU
        channels = [input_channels, 64, 64, 128, 128, 256, 256, 512, 512]
        blocks = []
        for i in range(8):
            blocks.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=False),  # inplace=False required for checkpointing
            ))
            # Downsample every 2 blocks
            if i % 2 == 1:
                blocks.append(nn.MaxPool2d(2))

        self.features = nn.Sequential(*blocks)
        # After 4 rounds of MaxPool2d(2) on 32x32 input: 2x2 spatial
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeepCNNCheckpointed(nn.Module):
    """Same architecture as DeepCNN but with gradient checkpointing.

    Wraps groups of layers in torch.utils.checkpoint.checkpoint() so that
    intermediate activations are recomputed during backward instead of stored.

    IMPORTANT: Layers inside checkpointed segments must NOT use inplace
    operations (e.g., ReLU(inplace=True)) because checkpointing needs to
    re-run the forward pass, which fails if tensors were modified in-place.
    """

    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        channels = [input_channels, 64, 64, 128, 128, 256, 256, 512, 512]

        # Create checkpoint segments -- each segment is a group of layers
        # whose activations will be recomputed during backward
        self.segment1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.segment2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels[3], channels[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.segment3 = nn.Sequential(
            nn.Conv2d(channels[4], channels[5], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[5]),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels[5], channels[6], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[6]),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.segment4 = nn.Sequential(
            nn.Conv2d(channels[6], channels[7], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[7]),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels[7], channels[8], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[8]),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # Each segment's activations are recomputed during backward
        # use_reentrant=False is the recommended setting for new code
        x = checkpoint(self.segment1, x, use_reentrant=False)
        x = checkpoint(self.segment2, x, use_reentrant=False)
        x = checkpoint(self.segment3, x, use_reentrant=False)
        x = checkpoint(self.segment4, x, use_reentrant=False)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------

def train_loop(model, optimizer, criterion, device, num_iterations, batch_size):
    """Run a training loop for a fixed number of iterations.

    Args:
        model: Model to train.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Target device.
        num_iterations: Number of training steps.
        batch_size: Batch size for synthetic data.

    Returns:
        Final loss value (float).
    """
    model.train()
    loss_val = 0.0
    for _ in range(num_iterations):
        inputs, labels = get_sample_batch(batch_size=batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
    return loss_val


def find_max_batch_size(model_class, device, start=64, max_attempts=10):
    """Find the maximum batch size that fits in GPU memory.

    Binary-search style: doubles batch size until OOM, then returns
    the last successful size.

    Args:
        model_class: Class to instantiate (DeepCNN or DeepCNNCheckpointed).
        device: Target device.
        start: Starting batch size.
        max_attempts: Maximum doubling attempts.

    Returns:
        Maximum batch size that completed a training step without OOM.
    """
    if device.type != "cuda":
        return start  # Memory search only meaningful on GPU

    max_bs = start
    bs = start

    for _ in range(max_attempts):
        try:
            torch.cuda.empty_cache()
            model = model_class().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            inputs, labels = get_sample_batch(batch_size=bs, device=device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            max_bs = bs
            logger.info(f"  batch_size={bs}: OK")
            bs *= 2
            del model, optimizer, inputs, labels, outputs, loss
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info(f"  batch_size={bs}: OOM")
                del model, optimizer
                torch.cuda.empty_cache()
                break
            raise

    return max_bs


def main():
    # ==============================================================
    # SETUP
    # ==============================================================
    print("\n" + "=" * 60)
    print("  Gradient Checkpointing: Trading Compute for Memory")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    # Show model info
    model_info = DeepCNN()
    total_params = sum(p.numel() for p in model_info.parameters())
    logger.info(f"Model: DeepCNN (8 conv blocks) with {total_params:,} parameters")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Training iterations per benchmark: {NUM_ITERATIONS}")
    del model_info

    # ==============================================================
    # SECTION 1: Baseline Training (No Checkpointing)
    # ==============================================================
    print("\n--- Section 1: Baseline Training (Full Activation Storage) ---\n")
    logger.info(
        "Standard training stores ALL intermediate activations from the forward "
        "pass. For our 8-block CNN, that means 8 sets of feature maps are kept "
        "in memory simultaneously during backward."
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model_baseline = DeepCNN().to(device)
    optimizer_b = torch.optim.SGD(model_baseline.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Warmup
    logger.info("Running warmup passes...")
    train_loop(model_baseline, optimizer_b, criterion, device, WARMUP_ITERATIONS, BATCH_SIZE)

    # Reset for fair measurement
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    @benchmark
    def baseline_training():
        return train_loop(
            model_baseline, optimizer_b, criterion, device, NUM_ITERATIONS, BATCH_SIZE
        )

    baseline_result = baseline_training()
    baseline_per_iter = baseline_result["time_seconds"] / NUM_ITERATIONS
    logger.info(
        f"Baseline: {baseline_result['time_seconds']:.4f}s total, "
        f"{baseline_per_iter * 1000:.2f}ms per step"
    )
    if baseline_result.get("memory_mb") is not None:
        logger.info(f"Baseline peak memory: {baseline_result['memory_mb']:.1f} MB")

    # ==============================================================
    # SECTION 2: Checkpointed Training
    # ==============================================================
    print("\n--- Section 2: Checkpointed Training (Activation Recomputation) ---\n")
    logger.info(
        "With gradient checkpointing, only the inputs to each segment are "
        "stored. During backward, the segment's forward pass is re-run to "
        "recompute activations on demand. This trades ~30-40%% more compute "
        "for significantly less memory."
    )
    logger.info(
        "IMPORTANT: Layers inside checkpointed regions must NOT use "
        "inplace=True (e.g., ReLU(inplace=True)) because the forward pass "
        "is re-executed and the original input tensor must be unchanged."
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model_ckpt = DeepCNNCheckpointed().to(device)
    optimizer_c = torch.optim.SGD(model_ckpt.parameters(), lr=0.01, momentum=0.9)

    # Warmup
    logger.info("Running warmup passes...")
    train_loop(model_ckpt, optimizer_c, criterion, device, WARMUP_ITERATIONS, BATCH_SIZE)

    # Reset for fair measurement
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    @benchmark
    def checkpointed_training():
        return train_loop(
            model_ckpt, optimizer_c, criterion, device, NUM_ITERATIONS, BATCH_SIZE
        )

    ckpt_result = checkpointed_training()
    ckpt_per_iter = ckpt_result["time_seconds"] / NUM_ITERATIONS
    logger.info(
        f"Checkpointed: {ckpt_result['time_seconds']:.4f}s total, "
        f"{ckpt_per_iter * 1000:.2f}ms per step"
    )
    if ckpt_result.get("memory_mb") is not None:
        logger.info(f"Checkpointed peak memory: {ckpt_result['memory_mb']:.1f} MB")

    compare_results(baseline_result, ckpt_result, "Gradient Checkpointing")

    # ==============================================================
    # SECTION 3: Memory Savings Analysis
    # ==============================================================
    print("\n--- Section 3: Memory Savings Analysis ---\n")

    if baseline_result.get("memory_mb") and ckpt_result.get("memory_mb"):
        b_mem = baseline_result["memory_mb"]
        c_mem = ckpt_result["memory_mb"]
        savings = b_mem - c_mem
        pct = (savings / b_mem) * 100 if b_mem > 0 else 0

        logger.info(f"Baseline peak memory:      {b_mem:.1f} MB")
        logger.info(f"Checkpointed peak memory:  {c_mem:.1f} MB")
        logger.info(f"Memory saved:              {savings:.1f} MB ({pct:.1f}%)")
        logger.info(
            f"This means you could increase batch size by roughly "
            f"{b_mem / c_mem:.1f}x before hitting OOM."
        )
    else:
        logger.info(
            "Memory comparison requires CUDA. On CPU, gradient checkpointing "
            "still works but the memory benefit is less dramatic."
        )

    # ==============================================================
    # SECTION 4: Maximum Batch Size Comparison
    # ==============================================================
    print("\n--- Section 4: Maximum Batch Size Comparison ---\n")

    if device.type == "cuda":
        logger.info(
            "Finding the maximum batch size each approach can handle "
            "before running out of GPU memory..."
        )

        logger.info("\nBaseline (no checkpointing):")
        max_bs_baseline = find_max_batch_size(DeepCNN, device)
        logger.info(f"  Maximum batch size: {max_bs_baseline}")

        logger.info("\nWith checkpointing:")
        max_bs_ckpt = find_max_batch_size(DeepCNNCheckpointed, device)
        logger.info(f"  Maximum batch size: {max_bs_ckpt}")

        if max_bs_ckpt > max_bs_baseline:
            logger.info(
                f"\nCheckpointing allows {max_bs_ckpt / max_bs_baseline:.1f}x "
                f"larger batch size on the same GPU!"
            )
    else:
        logger.info("Skipping max batch size search (requires CUDA GPU).")

    # ==============================================================
    # SECTION 5: Benchmark Summary
    # ==============================================================
    print("\n--- Benchmark Summary ---\n")

    results = [
        {
            "name": "Baseline (no ckpt)",
            "time_seconds": baseline_result["time_seconds"],
            "memory_mb": baseline_result.get("memory_mb"),
        },
        {
            "name": "Gradient checkpointing",
            "time_seconds": ckpt_result["time_seconds"],
            "memory_mb": ckpt_result.get("memory_mb"),
        },
    ]
    print_benchmark_table(results)

    # ==============================================================
    # Key Takeaways
    # ==============================================================
    print("\n--- Key Takeaways ---\n")
    logger.info(
        "1. Gradient checkpointing reduces activation memory from O(N) to "
        "O(sqrt(N)) -- the deeper the model, the bigger the savings."
    )
    logger.info(
        "2. The compute cost is ~30-40%% slower per step due to recomputing "
        "activations during backward. This is usually worth it because larger "
        "batch sizes improve GPU utilization."
    )
    logger.info(
        "3. NEVER use inplace operations (ReLU(inplace=True)) inside "
        "checkpointed segments -- the recomputation needs the original tensors."
    )
    logger.info(
        "4. use_reentrant=False is recommended for new code -- it handles "
        "edge cases better and will become the default in future PyTorch."
    )
    logger.info(
        "5. Combine with torch.compile and AMP for maximum training efficiency: "
        "checkpointing frees memory, AMP halves precision, compile fuses ops."
    )

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
