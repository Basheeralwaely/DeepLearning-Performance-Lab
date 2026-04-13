"""
Model Parallelism: Pipeline-Style Layer Splitting
===================================================

Model parallelism splits a single model across multiple GPUs so that each device
holds a portion of the model's layers. This is distinct from data parallelism
(where each GPU holds a full model copy and splits the data). Pipeline-style
model parallelism places consecutive layer groups on different devices and moves
tensors between them during the forward pass.

What this tutorial demonstrates:
  1. Splitting SimpleViT transformer layers across multiple GPU devices
  2. Tensor movement between devices during forward/backward passes
  3. Memory distribution across devices (parameter count per GPU)
  4. Pipeline execution overhead vs. single-device baseline
  5. Benchmark comparison of baseline vs. pipeline model

Single-GPU behavior:
  On a single GPU, all layer groups are placed on cuda:0. The .to() calls
  become no-ops (tensor is already on the target device), so there is no
  inter-device transfer overhead. The tutorial still demonstrates the code
  structure so you can see how it would work on a multi-GPU setup.

This tutorial does NOT use torch.distributed or mp.spawn because model
parallelism is about splitting a model across devices within a single process.
It uses plain CUDA device placement.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from utils import (
    setup_logging,
    print_benchmark_table,
    SimpleViT,
    get_sample_batch,
    print_device_info,
)

NUM_ITERATIONS = 50
WARMUP_ITERATIONS = 10
BATCH_SIZE = 32


class PipelineViT(nn.Module):
    """SimpleViT split across multiple devices for pipeline-style model parallelism.

    Takes a pre-built SimpleViT and distributes its layers across the given
    devices. Embedding layers go on the first device, transformer layers are
    split evenly across all devices, and the classification head goes on the
    last device.

    Args:
        base_model: A SimpleViT instance (will be moved to devices in-place).
        devices: List of torch.device to distribute the model across.
    """

    def __init__(self, base_model: nn.Module, devices: list[torch.device]) -> None:
        super().__init__()
        self.devices = devices

        # Embedding layers on first device
        self.patch_embed = base_model.patch_embed.to(devices[0])
        self.cls_token = nn.Parameter(base_model.cls_token.data.to(devices[0]))
        self.pos_embed = nn.Parameter(base_model.pos_embed.data.to(devices[0]))

        # Split transformer layers across devices
        layers = list(base_model.transformer.layers)
        split_size = len(layers) // len(devices)
        self.layer_groups = nn.ModuleList()
        for i, dev in enumerate(devices):
            start = i * split_size
            end = start + split_size if i < len(devices) - 1 else len(layers)
            group = nn.Sequential(*layers[start:end]).to(dev)
            self.layer_groups.append(group)

        # Classification head on last device
        self.norm = base_model.norm.to(devices[-1])
        self.head = base_model.head.to(devices[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with tensor movement between devices.

        Args:
            x: Input tensor of shape (batch, 3, image_size, image_size).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        # Patch embedding on first device
        x = x.to(self.devices[0])
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Prepend class token and add positional embeddings (first device)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        # Pass through layer groups, moving tensors between devices
        for i, (layer_group, device) in enumerate(zip(self.layer_groups, self.devices)):
            prev_device = x.device
            x = x.to(device)
            if prev_device != device:
                # Log tensor movement only when device actually changes
                pass  # Movement logged at setup time; per-step logging would be too noisy
            x = layer_group(x)

        # Classification head on last device
        x = x.to(self.devices[-1])
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)
        return logits


def train_loop(
    model: nn.Module,
    device: torch.device,
    num_steps: int,
    warmup_steps: int,
    batch_size: int,
    logger,
    label: str,
    all_devices: list[torch.device] | None = None,
) -> dict:
    """Run a training loop and return benchmark results.

    Args:
        model: The model to train.
        device: Device for input data (first device of the model).
        num_steps: Total training steps (including warmup).
        warmup_steps: Number of warmup steps before timing.
        batch_size: Batch size for synthetic data.
        logger: Logger instance.
        label: Label for this benchmark run.
        all_devices: List of all devices the model spans (for memory tracking).

    Returns:
        Dict with 'name', 'time_seconds', 'memory_mb' keys.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Determine output device (where the model's head produces logits)
    if isinstance(model, PipelineViT):
        output_device = model.devices[-1]
    else:
        output_device = device

    if all_devices is None:
        all_devices = [device]

    # Reset memory stats on all devices
    for dev in all_devices:
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)

    logger.info(f"[{label}] Starting training: {num_steps} steps ({warmup_steps} warmup)")

    # Warmup phase
    for step in range(warmup_steps):
        inputs, labels = get_sample_batch(batch_size=batch_size, device=device)
        labels = labels.to(output_device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Synchronize before timing
    for dev in all_devices:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    # Timed phase
    timed_steps = num_steps - warmup_steps
    start_time = time.perf_counter()

    for step in range(timed_steps):
        inputs, labels = get_sample_batch(batch_size=batch_size, device=device)
        labels = labels.to(output_device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if step == 0:
            logger.info(f"[{label}] First timed step loss: {loss.item():.4f}")

    # Synchronize after timing
    for dev in all_devices:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    elapsed = time.perf_counter() - start_time

    # Track peak memory across all devices
    peak_memory_mb = 0.0
    for dev in all_devices:
        if dev.type == "cuda":
            mem = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
            peak_memory_mb = max(peak_memory_mb, mem)

    logger.info(f"[{label}] Completed {timed_steps} timed steps in {elapsed:.4f}s")
    logger.info(f"[{label}] Throughput: {timed_steps * batch_size / elapsed:.0f} samples/sec")
    logger.info(f"[{label}] Peak memory: {peak_memory_mb:.1f} MB")

    return {
        "name": label,
        "time_seconds": elapsed,
        "memory_mb": peak_memory_mb,
    }


def log_memory_distribution(model: nn.Module, devices: list[torch.device], logger) -> None:
    """Log parameter memory distribution across devices.

    For each device, computes total parameter memory and count, showing
    how the model is distributed.

    Args:
        model: The model to analyze.
        devices: List of devices the model is distributed across.
        logger: Logger instance.
    """
    logger.info("Memory distribution across devices:")
    total_params = 0
    total_memory = 0

    for dev in devices:
        device_params = 0
        device_memory = 0
        for p in model.parameters():
            if p.device == dev:
                device_params += p.nelement()
                device_memory += p.nelement() * p.element_size()
        total_params += device_params
        total_memory += device_memory
        mem_mb = device_memory / (1024 * 1024)
        logger.info(f"  [Device {dev}] Parameters: {device_params:,}, Memory: {mem_mb:.1f} MB")

    total_mem_mb = total_memory / (1024 * 1024)
    logger.info(f"  [Total] Parameters: {total_params:,}, Memory: {total_mem_mb:.1f} MB")


def main() -> None:
    """Run model parallelism tutorial demonstrating pipeline-style layer splitting."""
    logger = setup_logging("model_parallelism")

    print("\n" + "=" * 60)
    print("  Model Parallelism Tutorial")
    print("  Pipeline-Style Layer Splitting Across GPUs")
    print("=" * 60 + "\n")

    print_device_info()

    # Detect available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus == 0:
        logger.warning("No GPU available. Model parallelism requires CUDA GPUs.")
        logger.info("Skipping tutorial -- please run on a machine with at least 1 GPU.")
        return

    # Build device list
    if num_gpus >= 2:
        devices = [torch.device(f"cuda:{i}") for i in range(min(num_gpus, 4))]
        logger.info(f"Found {num_gpus} GPUs -- using {len(devices)} for model parallelism")
        for i, dev in enumerate(devices):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(dev)}")
    else:
        devices = [torch.device("cuda:0")]
        logger.info(
            "Running with 1 GPU -- model layers will all be on cuda:0. "
            "On multi-GPU setups, layers would be distributed across devices."
        )
        logger.info(f"  Device 0: {torch.cuda.get_device_name(0)}")

    # ==================================================================
    # Section 1: Baseline (Single Device)
    # ==================================================================
    print("\n--- Section 1: Baseline (Single Device) ---\n")

    logger.info("Creating SimpleViT model on first device as baseline...")
    logger.info("Architecture: dim=256, depth=4, heads=8, mlp_dim=512")

    baseline_model = SimpleViT(dim=256, depth=4, heads=8, mlp_dim=512).to(devices[0])
    total_params = sum(p.numel() for p in baseline_model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    baseline_result = train_loop(
        model=baseline_model,
        device=devices[0],
        num_steps=NUM_ITERATIONS,
        warmup_steps=WARMUP_ITERATIONS,
        batch_size=BATCH_SIZE,
        logger=logger,
        label="Baseline (single device)",
        all_devices=[devices[0]],
    )

    # Free baseline model memory
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==================================================================
    # Section 2: Pipeline Model Parallel
    # ==================================================================
    print("\n--- Section 2: Pipeline Model Parallel ---\n")

    logger.info("Creating fresh SimpleViT and wrapping with PipelineViT...")
    logger.info(f"Splitting {4} transformer layers across {len(devices)} device(s)")

    base_model = SimpleViT(dim=256, depth=4, heads=8, mlp_dim=512)
    pipeline_model = PipelineViT(base_model, devices)

    # Log how layers are distributed
    for i, (group, dev) in enumerate(zip(pipeline_model.layer_groups, devices)):
        num_layers = len(group)
        logger.info(f"  Layer group {i}: {num_layers} transformer layer(s) on {dev}")

    logger.info(f"  Embedding layers: {devices[0]}")
    logger.info(f"  Classification head: {devices[-1]}")

    # Show memory distribution
    log_memory_distribution(pipeline_model, devices, logger)

    pipeline_result = train_loop(
        model=pipeline_model,
        device=devices[0],
        num_steps=NUM_ITERATIONS,
        warmup_steps=WARMUP_ITERATIONS,
        batch_size=BATCH_SIZE,
        logger=logger,
        label="Pipeline parallel",
        all_devices=devices,
    )

    # ==================================================================
    # Analysis
    # ==================================================================
    print("\n--- Analysis ---\n")

    if len(devices) == 1:
        logger.info(
            "Single GPU analysis: Pipeline model parallelism on 1 GPU adds slight "
            "overhead from the PipelineViT wrapper (extra .to() calls are no-ops to "
            "the same device). There is no memory benefit since all layers remain on "
            "the same GPU."
        )
        logger.info(
            "On a multi-GPU setup, you would see: (1) memory distributed across "
            "devices -- each GPU holds only its assigned layers, (2) inter-device "
            "tensor transfers adding latency during forward/backward, (3) potential "
            "pipeline bubbles where GPUs wait for data from previous stages."
        )
        logger.info(
            "Model parallelism is most useful when a model is too large to fit on "
            "a single GPU. For models that fit on one GPU, data parallelism (DDP) "
            "is generally more efficient."
        )
    else:
        speedup = baseline_result["time_seconds"] / pipeline_result["time_seconds"]
        logger.info(
            f"Pipeline model achieved {speedup:.2f}x vs baseline. "
            "Model parallelism distributes memory across GPUs but inter-device "
            "transfers add latency. Pipeline scheduling (micro-batching) can help "
            "reduce bubble time in production setups."
        )

    # ==================================================================
    # Section 3: Benchmark Comparison
    # ==================================================================
    print("\n--- Section 3: Benchmark Comparison ---\n")

    results = [baseline_result, pipeline_result]
    print_benchmark_table(results)

    # Key takeaways
    print("\n--- Key Takeaways ---\n")
    logger.info("Model parallelism splits model layers across GPUs (pipeline-style)")
    logger.info("Each GPU holds a subset of layers and processes tensors in sequence")
    logger.info("Tensors must be moved between devices at layer group boundaries")
    logger.info("Best for models too large to fit on a single GPU")
    logger.info("For models that fit on one GPU, prefer DDP (data parallelism)")
    logger.info("See ddp_training.py for data parallelism and fsdp_training.py for FSDP")

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
