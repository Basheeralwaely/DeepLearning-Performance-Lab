"""
DDP Training Tutorial: DistributedDataParallel
================================================

DistributedDataParallel (DDP) is PyTorch's primary tool for synchronous
data-parallel training across multiple GPUs. DDP replicates the model on
each GPU, splits data across replicas, and synchronizes gradients via
all-reduce after each backward pass so every replica stays in sync.

Key concepts demonstrated:
  1. Process group initialization (init_process_group with NCCL/Gloo backend)
  2. DDP model wrapping and gradient synchronization verification
  3. Training loop with barrier-synchronized timing
  4. Scaling efficiency comparison: baseline vs DDP

Hardware notes:
  - Multi-GPU: Full DDP parallelism with NCCL backend
  - Single GPU: Runs with Gloo backend, all distributed ops execute
    but there is no actual multi-GPU parallelism
  - CPU-only: Falls back to Gloo backend on CPU tensors

Launch method: mp.spawn (no torchrun required -- fully self-contained)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
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
MASTER_PORT = "12355"


def setup(rank, world_size):
    """Initialize the distributed process group.

    Sets up MASTER_ADDR and MASTER_PORT environment variables, then
    initializes the process group with NCCL (GPU) or Gloo (CPU) backend.

    Args:
        rank: Process rank (0 to world_size-1).
        world_size: Total number of processes.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = MASTER_PORT

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def log_rank0(msg, rank, logger):
    """Log a message only from rank 0.

    Args:
        msg: Message string to log.
        rank: Current process rank.
        logger: Logger instance.
    """
    if rank == 0:
        logger.info(msg)


def print_rank0(msg, rank):
    """Print a message only from rank 0.

    Args:
        msg: Message string to print.
        rank: Current process rank.
    """
    if rank == 0:
        print(msg)


def train_loop(model, device, rank, num_steps, warmup_steps, batch_size):
    """Run a training loop with timing and memory measurement.

    Executes warmup steps first (untimed), then timed training steps
    with barrier synchronization for accurate multi-process timing.

    Args:
        model: The model to train (DDP-wrapped or plain).
        device: torch.device to use.
        rank: Current process rank.
        num_steps: Total training steps (including warmup).
        warmup_steps: Number of untimed warmup steps.
        batch_size: Batch size per process.

    Returns:
        Dict with 'time_seconds' and 'memory_mb' (from rank 0 only).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    logger = setup_logging(f"ddp_train_rank{rank}")

    # Warmup phase (untimed)
    model.train()
    for step in range(warmup_steps):
        inputs, labels = get_sample_batch(batch_size=batch_size, device=device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Synchronize before timed section
    dist.barrier()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    # Timed training phase
    timed_steps = num_steps - warmup_steps
    for step in range(timed_steps):
        inputs, labels = get_sample_batch(batch_size=batch_size, device=device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0 and (step + 1) % 10 == 0:
            logger.info(
                f"Step [{step + 1}/{timed_steps}] - Loss: {loss.item():.4f}"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # Synchronize after timed section
    dist.barrier()

    elapsed = time.perf_counter() - start_time

    memory_mb = None
    if torch.cuda.is_available() and rank == 0:
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {"time_seconds": elapsed, "memory_mb": memory_mb}


def verify_gradient_sync(ddp_model, device, rank, world_size, logger):
    """Verify that DDP synchronizes gradients across all ranks.

    Runs one forward+backward pass, then uses all_reduce to compare
    gradient norms across ranks. In DDP, all ranks should have identical
    gradients after backward().

    Args:
        ddp_model: DDP-wrapped model.
        device: torch.device to use.
        rank: Current process rank.
        world_size: Total number of processes.
        logger: Logger instance for output.
    """
    criterion = nn.CrossEntropyLoss()

    inputs, labels = get_sample_batch(batch_size=BATCH_SIZE, device=device)
    outputs = ddp_model(inputs)
    loss = criterion(outputs, labels)

    ddp_model.zero_grad()
    loss.backward()

    # Compute local gradient norm
    local_norm = torch.tensor(0.0, device=device)
    for param in ddp_model.parameters():
        if param.grad is not None:
            local_norm += param.grad.data.norm(2).item() ** 2
    local_norm = local_norm.sqrt()

    # Gather norms from all ranks via all_reduce (sum)
    norm_sum = local_norm.clone()
    dist.all_reduce(norm_sum, op=dist.ReduceOp.SUM)

    # Each rank should have the same gradient norm (DDP guarantees this)
    expected_sum = local_norm * world_size

    if rank == 0:
        logger.info(f"Gradient norm on rank 0: {local_norm.item():.6f}")
        logger.info(f"Sum of all gradient norms: {norm_sum.item():.6f}")
        logger.info(
            f"Expected sum ({world_size} x rank0 norm): "
            f"{expected_sum.item():.6f}"
        )
        if abs(norm_sum.item() - expected_sum.item()) < 1e-3:
            logger.info(
                "VERIFIED: Gradients are synchronized across all ranks!"
            )
        else:
            logger.warning(
                "WARNING: Gradient norms differ across ranks. "
                "This may indicate a DDP configuration issue."
            )


def ddp_worker(rank, world_size):
    """Main worker function for each DDP process.

    Orchestrates the full DDP tutorial flow: process group setup,
    model creation, gradient verification, baseline and DDP training,
    and benchmark comparison.

    Args:
        rank: Process rank assigned by mp.spawn.
        world_size: Total number of processes.
    """
    setup(rank, world_size)
    logger = setup_logging(f"ddp_rank{rank}")

    try:
        # ============================================================
        # Tutorial Header
        # ============================================================
        if rank == 0:
            print("\n" + "=" * 60)
            print("  DDP Training Tutorial")
            print("=" * 60 + "\n")
            print_device_info()

        if world_size == 1:
            print_rank0(
                "Running with 1 GPU -- distributed ops will execute but "
                "no actual multi-GPU parallelism.",
                rank,
            )
            print_rank0("", rank)

        # ============================================================
        # Section 1: Process Group Setup
        # ============================================================
        print_rank0("\n--- Section 1: Process Group Setup ---\n", rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        log_rank0(f"Backend: {backend}", rank, logger)
        log_rank0(f"Rank: {rank} / World size: {world_size}", rank, logger)
        log_rank0(
            f"Process group initialized successfully on {backend} backend",
            rank,
            logger,
        )

        # ============================================================
        # Section 2: Model Creation & DDP Wrapping
        # ============================================================
        print_rank0("\n--- Section 2: Model Creation & DDP Wrapping ---\n", rank)

        device = torch.device(
            f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        )

        model = SimpleViT(dim=256, depth=4, heads=8, mlp_dim=512).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        log_rank0(f"Model: SimpleViT with {total_params:,} parameters", rank, logger)
        log_rank0(f"Device: {device}", rank, logger)

        ddp_model = DDP(
            model,
            device_ids=[rank] if torch.cuda.is_available() else None,
        )
        log_rank0(
            "Model wrapped with DistributedDataParallel", rank, logger
        )

        # ============================================================
        # Section 3: Gradient Synchronization Verification
        # ============================================================
        print_rank0(
            "\n--- Section 3: Gradient Synchronization Verification ---\n",
            rank,
        )
        log_rank0(
            "Running one forward+backward pass to verify gradient sync...",
            rank,
            logger,
        )
        verify_gradient_sync(ddp_model, device, rank, world_size, logger)

        # ============================================================
        # Section 4: Baseline Training (no DDP)
        # ============================================================
        print_rank0("\n--- Section 4: Baseline Training (no DDP) ---\n", rank)

        if rank == 0:
            logger.info("Training baseline model without DDP wrapping...")
            baseline_model = SimpleViT(
                dim=256, depth=4, heads=8, mlp_dim=512
            ).to(device)
            baseline_result = train_loop(
                baseline_model,
                device,
                rank,
                NUM_ITERATIONS,
                WARMUP_ITERATIONS,
                BATCH_SIZE,
            )
            logger.info(
                f"Baseline: {baseline_result['time_seconds']:.4f}s, "
                f"Memory: {baseline_result['memory_mb']}"
            )
        else:
            baseline_result = None

        # Ensure all ranks wait for baseline to complete
        dist.barrier()

        # ============================================================
        # Section 5: DDP Training
        # ============================================================
        print_rank0("\n--- Section 5: DDP Training ---\n", rank)
        log_rank0(
            f"Training DDP model with {world_size} process(es)...",
            rank,
            logger,
        )

        ddp_result = train_loop(
            ddp_model,
            device,
            rank,
            NUM_ITERATIONS,
            WARMUP_ITERATIONS,
            BATCH_SIZE,
        )

        if rank == 0:
            logger.info(
                f"DDP: {ddp_result['time_seconds']:.4f}s, "
                f"Memory: {ddp_result['memory_mb']}"
            )

        # ============================================================
        # Section 6: Benchmark Results
        # ============================================================
        if rank == 0:
            print("\n--- Section 6: Benchmark Results ---\n")

            results = [
                {
                    "name": "Baseline (no DDP)",
                    "time_seconds": baseline_result["time_seconds"],
                    "memory_mb": baseline_result["memory_mb"],
                },
                {
                    "name": f"DDP ({world_size} GPU(s))",
                    "time_seconds": ddp_result["time_seconds"],
                    "memory_mb": ddp_result["memory_mb"],
                },
            ]
            print_benchmark_table(results)

            # Scaling analysis
            if baseline_result["time_seconds"] > 0:
                speedup = (
                    baseline_result["time_seconds"]
                    / ddp_result["time_seconds"]
                )
                efficiency = speedup / world_size * 100
                logger.info(f"Speedup: {speedup:.2f}x")
                logger.info(f"Scaling efficiency: {efficiency:.1f}%")

                if world_size == 1:
                    logger.info(
                        "Note: With 1 GPU, DDP adds overhead for process "
                        "group management. Speedup is expected with multiple "
                        "GPUs."
                    )
                else:
                    logger.info(
                        f"With {world_size} GPUs, ideal speedup would be "
                        f"{world_size:.1f}x. Communication overhead reduces "
                        f"this to {speedup:.2f}x."
                    )

    finally:
        cleanup()


def main():
    """Entry point: detect GPU count and spawn DDP workers."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size < 1:
        world_size = 1

    print(f"Launching DDP tutorial with world_size={world_size}")
    mp.spawn(ddp_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
