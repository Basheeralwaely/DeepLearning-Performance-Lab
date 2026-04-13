"""
FSDP Training Tutorial: FullyShardedDataParallel
==================================================

FullyShardedDataParallel (FSDP) is PyTorch's implementation of ZeRO-style
model sharding for memory-efficient distributed training. Unlike DDP which
replicates the full model on every GPU, FSDP shards model parameters,
gradients, and optimizer states across ranks, dramatically reducing per-GPU
memory usage for large models.

Sharding strategies compared:
  - FULL_SHARD: Shards parameters, gradients, and optimizer states (ZeRO-3)
  - SHARD_GRAD_OP: Shards gradients and optimizer states only (ZeRO-2)
  - NO_SHARD: No sharding, behaves like DDP (useful as baseline)

Key concepts demonstrated:
  1. Process group initialization for FSDP
  2. FSDP wrapping with different sharding strategies
  3. Per-rank memory usage logging (allocated and peak)
  4. Performance comparison across all sharding strategies

Hardware notes:
  - Multi-GPU: Full FSDP sharding benefits with NCCL backend
  - Single GPU: Runs with Gloo backend, all distributed ops execute
    but sharding provides no actual memory savings
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
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
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
MASTER_PORT = "12356"


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
        model: The model to train (FSDP-wrapped or plain).
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
    logger = setup_logging(f"fsdp_train_rank{rank}")

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


def log_memory_per_rank(label, rank, world_size, logger):
    """Log GPU memory usage for the current rank.

    Reports both currently allocated and peak allocated memory in MB.
    Uses barrier synchronization so ranks log in order.

    Args:
        label: Descriptive label for this memory snapshot.
        rank: Current process rank.
        world_size: Total number of processes.
        logger: Logger instance for output.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(rank) / (1024 * 1024)
        peak = torch.cuda.max_memory_allocated(rank) / (1024 * 1024)
        # Log each rank in order using barrier
        for r in range(world_size):
            if rank == r:
                logger.info(
                    f"[Rank {rank}] {label}: "
                    f"allocated={allocated:.1f} MB, peak={peak:.1f} MB"
                )
            dist.barrier()
    else:
        if rank == 0:
            logger.info(
                f"[Rank {rank}] {label}: "
                f"memory tracking not available (CPU mode)"
            )
        dist.barrier()


def fsdp_worker(rank, world_size):
    """Main worker function for each FSDP process.

    Orchestrates the full FSDP tutorial flow: process group setup,
    baseline training, three FSDP sharding strategies, and benchmark
    comparison.

    Args:
        rank: Process rank assigned by mp.spawn.
        world_size: Total number of processes.
    """
    setup(rank, world_size)
    logger = setup_logging(f"fsdp_rank{rank}")

    try:
        # ============================================================
        # Tutorial Header
        # ============================================================
        if rank == 0:
            print("\n" + "=" * 60)
            print("  FSDP Training Tutorial")
            print("=" * 60 + "\n")
            print_device_info()

        if world_size == 1:
            print_rank0(
                "Running with 1 GPU -- distributed ops will execute but "
                "no actual multi-GPU parallelism.",
                rank,
            )
            print_rank0("", rank)

        device = torch.device(
            f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        )

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
        # Section 2: Baseline (no FSDP)
        # ============================================================
        print_rank0("\n--- Section 2: Baseline Training (no FSDP) ---\n", rank)

        baseline_model = SimpleViT(
            dim=256, depth=4, heads=8, mlp_dim=512
        ).to(device)
        total_params = sum(p.numel() for p in baseline_model.parameters())
        log_rank0(
            f"Model: SimpleViT with {total_params:,} parameters", rank, logger
        )
        log_rank0(f"Device: {device}", rank, logger)
        log_rank0("Training baseline model without FSDP wrapping...", rank, logger)

        baseline_result = train_loop(
            baseline_model,
            device,
            rank,
            NUM_ITERATIONS,
            WARMUP_ITERATIONS,
            BATCH_SIZE,
        )
        log_memory_per_rank("Baseline", rank, world_size, logger)

        if rank == 0:
            logger.info(
                f"Baseline: {baseline_result['time_seconds']:.4f}s, "
                f"Memory: {baseline_result['memory_mb']}"
            )

        del baseline_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ============================================================
        # Section 3: FSDP with FULL_SHARD
        # ============================================================
        print_rank0(
            "\n--- Section 3: FSDP with FULL_SHARD (ZeRO-3) ---\n", rank
        )
        log_rank0(
            "FULL_SHARD: Shards parameters, gradients, AND optimizer states "
            "across all ranks. Maximum memory savings but highest "
            "communication overhead.",
            rank,
            logger,
        )

        full_shard_model = SimpleViT(
            dim=256, depth=4, heads=8, mlp_dim=512
        ).to(device)
        full_shard_model = FSDP(
            full_shard_model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=rank if torch.cuda.is_available() else None,
        )
        log_rank0("Model wrapped with FSDP (FULL_SHARD)", rank, logger)

        full_shard_result = train_loop(
            full_shard_model,
            device,
            rank,
            NUM_ITERATIONS,
            WARMUP_ITERATIONS,
            BATCH_SIZE,
        )
        log_memory_per_rank("FSDP FULL_SHARD", rank, world_size, logger)

        if rank == 0:
            logger.info(
                f"FULL_SHARD: {full_shard_result['time_seconds']:.4f}s, "
                f"Memory: {full_shard_result['memory_mb']}"
            )

        del full_shard_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ============================================================
        # Section 4: FSDP with SHARD_GRAD_OP
        # ============================================================
        print_rank0(
            "\n--- Section 4: FSDP with SHARD_GRAD_OP (ZeRO-2) ---\n", rank
        )
        log_rank0(
            "SHARD_GRAD_OP: Shards gradients and optimizer states only. "
            "Parameters remain replicated. Lower communication cost than "
            "FULL_SHARD but uses more memory per rank.",
            rank,
            logger,
        )

        shard_grad_model = SimpleViT(
            dim=256, depth=4, heads=8, mlp_dim=512
        ).to(device)
        shard_grad_model = FSDP(
            shard_grad_model,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            device_id=rank if torch.cuda.is_available() else None,
        )
        log_rank0("Model wrapped with FSDP (SHARD_GRAD_OP)", rank, logger)

        shard_grad_result = train_loop(
            shard_grad_model,
            device,
            rank,
            NUM_ITERATIONS,
            WARMUP_ITERATIONS,
            BATCH_SIZE,
        )
        log_memory_per_rank("FSDP SHARD_GRAD_OP", rank, world_size, logger)

        if rank == 0:
            logger.info(
                f"SHARD_GRAD_OP: {shard_grad_result['time_seconds']:.4f}s, "
                f"Memory: {shard_grad_result['memory_mb']}"
            )

        del shard_grad_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ============================================================
        # Section 5: FSDP with NO_SHARD
        # ============================================================
        print_rank0(
            "\n--- Section 5: FSDP with NO_SHARD (DDP-like) ---\n", rank
        )
        log_rank0(
            "NO_SHARD: No sharding at all -- behaves like DDP. "
            "Useful as a baseline within the FSDP API to measure sharding "
            "overhead vs benefit.",
            rank,
            logger,
        )

        no_shard_model = SimpleViT(
            dim=256, depth=4, heads=8, mlp_dim=512
        ).to(device)
        no_shard_model = FSDP(
            no_shard_model,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            device_id=rank if torch.cuda.is_available() else None,
        )
        log_rank0("Model wrapped with FSDP (NO_SHARD)", rank, logger)

        no_shard_result = train_loop(
            no_shard_model,
            device,
            rank,
            NUM_ITERATIONS,
            WARMUP_ITERATIONS,
            BATCH_SIZE,
        )
        log_memory_per_rank("FSDP NO_SHARD", rank, world_size, logger)

        if rank == 0:
            logger.info(
                f"NO_SHARD: {no_shard_result['time_seconds']:.4f}s, "
                f"Memory: {no_shard_result['memory_mb']}"
            )

        del no_shard_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ============================================================
        # Section 6: Benchmark Comparison
        # ============================================================
        if rank == 0:
            print("\n--- Section 6: Benchmark Comparison ---\n")

            results = [
                {
                    "name": "Baseline (no FSDP)",
                    "time_seconds": baseline_result["time_seconds"],
                    "memory_mb": baseline_result["memory_mb"],
                },
                {
                    "name": "FULL_SHARD",
                    "time_seconds": full_shard_result["time_seconds"],
                    "memory_mb": full_shard_result["memory_mb"],
                },
                {
                    "name": "SHARD_GRAD_OP",
                    "time_seconds": shard_grad_result["time_seconds"],
                    "memory_mb": shard_grad_result["memory_mb"],
                },
                {
                    "name": "NO_SHARD",
                    "time_seconds": no_shard_result["time_seconds"],
                    "memory_mb": no_shard_result["memory_mb"],
                },
            ]
            print_benchmark_table(results)

            # Sharding trade-off analysis
            logger.info("Sharding Strategy Trade-offs:")
            logger.info(
                "  FULL_SHARD (ZeRO-3): Maximum memory savings, highest "
                "communication cost. Best for very large models that don't "
                "fit on a single GPU."
            )
            logger.info(
                "  SHARD_GRAD_OP (ZeRO-2): Moderate memory savings, lower "
                "communication than FULL_SHARD. Good balance for most "
                "use cases."
            )
            logger.info(
                "  NO_SHARD: No memory savings from sharding (like DDP). "
                "Useful as a baseline or when model fits comfortably in "
                "GPU memory."
            )

            if world_size == 1:
                logger.info(
                    "Note: With 1 GPU, sharding strategies show overhead "
                    "without memory benefits. Run with multiple GPUs to "
                    "see actual memory savings from FSDP."
                )

    finally:
        cleanup()


def main():
    """Entry point: detect GPU count and spawn FSDP workers."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size < 1:
        world_size = 1

    print(f"Launching FSDP tutorial with world_size={world_size}")
    mp.spawn(fsdp_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
