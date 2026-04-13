"""
DeepSpeed ZeRO: Memory-Efficient Distributed Training
======================================================

DeepSpeed's ZeRO (Zero Redundancy Optimizer) progressively partitions training
state across GPUs to reduce per-device memory. The three stages are:

  Stage 1: Optimizer State Partitioning -- each GPU stores only 1/N of the
           optimizer states (e.g., Adam momentum + variance).
  Stage 2: + Gradient Partitioning -- gradients are also partitioned. Each GPU
           reduces only its assigned portion of gradients.
  Stage 3: + Parameter Partitioning -- model parameters themselves are split.
           Each GPU holds only 1/N of parameters, gathering as needed during
           forward/backward passes. Maximum memory savings.

CPU Offloading: Optimizer states (Stage 2+) and parameters (Stage 3) can be
offloaded to CPU RAM, trading compute speed for memory capacity.

What this tutorial demonstrates:
  1. ZeRO Stage 1, 2, 3 configuration and initialization
  2. Memory reduction progression across stages
  3. CPU offloading for optimizer states and parameters
  4. Benchmark comparison of all configurations
  5. Graceful handling when DeepSpeed is not installed

Requirements:
  pip install deepspeed
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from utils import (
    setup_logging,
    print_benchmark_table,
    SimpleViT,
    get_sample_batch,
    print_device_info,
)

# Graceful dependency handling per D-08
try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

NUM_ITERATIONS = 50
WARMUP_ITERATIONS = 10
BATCH_SIZE = 32
MASTER_PORT = "12358"


def get_deepspeed_config(
    stage: int,
    offload_optimizer: bool = False,
    offload_params: bool = False,
) -> dict:
    """Build a DeepSpeed configuration dict for the given ZeRO stage.

    Uses an inline Python dict (not a JSON file) to keep the tutorial
    self-contained per D-09.

    Args:
        stage: ZeRO optimization stage (1, 2, or 3).
        offload_optimizer: If True and stage >= 2, offload optimizer states to CPU.
        offload_params: If True and stage == 3, offload parameters to CPU.

    Returns:
        DeepSpeed configuration dictionary.
    """
    config = {
        "train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8},
        },
        "zero_optimization": {
            "stage": stage,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8,
        },
        "fp16": {"enabled": False},
    }

    if offload_optimizer and stage >= 2:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if offload_params and stage == 3:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    return config


def setup(rank: int, world_size: int) -> None:
    """Initialize the distributed process group.

    CRITICAL: Environment variables MUST be set BEFORE init_process_group
    because DeepSpeed reads them during initialization (pitfall 4 from research).

    Args:
        rank: Process rank.
        world_size: Total number of processes.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = MASTER_PORT

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup() -> None:
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def log_rank0(logger, msg: str, rank: int) -> None:
    """Log a message only from rank 0 to avoid duplicate output.

    Args:
        logger: Logger instance.
        msg: Message to log.
        rank: Current process rank.
    """
    if rank == 0:
        logger.info(msg)


def print_rank0(msg: str, rank: int) -> None:
    """Print a message only from rank 0.

    Args:
        msg: Message to print.
        rank: Current process rank.
    """
    if rank == 0:
        print(msg)


def train_deepspeed(
    model_engine,
    device: torch.device,
    rank: int,
    num_steps: int,
    warmup_steps: int,
    batch_size: int,
    logger,
) -> dict:
    """Run a DeepSpeed training loop and return benchmark results.

    Uses the DeepSpeed engine API: model_engine(inputs), model_engine.backward(loss),
    model_engine.step() -- not the standard PyTorch optimizer API.

    Args:
        model_engine: DeepSpeed model engine from deepspeed.initialize().
        device: Device for input data.
        rank: Process rank.
        num_steps: Total training steps (including warmup).
        warmup_steps: Number of warmup steps before timing.
        batch_size: Batch size for synthetic data.
        logger: Logger instance.

    Returns:
        Dict with 'time_seconds' and 'memory_mb' keys.
    """
    criterion = nn.CrossEntropyLoss()

    # Warmup phase
    for step in range(warmup_steps):
        inputs, labels = get_sample_batch(batch_size=batch_size, device=device)
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)
        model_engine.backward(loss)
        model_engine.step()

    # Synchronize all ranks before timing
    dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    timed_steps = num_steps - warmup_steps
    start_time = time.perf_counter()

    for step in range(timed_steps):
        inputs, labels = get_sample_batch(batch_size=batch_size, device=device)
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)
        model_engine.backward(loss)
        model_engine.step()

        if step == 0 and rank == 0:
            logger.info(f"  First timed step loss: {loss.item():.4f}")

    # Synchronize after timing
    dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start_time

    memory_mb = 0.0
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "time_seconds": elapsed,
        "memory_mb": memory_mb,
    }


def explain_zero_concepts(logger) -> None:
    """Explain ZeRO concepts via logging when DeepSpeed is not available.

    Provides a detailed explanation of each ZeRO stage and CPU offloading
    so users can learn the concepts even without the library installed.

    Args:
        logger: Logger instance.
    """
    print("\n--- ZeRO Optimizer Concepts ---\n")

    logger.info(
        "ZeRO Stage 1: Optimizer State Partitioning -- each GPU stores only "
        "1/N of optimizer states (e.g., Adam momentum + variance). "
        "Reduces optimizer memory by N."
    )
    logger.info(
        "  Example: Adam with 100M params needs ~1.2GB for states. "
        "With 4 GPUs, each GPU stores ~300MB."
    )

    logger.info(
        "ZeRO Stage 2: + Gradient Partitioning -- gradients are also "
        "partitioned. Each GPU reduces only its assigned gradients. "
        "Further reduces memory."
    )
    logger.info(
        "  Example: Gradients for 100M float32 params = ~400MB. "
        "With 4 GPUs, each GPU handles ~100MB of gradients."
    )

    logger.info(
        "ZeRO Stage 3: + Parameter Partitioning -- model parameters "
        "themselves are partitioned. Each GPU holds only 1/N of parameters, "
        "gathering as needed for forward/backward. Maximum memory savings."
    )
    logger.info(
        "  Example: 100M float32 params = ~400MB. "
        "With 4 GPUs, each GPU stores ~100MB of params, gathering full "
        "params only during compute."
    )

    logger.info(
        "CPU Offloading: Optimizer states (Stage 2+) and parameters "
        "(Stage 3) can be offloaded to CPU RAM, trading compute speed "
        "for memory capacity."
    )
    logger.info(
        "  Stage 2 + CPU offload: optimizer states live in CPU RAM"
    )
    logger.info(
        "  Stage 3 + full offload: both params and optimizer states in CPU RAM"
    )

    print("\n--- Memory Reduction Summary ---\n")
    logger.info("Stage 0 (baseline): Full model + optimizer + gradients on each GPU")
    logger.info("Stage 1: ~4x optimizer memory reduction (with 4 GPUs)")
    logger.info("Stage 2: ~8x total memory reduction (optimizer + gradients)")
    logger.info("Stage 3: ~N x memory reduction (everything partitioned)")
    logger.info("+ CPU offload: Further reduction by moving state to CPU RAM")


def deepspeed_worker(rank: int, world_size: int) -> None:
    """Main worker function for DeepSpeed ZeRO tutorial.

    Runs ZeRO Stages 1, 2, 3, and CPU offloading configurations,
    collecting benchmark results for comparison.

    Args:
        rank: Process rank.
        world_size: Total number of processes.
    """
    setup(rank, world_size)
    logger = setup_logging(f"deepspeed_zero_rank{rank}")

    try:
        # Title and device info (rank 0 only)
        if rank == 0:
            print("\n" + "=" * 60)
            print("  DeepSpeed ZeRO Tutorial")
            print("  Memory-Efficient Distributed Training")
            print("=" * 60 + "\n")

            print_device_info()

        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # Single-GPU fallback per D-03
        if world_size == 1 and rank == 0:
            logger.info(
                "Running with 1 GPU -- ZeRO partitioning still executes but "
                "memory savings from sharding will not be visible (world_size=1). "
                "On multi-GPU setups, each stage would show progressive memory reduction."
            )

        # Check for DeepSpeed availability per D-08
        if not HAS_DEEPSPEED:
            if rank == 0:
                logger.warning(
                    "DeepSpeed not installed. Install with: pip install deepspeed"
                )
                logger.info(
                    "Showing DeepSpeed ZeRO concepts via explanation below..."
                )
                explain_zero_concepts(logger)
            return

        # ==============================================================
        # Section 1: Process Group Setup
        # ==============================================================
        if rank == 0:
            print("\n--- Section 1: Process Group Setup ---\n")

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        log_rank0(logger, f"Backend: {backend}", rank)
        log_rank0(logger, f"Rank: {rank}, World size: {world_size}", rank)
        log_rank0(
            logger,
            "Process group initialized BEFORE deepspeed.initialize() -- "
            "this is required when using mp.spawn (see pitfall 4 in research).",
            rank,
        )

        results = []

        # ==============================================================
        # Section 2: ZeRO Stage 1
        # ==============================================================
        if rank == 0:
            print("\n--- Section 2: ZeRO Stage 1 (Optimizer Partitioning) ---\n")

        log_rank0(
            logger,
            "ZeRO Stage 1 partitions optimizer states across GPUs. "
            "Adam stores momentum + variance per param = ~2x param memory. "
            "Stage 1 reduces this by 1/N across N GPUs.",
            rank,
        )

        model = SimpleViT(dim=256, depth=4, heads=8, mlp_dim=512)
        config = get_deepspeed_config(stage=1)
        log_rank0(logger, f"Config: ZeRO stage=1, batch_size={BATCH_SIZE}", rank)

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config,
        )

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(device) / (1024 * 1024)
            log_rank0(logger, f"Memory after ZeRO-1 init: {mem:.1f} MB", rank)

        result = train_deepspeed(
            model_engine, device, rank, NUM_ITERATIONS, WARMUP_ITERATIONS,
            BATCH_SIZE, logger,
        )
        result["name"] = "ZeRO Stage 1"
        results.append(result)
        log_rank0(
            logger,
            f"ZeRO-1: {result['time_seconds']:.4f}s, {result['memory_mb']:.1f} MB",
            rank,
        )

        del model_engine, optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ==============================================================
        # Section 3: ZeRO Stage 2
        # ==============================================================
        if rank == 0:
            print("\n--- Section 3: ZeRO Stage 2 (+ Gradient Partitioning) ---\n")

        log_rank0(
            logger,
            "ZeRO Stage 2 adds gradient partitioning on top of Stage 1. "
            "Each GPU reduces only its assigned gradient shard, further "
            "reducing memory.",
            rank,
        )

        model = SimpleViT(dim=256, depth=4, heads=8, mlp_dim=512)
        config = get_deepspeed_config(stage=2)
        log_rank0(logger, f"Config: ZeRO stage=2, batch_size={BATCH_SIZE}", rank)

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config,
        )

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(device) / (1024 * 1024)
            log_rank0(logger, f"Memory after ZeRO-2 init: {mem:.1f} MB", rank)

        result = train_deepspeed(
            model_engine, device, rank, NUM_ITERATIONS, WARMUP_ITERATIONS,
            BATCH_SIZE, logger,
        )
        result["name"] = "ZeRO Stage 2"
        results.append(result)
        log_rank0(
            logger,
            f"ZeRO-2: {result['time_seconds']:.4f}s, {result['memory_mb']:.1f} MB",
            rank,
        )

        del model_engine, optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ==============================================================
        # Section 4: ZeRO Stage 3
        # ==============================================================
        if rank == 0:
            print("\n--- Section 4: ZeRO Stage 3 (+ Parameter Partitioning) ---\n")

        log_rank0(
            logger,
            "ZeRO Stage 3 partitions everything: optimizer states, gradients, "
            "AND parameters. Each GPU holds only 1/N of the model. Parameters "
            "are gathered on-the-fly during forward/backward. Maximum memory savings.",
            rank,
        )

        model = SimpleViT(dim=256, depth=4, heads=8, mlp_dim=512)
        config = get_deepspeed_config(stage=3)
        log_rank0(logger, f"Config: ZeRO stage=3, batch_size={BATCH_SIZE}", rank)

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config,
        )

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(device) / (1024 * 1024)
            log_rank0(logger, f"Memory after ZeRO-3 init: {mem:.1f} MB", rank)

        result = train_deepspeed(
            model_engine, device, rank, NUM_ITERATIONS, WARMUP_ITERATIONS,
            BATCH_SIZE, logger,
        )
        result["name"] = "ZeRO Stage 3"
        results.append(result)
        log_rank0(
            logger,
            f"ZeRO-3: {result['time_seconds']:.4f}s, {result['memory_mb']:.1f} MB",
            rank,
        )

        del model_engine, optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ==============================================================
        # Section 5: ZeRO Stage 2 + CPU Offloading
        # ==============================================================
        if rank == 0:
            print("\n--- Section 5: ZeRO Stage 2 + CPU Offloading ---\n")

        log_rank0(
            logger,
            "ZeRO Stage 2 with CPU offloading moves optimizer states to CPU RAM. "
            "This frees GPU memory at the cost of CPU<->GPU data transfer overhead.",
            rank,
        )

        model = SimpleViT(dim=256, depth=4, heads=8, mlp_dim=512)
        config = get_deepspeed_config(stage=2, offload_optimizer=True)
        log_rank0(
            logger,
            f"Config: ZeRO stage=2, offload_optimizer=True, batch_size={BATCH_SIZE}",
            rank,
        )

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config,
        )

        log_rank0(logger, "Optimizer states offloaded to CPU RAM", rank)
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(device) / (1024 * 1024)
            log_rank0(logger, f"Memory after ZeRO-2+Offload init: {mem:.1f} MB", rank)

        result = train_deepspeed(
            model_engine, device, rank, NUM_ITERATIONS, WARMUP_ITERATIONS,
            BATCH_SIZE, logger,
        )
        result["name"] = "ZeRO-2 + CPU Offload"
        results.append(result)
        log_rank0(
            logger,
            f"ZeRO-2+Offload: {result['time_seconds']:.4f}s, {result['memory_mb']:.1f} MB",
            rank,
        )

        del model_engine, optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ==============================================================
        # Section 6: ZeRO Stage 3 + Full Offloading
        # ==============================================================
        if rank == 0:
            print("\n--- Section 6: ZeRO Stage 3 + Full Offloading ---\n")

        log_rank0(
            logger,
            "ZeRO Stage 3 with full offloading moves BOTH optimizer states "
            "AND parameters to CPU RAM. Maximum memory savings but highest "
            "CPU<->GPU transfer overhead.",
            rank,
        )

        model = SimpleViT(dim=256, depth=4, heads=8, mlp_dim=512)
        config = get_deepspeed_config(stage=3, offload_optimizer=True, offload_params=True)
        log_rank0(
            logger,
            f"Config: ZeRO stage=3, offload_optimizer=True, offload_params=True, "
            f"batch_size={BATCH_SIZE}",
            rank,
        )

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=config,
        )

        log_rank0(logger, "Parameters AND optimizer states offloaded to CPU RAM", rank)
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated(device) / (1024 * 1024)
            log_rank0(logger, f"Memory after ZeRO-3+Full-Offload init: {mem:.1f} MB", rank)

        result = train_deepspeed(
            model_engine, device, rank, NUM_ITERATIONS, WARMUP_ITERATIONS,
            BATCH_SIZE, logger,
        )
        result["name"] = "ZeRO-3 + Full Offload"
        results.append(result)
        log_rank0(
            logger,
            f"ZeRO-3+Full-Offload: {result['time_seconds']:.4f}s, {result['memory_mb']:.1f} MB",
            rank,
        )

        del model_engine, optimizer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ==============================================================
        # Section 7: Benchmark Comparison
        # ==============================================================
        if rank == 0:
            print("\n--- Section 7: Benchmark Comparison ---\n")

            print_benchmark_table(results)

            print("\n--- Analysis ---\n")
            logger.info("Memory reduction progression across ZeRO stages:")
            for r in results:
                logger.info(f"  {r['name']:<24} Time: {r['time_seconds']:.4f}s  Memory: {r['memory_mb']:.1f} MB")

            if len(results) >= 3:
                stage1_mem = results[0]["memory_mb"]
                stage3_mem = results[2]["memory_mb"]
                if stage1_mem > 0:
                    reduction = (1 - stage3_mem / stage1_mem) * 100
                    logger.info(
                        f"Stage 3 vs Stage 1 memory: {reduction:.1f}% reduction"
                    )

            if world_size == 1:
                logger.info(
                    "Note: With world_size=1, ZeRO partitioning has no peers to shard "
                    "across. Memory differences come from DeepSpeed's internal management. "
                    "With N GPUs, each stage would show ~1/N memory reduction."
                )

            logger.info(
                "CPU offloading trades compute speed for memory capacity. "
                "Use it when GPU memory is the bottleneck (large models)."
            )

            print("\n--- Key Takeaways ---\n")
            logger.info("ZeRO Stage 1: Partition optimizer states -> reduce optimizer memory by N")
            logger.info("ZeRO Stage 2: + Partition gradients -> further memory reduction")
            logger.info("ZeRO Stage 3: + Partition parameters -> maximum memory savings")
            logger.info("CPU Offloading: Move state to CPU RAM when GPU memory is tight")
            logger.info("Higher stages = more memory savings but more communication overhead")
            logger.info("See ddp_training.py for basic data parallelism and fsdp_training.py for FSDP")

            print("\n" + "=" * 60)
            print("  Tutorial Complete")
            print("=" * 60 + "\n")

    finally:
        cleanup()


def main() -> None:
    """Launch DeepSpeed ZeRO tutorial across available GPUs."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size < 1:
        world_size = 1
    mp.spawn(deepspeed_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
