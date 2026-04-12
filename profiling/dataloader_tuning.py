"""
DataLoader Tuning: Optimizing Data Loading for GPU Training
============================================================

The DataLoader is often the hidden bottleneck in GPU training pipelines.
While engineers focus on model architecture and GPU utilization, slow data
loading can leave the GPU idle between batches, wasting expensive compute.

Three key DataLoader parameters control throughput:

  - num_workers: Number of subprocesses for data loading. More workers can
    overlap CPU data preparation with GPU computation, but too many waste
    resources on context switching and memory.

  - pin_memory: When True, allocates data in page-locked (pinned) memory,
    enabling asynchronous CPU-to-GPU transfers via non_blocking=True.
    This avoids a synchronous copy through pageable memory.

  - prefetch_factor: Controls how many batches each worker pre-loads ahead
    of time. Higher values smooth out I/O variance but increase memory.

This tutorial uses a synthetic dataset with configurable I/O delay to
demonstrate these effects reproducibly. The artificial delay simulates
real-world disk reads (image decoding, augmentation, etc.) so the
bottleneck patterns are visible even without a large dataset on disk.

What this tutorial demonstrates:
  1. num_workers sweep across 4 configurations (0, 2, 4, 8)
  2. pin_memory impact on CPU-to-GPU transfer speed
  3. prefetch_factor tuning for pipeline smoothness
  4. End-to-end training pipeline comparison (data-bound vs optimized)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import (
    setup_logging,
    benchmark,
    compare_results,
    print_benchmark_table,
    SimpleCNN,
    get_device,
    print_device_info,
)

logger = setup_logging("dataloader_tuning")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_SIZE = 500
IO_DELAY = 0.01  # 10ms simulated disk I/O per sample
BATCH_SIZE = 64
NUM_BATCHES = 50  # batches to measure per configuration
NUM_WORKERS_CONFIGS = [0, 2, 4, 8]  # 4 configurations to sweep
IMAGE_SIZE = 32
NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Synthetic Dataset with controllable I/O delay
# ---------------------------------------------------------------------------
class SyntheticIODataset(Dataset):
    """A synthetic dataset that simulates disk I/O latency per sample.

    Each __getitem__ call sleeps for `io_delay` seconds before returning
    a random image tensor and label. This makes data loading the clear
    bottleneck, allowing us to measure how DataLoader parameters affect
    throughput independently of model speed.

    Args:
        size: Number of samples in the dataset.
        io_delay: Seconds to sleep per sample (simulates disk read).
        image_size: Spatial dimension of square images.
    """

    def __init__(self, size=1000, io_delay=0.01, image_size=32):
        self.size = size
        self.io_delay = io_delay
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        time.sleep(self.io_delay)
        image = torch.randn(3, self.image_size, self.image_size)
        label = torch.randint(0, NUM_CLASSES, (1,)).item()
        return image, label


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------
def measure_throughput(loader, num_batches, device):
    """Iterate a DataLoader and measure batches-per-second throughput.

    Args:
        loader: A torch DataLoader to iterate.
        num_batches: Maximum number of batches to consume.
        device: Target device for data transfer.

    Returns:
        Batches per second (float).
    """
    start = time.perf_counter()
    count = 0
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        count += 1
        if count >= num_batches:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return count / elapsed if elapsed > 0 else float("inf")


def main():
    # ==================================================================
    # Section 1: Setup
    # ==================================================================
    print("\n" + "=" * 60)
    print("  DataLoader Tuning: Optimizing Data Loading for GPU Training")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    cpu_count = multiprocessing.cpu_count()
    logger.info(f"CPU cores available: {cpu_count}")
    logger.info(f"Dataset size: {DATASET_SIZE} samples")
    logger.info(f"I/O delay per sample: {IO_DELAY * 1000:.0f}ms")
    logger.info(f"Batch size: {BATCH_SIZE}")
    expected_bottleneck = IO_DELAY * BATCH_SIZE
    logger.info(
        f"Expected bottleneck per batch (sequential): "
        f"{expected_bottleneck * 1000:.0f}ms "
        f"({IO_DELAY * 1000:.0f}ms x {BATCH_SIZE} samples)"
    )
    logger.info(f"Batches per measurement: {NUM_BATCHES}")

    dataset = SyntheticIODataset(
        size=DATASET_SIZE, io_delay=IO_DELAY, image_size=IMAGE_SIZE
    )

    # ==================================================================
    # Section 2: num_workers Sweep
    # ==================================================================
    print("\n--- Section 2: num_workers Sweep ---\n")
    logger.info(
        "Testing how the number of data loading workers affects throughput."
    )
    logger.info(
        "With num_workers=0, the main process loads data sequentially. "
        "Workers > 0 use subprocesses to overlap loading with GPU compute."
    )

    sweep_results = []
    throughputs = {}

    for nw in NUM_WORKERS_CONFIGS:
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=nw,
            pin_memory=True,
            prefetch_factor=2 if nw > 0 else None,
        )
        throughput = measure_throughput(loader, NUM_BATCHES, device)
        throughputs[nw] = throughput
        seconds_per_batch = 1.0 / throughput if throughput > 0 else float("inf")
        logger.info(
            f"  workers={nw}: {throughput:.2f} batches/sec "
            f"({seconds_per_batch * 1000:.1f} ms/batch)"
        )
        sweep_results.append({
            "name": f"workers={nw}",
            "time_seconds": seconds_per_batch,
            "memory_mb": None,
        })

    print_benchmark_table(sweep_results)

    # Analysis
    best_nw = max(throughputs, key=throughputs.get)
    baseline_tp = throughputs[0]
    best_tp = throughputs[best_nw]
    speedup_vs_zero = best_tp / baseline_tp if baseline_tp > 0 else float("inf")
    logger.info(f"Fastest config: num_workers={best_nw} ({best_tp:.2f} batches/sec)")
    logger.info(f"Speedup vs num_workers=0: {speedup_vs_zero:.2f}x")
    logger.info(
        f"CPU cores: {cpu_count}. Rule of thumb: start with num_workers = "
        f"num_cpus (here {cpu_count}), then tune down if memory-constrained."
    )

    # ==================================================================
    # Section 3: pin_memory Impact
    # ==================================================================
    print("\n--- Section 3: pin_memory Impact ---\n")
    test_nw = best_nw if best_nw > 0 else 4
    logger.info(
        f"Comparing pin_memory=True vs False with num_workers={test_nw}"
    )
    logger.info(
        "Pinned (page-locked) memory enables async CPU-to-GPU transfer "
        "via non_blocking=True, avoiding a synchronous copy through "
        "pageable memory."
    )

    pin_results = []
    for pin in [True, False]:
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=test_nw,
            pin_memory=pin,
            prefetch_factor=2,
        )
        tp = measure_throughput(loader, NUM_BATCHES, device)
        spb = 1.0 / tp if tp > 0 else float("inf")
        label = f"pin_memory={pin}"
        logger.info(f"  {label}: {tp:.2f} batches/sec ({spb * 1000:.1f} ms/batch)")
        pin_results.append({
            "name": label,
            "time_seconds": spb,
            "memory_mb": None,
        })

    print_benchmark_table(pin_results)

    if pin_results[0]["time_seconds"] < pin_results[1]["time_seconds"]:
        pin_speedup = (
            pin_results[1]["time_seconds"] / pin_results[0]["time_seconds"]
        )
        logger.info(
            f"pin_memory=True is {pin_speedup:.2f}x faster for data transfer."
        )
    else:
        logger.info(
            "pin_memory effect is minimal here (I/O delay dominates transfer time)."
        )
    logger.info(
        "Impact is larger with big tensors and real GPU workloads where "
        "transfer overlaps with compute."
    )

    # ==================================================================
    # Section 4: prefetch_factor Comparison
    # ==================================================================
    print("\n--- Section 4: prefetch_factor Comparison ---\n")
    logger.info(
        f"Comparing prefetch_factor values with num_workers={test_nw}"
    )
    logger.info(
        "prefetch_factor controls how many batches each worker pre-loads "
        "ahead of time. Higher values smooth out I/O variance but increase "
        "memory usage."
    )

    prefetch_values = [1, 2, 4]
    prefetch_results = []

    for pf in prefetch_values:
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=test_nw,
            pin_memory=True,
            prefetch_factor=pf,
        )
        tp = measure_throughput(loader, NUM_BATCHES, device)
        spb = 1.0 / tp if tp > 0 else float("inf")
        label = f"prefetch_factor={pf}"
        logger.info(f"  {label}: {tp:.2f} batches/sec ({spb * 1000:.1f} ms/batch)")
        prefetch_results.append({
            "name": label,
            "time_seconds": spb,
            "memory_mb": None,
        })

    print_benchmark_table(prefetch_results)

    # ==================================================================
    # Section 5: End-to-End Training Pipeline Comparison
    # ==================================================================
    print("\n--- Section 5: End-to-End Training Pipeline Comparison ---\n")
    logger.info(
        "Comparing full training step (dataload + forward + backward) "
        "with num_workers=0 vs best config."
    )

    model = SimpleCNN(input_size=IMAGE_SIZE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()

    num_train_steps = 20

    def run_training_steps(loader, steps):
        """Run a fixed number of training steps and return elapsed time."""
        step = 0
        for data, target in loader:
            if step >= steps:
                break
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            step += 1

    # Slow config: num_workers=0
    slow_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
    )

    @benchmark
    def train_slow():
        run_training_steps(slow_loader, num_train_steps)

    # Fast config: best num_workers + pin_memory
    fast_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=test_nw,
        pin_memory=True,
        prefetch_factor=2,
    )

    @benchmark
    def train_fast():
        run_training_steps(fast_loader, num_train_steps)

    logger.info(f"Running {num_train_steps} training steps with num_workers=0...")
    slow_result = train_slow()
    logger.info(f"  Slow pipeline: {slow_result['time_seconds']:.3f}s")

    logger.info(
        f"Running {num_train_steps} training steps with "
        f"num_workers={test_nw}, pin_memory=True..."
    )
    fast_result = train_fast()
    logger.info(f"  Fast pipeline: {fast_result['time_seconds']:.3f}s")

    compare_results(slow_result, fast_result, "DataLoader Tuning")

    # ==================================================================
    # Section 6: Key Takeaways
    # ==================================================================
    print("\n--- Key Takeaways ---\n")
    logger.info(
        "1. num_workers sweet spot: Start with CPU count, tune down for "
        "memory-constrained systems. Too many workers cause context-switching overhead."
    )
    logger.info(
        "2. pin_memory is essential for GPU training: Enables async transfers "
        "via non_blocking=True, preventing the GPU from waiting on data."
    )
    logger.info(
        "3. prefetch_factor tuning: Higher values (2-4) smooth I/O variance "
        "but increase memory. Default of 2 is usually good."
    )
    logger.info(
        "4. How to identify data-bound training: If GPU utilization is low "
        "while CPU is busy, your pipeline is data-bound. Use nvidia-smi and "
        "htop side-by-side to diagnose."
    )
    logger.info(
        "5. Profile DataLoader separately before optimizing the model. "
        "A 2x data loading speedup is free performance that no model "
        "optimization can match."
    )

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
