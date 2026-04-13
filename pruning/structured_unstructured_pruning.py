"""
Structured vs Unstructured Pruning: A Practical Comparison
===========================================================

Pruning removes redundant weights from neural networks to reduce model size
and (potentially) speed up inference. There are two fundamental approaches:

**Unstructured Pruning (L1 Magnitude):**
  - Zeros out individual weights based on their absolute magnitude
  - The weight tensors keep their ORIGINAL shape -- zeros stored in dense format
  - Standard GPU kernels still process every element, including zeros
  - Result: smaller file if compressed, but NO inference speedup on standard GPUs
  - Requires sparse hardware (e.g., NVIDIA A100 2:4 sparsity) for actual speedup

**Structured Pruning (Channel Removal):**
  - Removes entire output channels (filters) from convolutional layers
  - Produces physically smaller weight tensors with fewer dimensions
  - Standard GPU kernels run faster because there is genuinely less computation
  - Result: real model compression AND real inference speedup

This tutorial demonstrates both techniques on a SimpleCNN, with an iterative
prune-then-fine-tune workflow at multiple sparsity levels. Benchmark tables
compare model size and inference speed across all pruning configurations.

When to use:
  - Unstructured pruning: when targeting sparse hardware or compression formats
  - Structured pruning: when you need real inference speedup on standard GPUs
"""

import copy
import os
import sys
import logging
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

logger = setup_logging("pruning_tutorial")

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
SPARSITY_LEVELS = [0.2, 0.5, 0.7, 0.9]
BATCH_SIZE = 64
NUM_INFERENCE_ITERATIONS = 100
FINE_TUNE_EPOCHS = 2
LEARNING_RATE = 0.001
NUM_VAL_BATCHES = 5


# ---------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------

class PrunedCNN(nn.Module):
    """A structurally pruned CNN with physically smaller layers.

    Used by build_pruned_model() to produce a proper nn.Module subclass
    (instead of monkey-patching forward on a bare nn.Module), ensuring
    compatibility with torch.jit.script, torch.compile, and serialization.
    """

    def __init__(self, features, classifier):
        super().__init__()
        self.features = features
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def measure_model_size(model):
    """Measure parameter count and serialized file size of a model.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        dict with 'param_count' (int) and 'file_size_mb' (float).
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
def run_inference(model, inputs, num_iterations):
    """Run repeated inference passes for timing measurement.

    Args:
        model: Model in eval mode.
        inputs: Input tensor batch.
        num_iterations: Number of forward passes to run.

    Returns:
        None (timing captured by benchmark decorator).
    """
    with torch.inference_mode():
        for _ in range(num_iterations):
            _ = model(inputs)
    return None


def fine_tune(model, device, epochs, lr):
    """Run a short fine-tuning pass on synthetic data after pruning.

    This simulates the iterative prune-then-fine-tune workflow that helps
    the model recover from weight removal.

    Args:
        model: Model to fine-tune (modified in-place).
        device: Target device.
        epochs: Number of training iterations.
        lr: Learning rate for SGD optimizer.
    """
    logger.info(f"Fine-tuning for {epochs} epochs...")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        inputs, labels = get_sample_batch(batch_size=BATCH_SIZE, device=device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        logger.info(f"  Epoch {epoch + 1}/{epochs} -- loss: {loss.item():.4f}")


def evaluate_accuracy(model, device, num_batches):
    """Evaluate model accuracy on synthetic validation data.

    Runs inference on fixed synthetic batches and computes the fraction
    of predictions (argmax of logits) matching synthetic labels.

    Args:
        model: Model in eval mode.
        device: Target device.
        num_batches: Number of synthetic batches to evaluate on.

    Returns:
        float: Accuracy as a fraction (0.0 to 1.0).
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


def build_pruned_model(original_model, prune_ratio, device):
    """Build a physically smaller model by structurally pruning channels.

    Applies structured L1 pruning to the first conv layer (features.0),
    identifies surviving channels, then constructs a new model with
    smaller layer dimensions and copies the relevant weights.

    Args:
        original_model: The baseline SimpleCNN model.
        prune_ratio: Fraction of output channels to remove (0.0 to 1.0).
        device: Target device.

    Returns:
        A new nn.Module with physically smaller layers.
    """
    # Deep-copy and apply structured pruning to identify which channels survive
    temp_model = copy.deepcopy(original_model)
    conv0 = temp_model.features[0]

    # Apply L1 structured pruning along output channel dimension (dim=0)
    prune.ln_structured(conv0, name="weight", amount=prune_ratio, n=1, dim=0)

    # Identify surviving channel indices from the pruning mask
    # mask shape: (out_channels, in_channels, kH, kW)
    mask = conv0.weight_mask
    # A channel survives if any of its mask values are non-zero
    surviving_mask = mask.sum(dim=(1, 2, 3)) > 0
    surviving_indices = torch.where(surviving_mask)[0]
    num_surviving = len(surviving_indices)

    logger.info(
        f"  Structured pruning: {conv0.weight_orig.shape[0]} -> "
        f"{num_surviving} channels in features.0 "
        f"({int(prune_ratio * 100)}% removed)"
    )

    # Make pruning permanent to get clean weights
    prune.remove(conv0, "weight")

    # Build new physically smaller model
    # features.0: Conv2d(3, num_surviving, 3, padding=1)
    new_conv0 = nn.Conv2d(3, num_surviving, kernel_size=3, padding=1)
    new_conv0.weight.data = conv0.weight.data[surviving_indices].clone()
    new_conv0.bias.data = conv0.bias.data[surviving_indices].clone()

    # features.3: Conv2d(num_surviving, 64, 3, padding=1) -- input channels change
    old_conv1 = temp_model.features[3]
    new_conv1 = nn.Conv2d(num_surviving, 64, kernel_size=3, padding=1)
    new_conv1.weight.data = old_conv1.weight.data[:, surviving_indices].clone()
    new_conv1.bias.data = old_conv1.bias.data.clone()

    # Build new feature extractor
    new_features = nn.Sequential(
        new_conv0,
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        new_conv1,
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        copy.deepcopy(temp_model.features[6]),   # Conv2d(64, 128, 3) unchanged
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )

    # Build complete model using proper nn.Module subclass
    new_model = PrunedCNN(new_features, copy.deepcopy(temp_model.classifier))

    return new_model.to(device)


# ---------------------------------------------------------------
# Main tutorial
# ---------------------------------------------------------------

def main():
    # ==============================================================
    # SETUP
    # ==============================================================
    print("\n" + "=" * 60)
    print("  Pruning Tutorial: Structured vs Unstructured Comparison")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    # Create baseline model
    baseline_model = SimpleCNN().to(device)
    baseline_model.eval()

    total_params = sum(p.numel() for p in baseline_model.parameters())
    logger.info(f"Baseline model: {total_params:,} parameters")

    baseline_size = measure_model_size(baseline_model)
    logger.info(
        f"Baseline file size: {baseline_size['file_size_mb']:.2f} MB"
    )

    # Sample input for inference benchmarks
    inputs, _ = get_sample_batch(batch_size=BATCH_SIZE, device=device)

    # Warm-up: ensure CUDA kernels are compiled before timed runs
    logger.info("Running warm-up passes...")
    with torch.inference_mode():
        for _ in range(10):
            _ = baseline_model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure baseline inference
    baseline_bench = run_inference(baseline_model, inputs, NUM_INFERENCE_ITERATIONS)
    logger.info(
        f"Baseline inference: {baseline_bench['time_seconds']:.4f}s "
        f"for {NUM_INFERENCE_ITERATIONS} iterations"
    )

    # Measure baseline accuracy
    baseline_acc = evaluate_accuracy(baseline_model, device, NUM_VAL_BATCHES)
    logger.info(f"Baseline accuracy: {baseline_acc * 100:.1f}%")

    # Collect all results for final comparison tables
    bench_results = [
        {
            "name": "Baseline (no pruning)",
            "time_seconds": baseline_bench["time_seconds"],
            "memory_mb": baseline_bench.get("memory_mb"),
        }
    ]
    size_results = [
        {
            "name": "Baseline (no pruning)",
            "param_count": baseline_size["param_count"],
            "file_size_mb": baseline_size["file_size_mb"],
            "accuracy": baseline_acc,
        }
    ]

    # ==============================================================
    # PART 1: Unstructured Pruning (L1 Magnitude)
    # ==============================================================
    print("\n" + "=" * 60)
    print("  PART 1: Unstructured Pruning (L1 Magnitude)")
    print("=" * 60 + "\n")

    logger.info(
        "Unstructured pruning zeros out individual weights by magnitude. "
        "The weight tensor stays the SAME SIZE -- zeros are stored in a "
        "dense tensor. Standard GPU kernels process all elements including "
        "zeros, so there is NO inference speedup on standard hardware."
    )

    for sparsity in SPARSITY_LEVELS:
        print(f"\n--- Unstructured Pruning: {int(sparsity * 100)}% sparsity ---\n")

        # Deep-copy baseline
        model_copy = copy.deepcopy(baseline_model)

        # Apply L1 unstructured pruning to all Conv2d and Linear layers
        pruned_layers = []
        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name="weight", amount=sparsity)
                pruned_layers.append((name, module))

        logger.info(
            f"Applied L1 unstructured pruning at {int(sparsity * 100)}% "
            f"to {len(pruned_layers)} layers"
        )

        # Fine-tune to recover from pruning
        fine_tune(model_copy, device, FINE_TUNE_EPOCHS, LEARNING_RATE)

        # Make pruning permanent (remove masks, bake zeros into weights)
        for name, module in pruned_layers:
            prune.remove(module, "weight")

        model_copy.eval()

        # Measure model size
        size_info = measure_model_size(model_copy)

        # Count zeros
        total_weights = 0
        zero_weights = 0
        for p in model_copy.parameters():
            total_weights += p.numel()
            zero_weights += (p == 0).sum().item()

        logger.info(
            f"  Sparsity {int(sparsity * 100)}%: "
            f"{zero_weights:,} / {total_weights:,} weights are zero "
            f"({(zero_weights / total_weights * 100) if total_weights > 0 else 0.0:.1f}% actual sparsity)"
        )
        logger.info(
            f"  Parameters: {size_info['param_count']:,} (unchanged -- "
            f"zeros still stored in dense tensor)"
        )
        logger.info(f"  File size: {size_info['file_size_mb']:.2f} MB")

        # Measure inference speed
        bench = run_inference(model_copy, inputs, NUM_INFERENCE_ITERATIONS)
        logger.info(f"  Inference: {bench['time_seconds']:.4f}s")

        # Measure accuracy after pruning
        acc = evaluate_accuracy(model_copy, device, NUM_VAL_BATCHES)
        logger.info(f"  Accuracy: {acc * 100:.1f}%")

        label = f"Unstructured {int(sparsity * 100)}%"
        bench_results.append({
            "name": label,
            "time_seconds": bench["time_seconds"],
            "memory_mb": bench.get("memory_mb"),
        })
        size_results.append({
            "name": label,
            "param_count": size_info["param_count"],
            "file_size_mb": size_info["file_size_mb"],
            "accuracy": acc,
        })

    # Critical explanation about unstructured pruning and inference speed
    print("\n" + "-" * 60)
    print(
        "NOTE: Unstructured pruning does NOT reduce inference time on "
        "standard GPUs.\nThe weight tensors remain dense (same shape), "
        "and standard CUDA kernels\nprocess all elements including zeros. "
        "Sparse hardware (e.g., NVIDIA A100\nstructured sparsity support) "
        "is required for actual speedup from\nunstructured sparsity."
    )
    print("-" * 60 + "\n")

    # ==============================================================
    # PART 2: Structured Pruning (Channel Removal)
    # ==============================================================
    print("\n" + "=" * 60)
    print("  PART 2: Structured Pruning (Channel Removal)")
    print("=" * 60 + "\n")

    logger.info(
        "Structured pruning physically removes entire channels/filters, "
        "producing genuinely smaller layers. This results in real inference "
        "speedup because the GPU has less computation to perform."
    )

    for prune_ratio in [0.25, 0.5]:
        print(
            f"\n--- Structured Pruning: {int(prune_ratio * 100)}% "
            f"channels removed ---\n"
        )

        # Build a physically smaller model
        pruned_model = build_pruned_model(baseline_model, prune_ratio, device)

        # Fine-tune the pruned model
        fine_tune(pruned_model, device, FINE_TUNE_EPOCHS, LEARNING_RATE)

        pruned_model.eval()

        # Measure size
        size_info = measure_model_size(pruned_model)
        logger.info(
            f"  New parameter count: {size_info['param_count']:,} "
            f"(was {baseline_size['param_count']:,})"
        )
        logger.info(f"  File size: {size_info['file_size_mb']:.2f} MB")

        # Measure inference speed
        bench = run_inference(pruned_model, inputs, NUM_INFERENCE_ITERATIONS)
        logger.info(f"  Inference: {bench['time_seconds']:.4f}s")

        # Measure accuracy after pruning
        acc = evaluate_accuracy(pruned_model, device, NUM_VAL_BATCHES)
        logger.info(f"  Accuracy: {acc * 100:.1f}%")

        label = f"Structured {int(prune_ratio * 100)}% removed"
        bench_results.append({
            "name": label,
            "time_seconds": bench["time_seconds"],
            "memory_mb": bench.get("memory_mb"),
        })
        size_results.append({
            "name": label,
            "param_count": size_info["param_count"],
            "file_size_mb": size_info["file_size_mb"],
            "accuracy": acc,
        })

    # ==============================================================
    # BENCHMARK COMPARISON TABLES
    # ==============================================================
    print("\n" + "=" * 60)
    print("  Benchmark Comparison")
    print("=" * 60 + "\n")

    # Model size comparison table
    print("--- Model Size Comparison ---\n")
    print(f"+{'-' * 32}+{'-' * 14}+{'-' * 16}+{'-' * 12}+")
    print(
        f"| {'Configuration':<30} | {'Params':>12} | {'File Size (MB)':>14} | "
        f"{'Accuracy':>10} |"
    )
    print(f"+{'-' * 32}+{'-' * 14}+{'-' * 16}+{'-' * 12}+")
    for r in size_results:
        print(
            f"| {r['name']:<30} | {r['param_count']:>12,} | "
            f"{r['file_size_mb']:>14.2f} | "
            f"{r.get('accuracy', 0) * 100:>9.1f}% |"
        )
    print(f"+{'-' * 32}+{'-' * 14}+{'-' * 16}+{'-' * 12}+")

    # Inference speed comparison table
    print("\n--- Inference Speed Comparison ---\n")
    print_benchmark_table(bench_results)

    # ==============================================================
    # Key Takeaways
    # ==============================================================
    print("\n--- Key Takeaways ---\n")
    logger.info(
        "1. Unstructured pruning zeros individual weights but keeps tensor "
        "shapes identical -- no speedup on standard GPUs"
    )
    logger.info(
        "2. Structured pruning physically removes channels, producing "
        "smaller tensors and real speedup"
    )
    logger.info(
        "3. Fine-tuning after pruning helps recover accuracy "
        "(iterative prune-then-fine-tune)"
    )
    logger.info(
        "4. Choose structured pruning when targeting standard GPU inference; "
        "unstructured when targeting sparse hardware or compression"
    )

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
