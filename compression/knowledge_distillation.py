"""
Knowledge Distillation: Teacher-Student Model Compression
==========================================================

Knowledge distillation (Hinton et al., 2015) is a model compression technique
where a smaller "student" model learns to mimic the behavior of a larger
"teacher" model. Instead of training the student only on hard labels (one-hot
ground truth), the student also learns from the teacher's soft probability
distribution over all classes.

Key concepts:
  - Temperature scaling: Dividing logits by a temperature T > 1 before softmax
    produces softer probability distributions that reveal inter-class
    similarities the teacher has learned. Higher T means softer distributions.

  - Soft targets: The teacher's softened output probabilities carry richer
    information than hard labels -- they encode which classes the teacher
    considers similar (the "dark knowledge").

  - Distillation loss: A weighted combination of two terms:
      L = alpha * KL_div(student_soft, teacher_soft) * T^2 + (1-alpha) * CE(student, labels)
    The T^2 factor compensates for the reduced gradient magnitudes from
    temperature scaling. Alpha controls the balance between mimicking the
    teacher (soft loss) and fitting the true labels (hard loss).

What this tutorial demonstrates:
  1. Training a larger teacher model on synthetic data
  2. Training a smaller student model from scratch (baseline)
  3. Distilling the teacher's knowledge into an identical student architecture
  4. Three-way comparison of model size and inference speed
"""

import os
import sys
import logging
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (
    setup_logging,
    benchmark,
    print_benchmark_table,
    get_sample_batch,
    get_device,
    print_device_info,
)

logger = setup_logging("distillation_tutorial")

# ── Hyperparameters ──────────────────────────────────────────────────────────
TEMPERATURE = 4.0
ALPHA = 0.7
BATCH_SIZE = 64
TEACHER_EPOCHS = 10
STUDENT_EPOCHS = 10
DISTILL_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_INFERENCE_ITERATIONS = 100
NUM_CLASSES = 10
INPUT_SIZE = 32
NUM_VAL_BATCHES = 5


# ── Model Definitions ───────────────────────────────────────────────────────

class TeacherCNN(nn.Module):
    """Larger SimpleCNN variant: channels 64->128->256, classifier->512->10.

    Same 3-conv + classifier architecture family as SimpleCNN but with wider
    layers, producing a model with significantly more parameters that serves
    as the knowledge source for distillation.
    """

    def __init__(self, input_size=INPUT_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        reduced_size = input_size // 8  # 3x MaxPool2d(2) halves each dim
        flat_dim = 256 * reduced_size * reduced_size
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class StudentCNN(nn.Module):
    """Smaller SimpleCNN variant: channels 16->32->64, classifier->128->10.

    Same 3-conv + classifier architecture family but with narrower layers,
    producing a compact model suitable as the distillation target.
    """

    def __init__(self, input_size=INPUT_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        reduced_size = input_size // 8  # 3x MaxPool2d(2) halves each dim
        flat_dim = 64 * reduced_size * reduced_size
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Helper Functions ─────────────────────────────────────────────────────────

def measure_model_size(model):
    """Measure model parameter count and serialized file size.

    Returns:
        dict with "param_count" (int) and "file_size_mb" (float).
    """
    param_count = sum(p.numel() for p in model.parameters())

    # Save to temp file to measure actual serialized size
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
    """Run repeated inference passes for benchmarking throughput."""
    with torch.inference_mode():
        for _ in range(num_iterations):
            _ = model(inputs)
        if inputs.device.type == "cuda":
            torch.cuda.synchronize()


def train_model(model, device, epochs, lr):
    """Train a model on synthetic data using cross-entropy loss.

    Generates 5 fixed synthetic batches and cycles through them each epoch
    to provide consistent training signal with deterministic data.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-generate fixed synthetic batches for consistent training
    batches = [
        get_sample_batch(batch_size=BATCH_SIZE, device=device) for _ in range(5)
    ]

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in batches:
            logits = model(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(batches)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}"
            )


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


def distill_knowledge(teacher, student, device, epochs, lr, temperature, alpha):
    """Transfer knowledge from teacher to student using Hinton's distillation.

    The distillation loss combines:
      - Soft target loss: KL divergence between temperature-scaled softmax outputs
        of teacher and student, multiplied by T^2 to correct gradient magnitude.
      - Hard target loss: Standard cross-entropy with ground truth labels.

    Final loss = alpha * soft_loss + (1 - alpha) * hard_loss
    """
    teacher.eval()
    student.train()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    # Pre-generate fixed synthetic batches for consistent distillation
    batches = [
        get_sample_batch(batch_size=BATCH_SIZE, device=device) for _ in range(5)
    ]

    for epoch in range(epochs):
        epoch_soft = 0.0
        epoch_hard = 0.0
        epoch_total = 0.0

        for inputs, labels in batches:
            student_logits = student(inputs)

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Soft target loss: KL divergence with temperature scaling
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature * temperature)

            # Hard target loss: standard cross-entropy
            hard_loss = F.cross_entropy(student_logits, labels)

            # Combined loss
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_soft += soft_loss.item()
            epoch_hard += hard_loss.item()
            epoch_total += loss.item()

        avg_soft = epoch_soft / len(batches)
        avg_hard = epoch_hard / len(batches)
        avg_total = epoch_total / len(batches)

        if (epoch + 1) % 2 == 0 or epoch == 0:
            logger.info(
                f"  Distill epoch {epoch + 1}/{epochs} | "
                f"Loss: {avg_total:.4f} (soft: {avg_soft:.4f}, "
                f"hard: {avg_hard:.4f})"
            )


# ── Main Tutorial ────────────────────────────────────────────────────────────

def main():
    # ── Section A: Setup ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Knowledge Distillation: Teacher-Student Compression")
    print("=" * 60 + "\n")

    device = get_device()
    print_device_info()

    teacher = TeacherCNN().to(device)
    student_scratch = StudentCNN().to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student_scratch.parameters())
    compression_ratio = teacher_params / student_params

    logger.info(f"Teacher: {teacher_params:,} params")
    logger.info(f"Student: {student_params:,} params")
    logger.info(
        f"Compression ratio: {compression_ratio:.1f}x "
        f"({student_params / teacher_params * 100:.1f}% of teacher)"
    )

    # ── Section B: Train Teacher ─────────────────────────────────────────
    print("\n--- STEP 1: Training Teacher Model ---\n")
    logger.info(
        f"Training TeacherCNN for {TEACHER_EPOCHS} epochs with "
        f"lr={LEARNING_RATE}..."
    )
    train_model(teacher, device, TEACHER_EPOCHS, LEARNING_RATE)
    teacher.eval()

    teacher_acc = evaluate_accuracy(teacher, device, NUM_VAL_BATCHES)
    logger.info(f"Teacher accuracy: {teacher_acc * 100:.1f}%")

    teacher_size = measure_model_size(teacher)
    logger.info(
        f"Teacher size: {teacher_size['param_count']:,} params, "
        f"{teacher_size['file_size_mb']:.2f} MB"
    )

    # Warm-up
    x_bench, _ = get_sample_batch(batch_size=BATCH_SIZE, device=device)
    with torch.inference_mode():
        for _ in range(10):
            _ = teacher(x_bench)
    if device.type == "cuda":
        torch.cuda.synchronize()

    teacher_bench = run_inference(teacher, x_bench, NUM_INFERENCE_ITERATIONS)
    logger.info(
        f"Teacher inference: {teacher_bench['time_seconds']:.4f}s "
        f"({NUM_INFERENCE_ITERATIONS} iterations)"
    )

    # ── Section C: Train Student From Scratch ────────────────────────────
    print("\n--- STEP 2: Training Student From Scratch (Baseline) ---\n")
    logger.info(
        f"Training StudentCNN from scratch for {STUDENT_EPOCHS} epochs with "
        f"lr={LEARNING_RATE}..."
    )
    train_model(student_scratch, device, STUDENT_EPOCHS, LEARNING_RATE)
    student_scratch.eval()

    scratch_acc = evaluate_accuracy(student_scratch, device, NUM_VAL_BATCHES)
    logger.info(f"Student (scratch) accuracy: {scratch_acc * 100:.1f}%")

    scratch_size = measure_model_size(student_scratch)
    logger.info(
        f"Student (scratch) size: {scratch_size['param_count']:,} params, "
        f"{scratch_size['file_size_mb']:.2f} MB"
    )

    # Warm-up
    with torch.inference_mode():
        for _ in range(10):
            _ = student_scratch(x_bench)
    if device.type == "cuda":
        torch.cuda.synchronize()

    scratch_bench = run_inference(
        student_scratch, x_bench, NUM_INFERENCE_ITERATIONS
    )
    logger.info(
        f"Student (scratch) inference: {scratch_bench['time_seconds']:.4f}s "
        f"({NUM_INFERENCE_ITERATIONS} iterations)"
    )

    # ── Section D: Distill Knowledge Into Student ────────────────────────
    print("\n--- STEP 3: Knowledge Distillation (Teacher -> Student) ---\n")
    logger.info(f"Temperature: {TEMPERATURE} (higher = softer distributions)")
    hard_weight = 1.0 - ALPHA
    logger.info(
        f"Alpha: {ALPHA} (soft target weight: {ALPHA}, "
        f"hard target weight: {hard_weight:.1f})"
    )
    logger.info(
        f"Loss = {ALPHA} * KL_div(student_soft, teacher_soft) * T^2 "
        f"+ {hard_weight:.1f} * CE(student, labels)"
    )
    logger.info("")

    student_distilled = StudentCNN().to(device)
    logger.info(
        f"Distilling teacher knowledge into fresh StudentCNN for "
        f"{DISTILL_EPOCHS} epochs..."
    )
    distill_knowledge(
        teacher,
        student_distilled,
        device,
        DISTILL_EPOCHS,
        LEARNING_RATE,
        TEMPERATURE,
        ALPHA,
    )
    student_distilled.eval()

    distill_acc = evaluate_accuracy(student_distilled, device, NUM_VAL_BATCHES)
    logger.info(f"Student (distilled) accuracy: {distill_acc * 100:.1f}%")

    distill_size = measure_model_size(student_distilled)
    logger.info(
        f"Student (distilled) size: {distill_size['param_count']:,} params, "
        f"{distill_size['file_size_mb']:.2f} MB"
    )

    # Warm-up
    with torch.inference_mode():
        for _ in range(10):
            _ = student_distilled(x_bench)
    if device.type == "cuda":
        torch.cuda.synchronize()

    distill_bench = run_inference(
        student_distilled, x_bench, NUM_INFERENCE_ITERATIONS
    )
    logger.info(
        f"Student (distilled) inference: {distill_bench['time_seconds']:.4f}s "
        f"({NUM_INFERENCE_ITERATIONS} iterations)"
    )

    # ── Section E: Three-Way Comparison ──────────────────────────────────
    print("\n--- RESULTS: Three-Way Comparison ---\n")

    # Model size comparison table
    logger.info("Model Size Comparison:")
    print()
    print(f"+{'-' * 26}+{'-' * 14}+{'-' * 16}+{'-' * 20}+{'-' * 12}+")
    print(
        f"| {'Model':<24} | {'Params':>12} | {'Size (MB)':>14} | "
        f"{'Compression Ratio':>18} | {'Accuracy':>10} |"
    )
    print(f"+{'-' * 26}+{'-' * 14}+{'-' * 16}+{'-' * 20}+{'-' * 12}+")

    models_info = [
        ("Teacher", teacher_size, 1.0, teacher_acc),
        ("Student (scratch)", scratch_size, teacher_size["param_count"] / scratch_size["param_count"], scratch_acc),
        ("Student (distilled)", distill_size, teacher_size["param_count"] / distill_size["param_count"], distill_acc),
    ]
    for name, size_info, ratio, acc in models_info:
        print(
            f"| {name:<24} | {size_info['param_count']:>12,} | "
            f"{size_info['file_size_mb']:>14.2f} | {ratio:>17.1f}x | "
            f"{acc * 100:>9.1f}% |"
        )
    print(f"+{'-' * 26}+{'-' * 14}+{'-' * 16}+{'-' * 20}+{'-' * 12}+")
    print()

    # Inference speed comparison using print_benchmark_table
    logger.info("Inference Speed Comparison:")
    results = [
        {
            "name": "Teacher",
            "time_seconds": teacher_bench["time_seconds"],
            "memory_mb": teacher_bench.get("memory_mb"),
        },
        {
            "name": "Student (scratch)",
            "time_seconds": scratch_bench["time_seconds"],
            "memory_mb": scratch_bench.get("memory_mb"),
        },
        {
            "name": "Student (distilled)",
            "time_seconds": distill_bench["time_seconds"],
            "memory_mb": distill_bench.get("memory_mb"),
        },
    ]
    print_benchmark_table(results)

    # Summary
    param_reduction = (1 - student_params / teacher_params) * 100
    speed_ratio = (
        distill_bench["time_seconds"] / teacher_bench["time_seconds"] * 100
        if teacher_bench["time_seconds"] > 0
        else 0
    )
    logger.info(
        f"Knowledge distillation produces a student model with "
        f"{param_reduction:.0f}% fewer parameters, "
        f"{speed_ratio:.0f}% of teacher's inference time, "
        f"and {distill_acc * 100:.1f}% accuracy (teacher: {teacher_acc * 100:.1f}%)"
    )

    print("\n" + "=" * 60)
    print("  Tutorial Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
