"""Simple model factories for DeepLearning Performance Lab tutorials.

Provides lightweight CNN and MLP architectures that tutorials use as subjects
for demonstrating performance techniques. These models are intentionally simple
so the focus remains on the optimization trick, not the model architecture.
"""

from typing import Optional

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A simple 3-conv-layer + 2-FC-layer CNN for tutorial demonstrations.

    Args:
        input_channels: Number of input channels (default: 3 for RGB).
        num_classes: Number of output classes (default: 10).
        input_size: Spatial dimension of square input images (default: 32).
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        input_size: int = 32,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Calculate flattened size after 3 rounds of MaxPool2d(2)
        reduced_size = input_size // 8
        flat_dim = 128 * reduced_size * reduced_size

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleMLP(nn.Module):
    """A simple 3-layer MLP with ReLU activations for tutorial demonstrations.

    Args:
        input_dim: Dimensionality of input features (default: 784).
        hidden_dim: Hidden layer size (default: 256).
        output_dim: Number of output classes (default: 10).
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Logits tensor of shape (batch, output_dim).
        """
        return self.layers(x)


def get_sample_batch(
    batch_size: int = 32,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random sample batch for testing and benchmarking.

    Args:
        batch_size: Number of samples in the batch.
        channels: Number of input channels.
        height: Spatial height of input images.
        width: Spatial width of input images.
        num_classes: Number of output classes; labels are drawn from [0, num_classes).
        device: Target device (defaults to CPU if None).

    Returns:
        A tuple of (inputs, labels) where inputs is a random float tensor
        and labels is a random integer tensor of class indices.
    """
    inputs = torch.randn(batch_size, channels, height, width, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    return inputs, labels
