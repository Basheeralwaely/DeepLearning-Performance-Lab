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
        if reduced_size == 0:
            raise ValueError(
                f"input_size={input_size} is too small for 3 MaxPool2d(2) layers. "
                f"Minimum input_size is 8."
            )
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


class SimpleViT(nn.Module):
    """A minimal Vision Transformer for mixed precision tutorial demos.

    Designed for Transformer Engine FP8 compatibility: all Linear layer
    dimensions are divisible by 16.

    Args:
        image_size: Input image spatial dimension (square). Default: 32.
        patch_size: Size of each patch (must divide image_size). Default: 4.
        num_classes: Number of output classes. Default: 10.
        dim: Hidden/embedding dimension (divisible by 16). Default: 256.
        depth: Number of transformer encoder layers. Default: 4.
        heads: Number of attention heads. Default: 8.
        mlp_dim: FFN hidden dimension (divisible by 16). Default: 512.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        mlp_dim: int = 512,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        assert dim % heads == 0, (
            f"dim ({dim}) must be divisible by heads ({heads})"
        )

        num_patches = (image_size // patch_size) ** 2

        # Patch embedding: conv2d acts as linear projection of flattened patches
        self.patch_embed = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size
        )

        # Learnable class token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Transformer encoder with Pre-LN (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        # Classification head applied to class token
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            x: Input tensor of shape (batch, 3, image_size, image_size).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        # Patch embedding: (B, 3, H, W) -> (B, dim, H/P, W/P) -> (B, num_patches, dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Prepend class token
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer encoder
        x = self.transformer(x)

        # Extract class token and classify
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)
        return logits


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
