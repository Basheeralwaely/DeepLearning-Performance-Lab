"""
WAN 2.1 Diffusion Transformer (DiT) — Built from Source

This tutorial reconstructs the WAN 2.1 video DiT architecture block-by-block,
following the actual implementation in diffsynth/models/wan_video_dit.py.

Key differences from a standard ViT (see Vit_tutorial.py / Vit_advanced_tutorial.py):
  1. 3D patch embedding via Conv3d — patches span (temporal, height, width)
  2. 3D Rotary Position Embeddings — separate frequency bands for frame/row/col
  3. Cross-attention with text (and optional image) conditioning
  4. adaLN-Zero with per-block LEARNED modulation parameters + timestep modulation
  5. Modulated output head that unpatchifies back to video shape
  6. GELU(tanh) MLP, not SwiGLU (WAN's actual choice — simpler, works well here)
  7. No CLS token — every patch token is projected to output channels

Architecture card (WAN 2.1 T2V 1.3B config):
  patch_size = [1, 2, 2]   dim = 1536        ffn_dim = 8960
  num_heads  = 12           num_layers = 30   freq_dim = 256
  text_dim   = 4096         out_dim = 16      in_dim = 16
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Step 1 — Sinusoidal Timestep Embedding
# ---------------------------------------------------------------------------
# The diffusion timestep is encoded as a 1D sinusoidal vector (same idea as
# positional encoding in "Attention Is All You Need", but applied to a scalar
# timestep instead of a token position). This gives the model a smooth,
# continuous representation of where it is in the denoising schedule.
#
# WAN uses float64 for the outer product to avoid precision loss at high
# frequencies, then casts back to the input dtype.

def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """
    Sinusoidal embedding for scalar positions (timesteps).
    position: (B,) tensor of timestep values.
    Returns:  (B, dim) embedding.
    """
    half = dim // 2
    # Frequency bands: 10000^(-2i/dim) for i in [0, half)
    inv_freq = torch.pow(
        10000,
        -torch.arange(half, dtype=torch.float64, device=position.device) / half
    )
    # Outer product: each position × each frequency
    sinusoid = torch.outer(position.to(torch.float64), inv_freq)  # (B, half)
    # Concatenate cos and sin — gives dim features per timestep
    emb = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return emb.to(position.dtype)


# ---------------------------------------------------------------------------
# Step 2 — 3D Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------
# Standard RoPE is 1D (for text sequences). For video we need 3D: one set of
# frequency bands for the frame index, one for the row index, one for the
# column index. The head dimension is split three ways:
#   dim_t = dim - 2*(dim//3)   (temporal — gets the remainder)
#   dim_h = dim//3             (height)
#   dim_w = dim//3             (width)
#
# Each axis gets its own set of precomputed complex rotation factors.
# At forward time we index into these based on the actual (f, h, w) grid
# size and concatenate them into a single rotation tensor.

def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    """Precompute 1D rotary frequencies as complex numbers."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    # e^(i*theta) — unit-magnitude complex numbers encoding each position
    return torch.polar(torch.ones_like(freqs), freqs)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    """
    Precompute 3D rotary frequencies for (temporal, height, width).
    Returns a tuple of three complex tensors, one per axis.
    """
    dim_t = dim - 2 * (dim // 3)  # temporal gets the remainder
    dim_h = dim // 3
    dim_w = dim // 3
    f_freqs = precompute_freqs_cis(dim_t, end, theta)
    h_freqs = precompute_freqs_cis(dim_h, end, theta)
    w_freqs = precompute_freqs_cis(dim_w, end, theta)
    return f_freqs, h_freqs, w_freqs


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int):
    """
    Apply rotary embeddings to a query or key tensor.
    x:     (B, S, num_heads * head_dim)  — packed heads
    freqs: (S, 1, head_dim//2) complex   — precomputed rotations
    Returns same shape as x.
    """
    B, S, _ = x.shape
    head_dim = x.shape[-1] // num_heads
    # Reshape to (B, S, H, D) then view as complex pairs
    x = x.view(B, S, num_heads, head_dim)
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(B, S, num_heads, -1, 2)
    )
    # Multiply by rotation factors and convert back to real
    x_out = torch.view_as_real(x_complex * freqs).flatten(2)
    return x_out.to(x.dtype)


# ---------------------------------------------------------------------------
# Step 3 — RMSNorm
# ---------------------------------------------------------------------------
# Faster than LayerNorm: no mean subtraction, just scale by 1/RMS.
# Used for QK-normalization in attention (prevents logit magnitude blow-ups
# during bf16 training). The float() upcast is important for numerical
# stability under FSDP + bf16 mixed precision.

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(dtype)


# ---------------------------------------------------------------------------
# Step 4 — Flash Attention Dispatch
# ---------------------------------------------------------------------------
# WAN tries Flash Attention 3, then FA2, then SageAttention, then falls back
# to PyTorch's scaled_dot_product_attention. For this tutorial we use the
# PyTorch SDPA path (which itself dispatches to FlashAttention/memory-efficient
# kernels when available).

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    num_heads: int) -> torch.Tensor:
    """
    q: (B, Sq, num_heads * head_dim), k/v: (B, Skv, num_heads * head_dim).
    Sq and Skv may differ (cross-attention).
    Returns: (B, Sq, num_heads * head_dim).
    """
    B, Sq, D = q.shape
    Skv = k.shape[1]
    head_dim = D // num_heads
    # Reshape to (B, H, S, D) for SDPA
    q = q.view(B, Sq, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, Skv, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, Skv, num_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    # Back to (B, Sq, H*D)
    return out.transpose(1, 2).reshape(B, Sq, D)


# ---------------------------------------------------------------------------
# Step 5 — Self-Attention with QK-Norm + RoPE
# ---------------------------------------------------------------------------
# This is the exact WAN self-attention:
#   1. Project x -> q, k, v (separate linears, WITH bias — WAN keeps bias here)
#   2. RMSNorm on q and k independently (QK-norm)
#   3. Apply 3D RoPE to q and k (not v)
#   4. Flash attention
#   5. Output projection
#
# Note: WAN uses full-dimension RMSNorm on q/k (norm over all heads packed),
# which differs from the per-head norm in Vit_advanced_tutorial.py.

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # QK-normalization: prevents attention logit blow-ups in bf16
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        # Rotary embeddings applied to Q and K (not V)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        return self.o(flash_attention(q, k, v, self.num_heads))


# ---------------------------------------------------------------------------
# Step 6 — Cross-Attention (Text + Optional Image Conditioning)
# ---------------------------------------------------------------------------
# Cross-attention lets the DiT attend to text embeddings (and optionally
# CLIP image embeddings). Q comes from the latent tokens, K/V from context.
#
# When has_image_input=True, the context is split:
#   - First 257 tokens = CLIP image features (1 CLS + 256 patch tokens)
#   - Remaining tokens = text features
# Two separate attention paths are computed and added together.

class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6,
                 has_image_input: bool = False):
        super().__init__()
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.has_image_input = has_image_input
        if has_image_input:
            # Separate K/V projections for image features
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.has_image_input:
            img_ctx = context[:, :257]   # CLIP image tokens
            text_ctx = context[:, 257:]  # text tokens
        else:
            text_ctx = context

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(text_ctx))
        v = self.v(text_ctx)
        out = flash_attention(q, k, v, self.num_heads)

        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img_ctx))
            v_img = self.v_img(img_ctx)
            out = out + flash_attention(q, k_img, v_img, self.num_heads)

        return self.o(out)


# ---------------------------------------------------------------------------
# Step 7 — adaLN-Zero Modulation
# ---------------------------------------------------------------------------
# WAN's adaLN-Zero is slightly different from the standard DiT paper:
#   - Each block has a LEARNED modulation parameter (nn.Parameter) that is
#     ADDED to the timestep modulation before chunking into 6 signals.
#   - This means each block learns a baseline modulation offset, and the
#     timestep signal shifts it.
#   - Initialized with small random values (randn / sqrt(dim)), not zeros.
#
# The modulate() function: x * (1 + scale) + shift

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    """Apply adaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# Step 8 — DiT Block
# ---------------------------------------------------------------------------
# The core transformer block in WAN. Three sub-layers:
#   1. Self-attention  (adaLN-modulated norm → self-attn → gated residual)
#   2. Cross-attention (standard LayerNorm → cross-attn → add residual)
#   3. FFN             (adaLN-modulated norm → GELU MLP → gated residual)
#
# Key details:
#   - norm1, norm2: elementwise_affine=False (adaLN provides scale/shift)
#   - norm3: standard LayerNorm WITH affine (cross-attn uses fixed norm)
#   - gate = learned modulation signal that scales the residual
#   - self.modulation: per-block learned parameter added to timestep signal

class DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int,
                 has_image_input: bool = False, eps: float = 1e-6):
        super().__init__()

        # Self-attention with QK-norm + RoPE
        self.self_attn = SelfAttention(dim, num_heads, eps)

        # Cross-attention with text/image conditioning
        self.cross_attn = CrossAttention(dim, num_heads, eps,
                                         has_image_input=has_image_input)

        # adaLN norms (no affine — modulation supplies scale/shift)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        # Cross-attn norm (standard, with affine)
        self.norm3 = nn.LayerNorm(dim, eps=eps)

        # FFN: GELU(tanh) MLP — WAN uses this, not SwiGLU
        # Expansion ratio: ffn_dim/dim ≈ 5.8x for the 1.3B model
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # Per-block learned modulation parameter — added to timestep signal
        # before chunking into (shift, scale, gate) × 2 sub-layers
        # Initialized with small random values, NOT zeros
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                t_mod: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, S, dim)     — latent patch tokens
        context: (B, T, dim)     — text (+ image) embeddings
        t_mod:   (B, 6, dim)     — timestep modulation signal
        freqs:   (S, 1, dim//2)  — 3D RoPE complex rotation factors
        """
        # Combine per-block learned modulation with timestep modulation,
        # then chunk into 6 signals
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        # --- Self-attention sub-layer ---
        # adaLN: modulate(norm(x), shift, scale)
        normed = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.self_attn(normed, freqs)

        # --- Cross-attention sub-layer ---
        # Standard norm (no adaLN), simple residual (no gate)
        x = x + self.cross_attn(self.norm3(x), context)

        # --- FFN sub-layer ---
        normed = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.ffn(normed)

        return x


# ---------------------------------------------------------------------------
# Step 9 — Modulated Output Head
# ---------------------------------------------------------------------------
# The output head also uses adaLN-Zero modulation (shift + scale from the
# timestep embedding). It projects each patch token to out_dim * prod(patch_size)
# values, which are then reshaped to reconstruct the video tensor.
#
# Like the DiT blocks, the head has its own learned modulation parameter.

class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int],
                 eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        # 2 modulation signals: shift + scale
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, S, dim)
        t_emb: (B, dim) — timestep embedding (before the 6x projection)
        """
        shift, scale = (
            self.modulation.to(dtype=t_emb.dtype, device=t_emb.device) + t_emb
        ).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


# ---------------------------------------------------------------------------
# Step 10 — CLIP Image Embedding Projector
# ---------------------------------------------------------------------------
# When doing image-to-video, CLIP features (1280-dim) are projected to the
# model dimension and prepended to the text context. Optional positional
# embedding for the CLIP tokens.

class ImageEmbeddingMLP(nn.Module):
    def __init__(self, clip_dim: int = 1280, out_dim: int = 1536):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, clip_dim),
            nn.GELU(),
            nn.Linear(clip_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ---------------------------------------------------------------------------
# Step 11 — Full WAN Video DiT
# ---------------------------------------------------------------------------
# Putting it all together. The forward pass:
#
#   Input video latent (B, C, F, H, W)
#     ↓ Conv3d patchify → (B, S, dim)   where S = f*h*w
#     ↓ + optional image latent concatenation
#
#   Timestep (B,)
#     ↓ sinusoidal_embedding_1d → (B, freq_dim)
#     ↓ time_embedding MLP      → (B, dim)
#     ↓ time_projection          → (B, 6*dim) → (B, 6, dim)
#
#   Text context (B, T, text_dim)
#     ↓ text_embedding MLP → (B, T, dim)
#     ↓ + optional CLIP image embedding prepended
#
#   3D RoPE: precomputed once, indexed by (f, h, w) grid each forward pass
#
#   For each DiT block:
#     x = block(x, context, t_mod, freqs)
#
#   Output head (modulated) → (B, S, out_dim * prod(patch_size))
#     ↓ unpatchify → (B, out_dim, F, H, W)

class WanDiT(nn.Module):
    def __init__(
        self,
        dim: int = 1536,
        in_dim: int = 16,
        ffn_dim: int = 8960,
        out_dim: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_heads: int = 12,
        num_layers: int = 30,
        has_image_input: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.has_image_input = has_image_input

        # --- Embeddings ---
        # 3D patch embedding: Conv3d with kernel=stride=patch_size
        # For patch_size=[1,2,2]: no temporal downsampling, 2x spatial downsampling
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        # Text conditioning: project from text encoder dim to model dim
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )

        # Timestep conditioning pipeline:
        #   sinusoidal(freq_dim) → MLP(freq_dim → dim) → SiLU + Linear(dim → 6*dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)
        )

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, ffn_dim, has_image_input, eps)
            for _ in range(num_layers)
        ])

        # --- Output ---
        self.head = Head(dim, out_dim, patch_size, eps)

        # --- 3D RoPE (precomputed once, shared across all blocks) ---
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        # --- Optional image-to-video components ---
        if has_image_input:
            self.img_emb = ImageEmbeddingMLP(1280, dim)

    def patchify(self, x: torch.Tensor):
        """
        3D patch embedding + reshape to sequence.
        Input:  (B, C, F, H, W) video latent
        Output: (B, f*h*w, dim), grid_size=(f, h, w)
        """
        x = self.patch_embedding(x)  # (B, dim, f, h, w)
        B, C, f, h, w = x.shape
        x = x.reshape(B, C, f * h * w).transpose(1, 2)  # (B, f*h*w, dim)
        return x, (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]):
        """
        Reshape sequence back to video tensor.
        Input:  (B, f*h*w, out_dim * patch_t * patch_h * patch_w)
        Output: (B, out_dim, F, H, W)
        """
        f, h, w = grid_size
        pt, ph, pw = self.patch_size
        # Reshape: (B, f*h*w, pt*ph*pw*C) → (B, C, f*pt, h*ph, w*pw)
        B = x.shape[0]
        C = x.shape[-1] // (pt * ph * pw)
        x = x.view(B, f, h, w, pt, ph, pw, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # (B, C, f, pt, h, ph, w, pw)
        x = x.reshape(B, C, f * pt, h * ph, w * pw)
        return x

    def build_rope_freqs(self, f: int, h: int, w: int,
                         device: torch.device) -> torch.Tensor:
        """
        Build 3D RoPE rotation factors for a (f, h, w) grid.
        Each spatial position gets rotation from its (frame, row, col) coordinate.
        The three axis frequencies are concatenated along the last dim.
        Returns: (f*h*w, 1, head_dim//2) complex tensor.
        """
        freqs = torch.cat([
            # Temporal: index by frame, expand over (h, w)
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            # Height: index by row, expand over (f, w)
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            # Width: index by col, expand over (f, h)
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1)
        # Flatten spatial dims: (f*h*w, head_dim//2) → add head broadcast dim
        return freqs.reshape(f * h * w, 1, -1).to(device)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:            (B, in_dim, F, H, W)    — noisy video latent
        timestep:     (B,)                     — diffusion timestep
        context:      (B, T, text_dim)         — text encoder output
        clip_feature: (B, 257, 1280)           — CLIP image features (I2V only)
        y:            (B, C_ref, F, H, W)      — reference image latent (I2V only)
        Returns:      (B, out_dim, F, H, W)    — predicted noise / v-prediction
        """
        # --- Timestep embedding ---
        # Scalar timestep → sinusoidal → MLP → 6-way modulation signal
        t_emb = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype)
        )
        # (B, dim) → (B, 6*dim) → (B, 6, dim) — one set of modulation params
        t_mod = self.time_projection(t_emb).unflatten(1, (6, self.dim))

        # --- Text embedding ---
        context = self.text_embedding(context)

        # --- Optional image conditioning (I2V) ---
        if self.has_image_input and y is not None:
            # Concatenate reference latent with noisy latent along channel dim
            x = torch.cat([x, y], dim=1)
            if clip_feature is not None:
                clip_emb = self.img_emb(clip_feature)
                context = torch.cat([clip_emb, context], dim=1)

        # --- Patchify ---
        x, (f, h, w) = self.patchify(x)

        # --- Build 3D RoPE for this grid ---
        freqs = self.build_rope_freqs(f, h, w, x.device)

        # --- Transformer blocks ---
        for block in self.blocks:
            x = block(x, context, t_mod, freqs)

        # --- Output head (modulated) ---
        x = self.head(x, t_emb)

        # --- Unpatchify back to video shape ---
        x = self.unpatchify(x, (f, h, w))
        return x


# ---------------------------------------------------------------------------
# Demo: Instantiate and run a forward pass with dummy data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAN 2.1 DiT Tutorial")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of DiT blocks (default 4 for demo; real model uses 30)")
    parser.add_argument("--dim", type=int, default=768,
                        help="Model dimension (default 768 for demo; real model uses 1536)")
    parser.add_argument("--num_heads", type=int, default=6,
                        help="Attention heads (head_dim must be divisible by 6 for 3D RoPE; "
                             "real model: dim=1536, heads=12 → head_dim=128)")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--frames", type=int, default=8,
                        help="Number of video frames")
    parser.add_argument("--height", type=int, default=32,
                        help="Spatial height of the latent")
    parser.add_argument("--width", type=int, default=32,
                        help="Spatial width of the latent")
    args = parser.parse_args()

    device = args.device
    B = args.batch_size
    in_dim = 16      # VAE latent channels
    out_dim = 16
    text_dim = 4096  # T5 text encoder
    freq_dim = 256

    print("=" * 70)
    print("WAN 2.1 DiT — Tutorial Forward Pass")
    print("=" * 70)

    # Build model (small config for demo)
    ffn_dim = int(args.dim * 8960 / 1536)  # maintain the ~5.8x ratio
    model = WanDiT(
        dim=args.dim,
        in_dim=in_dim,
        ffn_dim=ffn_dim,
        out_dim=out_dim,
        text_dim=text_dim,
        freq_dim=freq_dim,
        patch_size=(1, 2, 2),
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        has_image_input=False,
        eps=1e-6,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel config:")
    print(f"  dim={args.dim}, ffn_dim={ffn_dim}, heads={args.num_heads}, layers={args.num_layers}")
    print(f"  patch_size=(1, 2, 2), in_dim={in_dim}, out_dim={out_dim}")
    print(f"  Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Dummy inputs
    F_frames, H, W = args.frames, args.height, args.width
    x = torch.randn(B, in_dim, F_frames, H, W, device=device)
    timestep = torch.randint(0, 1000, (B,), device=device)
    context = torch.randn(B, 77, text_dim, device=device)  # 77 text tokens

    print(f"\nInput shapes:")
    print(f"  Video latent:  {tuple(x.shape)}")
    print(f"  Timestep:      {tuple(timestep.shape)}")
    print(f"  Text context:  {tuple(context.shape)}")

    # Grid size after patchifying
    f = F_frames // 1  # patch_size[0] = 1
    h = H // 2         # patch_size[1] = 2
    w = W // 2         # patch_size[2] = 2
    seq_len = f * h * w
    print(f"\n  Grid after patchify: ({f}, {h}, {w}) → sequence length = {seq_len}")

    # Forward pass
    with torch.no_grad():
        output = model(x, timestep, context)

    print(f"\nOutput shape:    {tuple(output.shape)}")
    print(f"  Expected:      ({B}, {out_dim}, {F_frames}, {H}, {W})")
    assert output.shape == (B, out_dim, F_frames, H, W), "Shape mismatch!"
    print("\nForward pass successful.")

    # Show block structure
    print(f"\n{'=' * 70}")
    print("Block structure (one DiTBlock):")
    print("=" * 70)
    block = model.blocks[0]
    print(f"  self_attn:   SelfAttention(dim={args.dim}, heads={args.num_heads}, QK-norm=RMSNorm, RoPE=3D)")
    print(f"  cross_attn:  CrossAttention(dim={args.dim}, heads={args.num_heads}, QK-norm=RMSNorm)")
    print(f"  norm1/norm2: LayerNorm(elementwise_affine=False)  ← adaLN supplies scale/shift")
    print(f"  norm3:       LayerNorm(elementwise_affine=True)   ← standard for cross-attn")
    print(f"  ffn:         Linear({args.dim}→{ffn_dim}) → GELU(tanh) → Linear({ffn_dim}→{args.dim})")
    print(f"  modulation:  nn.Parameter(1, 6, {args.dim})       ← per-block learned offset")

    print(f"\n{'=' * 70}")
    print("Key differences from standard ViT (Vit_tutorial.py):")
    print("=" * 70)
    print("  1. 3D Conv3d patch embedding instead of 2D Conv2d")
    print("  2. 3D RoPE (temporal + height + width) instead of learned pos_embed")
    print("  3. Cross-attention sub-layer for text conditioning")
    print("  4. adaLN-Zero with per-block LEARNED modulation + timestep signal")
    print("  5. Gated residuals (gate * residual) instead of simple addition")
    print("  6. Modulated output head (not a plain linear)")
    print("  7. No CLS token — every patch is projected to output channels")
    print("  8. Unpatchify reconstructs (B, C, F, H, W) video tensor")
    print("\nThis architecture is specialized for video diffusion, with strong temporal and spatial inductive biases, and powerful conditioning mechanisms for text and image inputs.")
