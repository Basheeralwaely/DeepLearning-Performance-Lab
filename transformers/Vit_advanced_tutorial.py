import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    # image dataset size = B, 3, 224, 244 
    # number of patches = (224/16) * (224/16) = 14 * 14 = 196
    # transformer input = B, 197, 768 (if embedding dimension is 768)

    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.proj(x)           # (B, embed_dim, 14, 14)
        x = x.flatten(2)           # (B, embed_dim, 196)
        x = x.transpose(1, 2)      # (B, 196, embed_dim)
        return x
    
### Advance Trasformer looks like this with adaLN-Zero conditioning, which is critical for stable training of diffusion models.
"""Walking through the meaningful changes:
Attention — Flash attention via SDPA. The original builds the attention matrix explicitly with q @ k.transpose(-2, -1), which materializes an (N, N) tensor in memory. For video DiT with, say, 16 frames × 60×90 latent tokens, that matrix is enormous. F.scaled_dot_product_attention dispatches to FlashAttention-2/3 or memory-efficient kernels automatically, avoiding that materialization and running 2–4× faster in bf16. It also handles scaling, masking, and dropout internally, so the code is simpler.
QK-normalization. Applying RMSNorm to Q and K independently before the dot product is the single most important stability trick for large-scale training. Without it, some attention logits drift to extreme magnitudes during training and cause bf16 overflow. Every modern video DiT (Wan2.1, CogVideoX, Sora-style models) uses this.
Separate Q/K/V projections with GQA support. I split the fused qkv into three linears so you can optionally reduce the number of KV heads. With n_kv_heads=n_heads it's standard multi-head attention; with n_kv_heads < n_heads it's grouped-query attention, which cuts KV cache memory proportionally. For training a video model from scratch this matters less than for inference, but it's almost free to keep the option.
RoPE plumbing. The freqs_cis argument is optional — pass None and you get a normal attention layer, pass a precomputed rotation tensor and you get rotary embeddings. For your video work you'd compute 3D RoPE (separate frequency bands for temporal / height / width axes) once in the model and pass it down to every block. I kept the RoPE helper minimal here; the full 3D version just concatenates three 1D RoPE tensors along the head dimension.
Bias=False on all linears. LLaMA-style convention. Biases add parameters and memory bandwidth without measurably helping performance. Safe to omit everywhere inside transformer blocks.
MLP → SwiGLU. Three linears instead of two, with a gating multiplication (silu(w1(x)) * w2(x)). The key subtlety is the hidden_dim adjustment: a GeLU MLP with hidden_dim = 4D has ~8D² params, while a SwiGLU naively at hidden_dim = 4D would have ~12D². Scaling hidden_dim by 2/3 brings it back to ~8D² params and roughly matched FLOPs, then rounding to a multiple of 256 keeps GEMM shapes hardware-friendly (tensor cores prefer aligned dimensions).
How this composes with the DiTBlock from before. The Attention(dim, n_heads, qk_norm=True) call in that block now hits this implementation. If you want to wire up RoPE, change the block's forward signature to forward(self, x, c, freqs_cis) and thread freqs_cis through to the attention call. Same for attention masks if you need packed sequences or causal masking for any part of the model.
One thing worth flagging: if you're using FSDP with bfloat16 mixed precision, the RMSNorm upcast to float32 inside its forward is important — keep it. Some implementations skip the upcast and it bites you at scale.
"""
class RMSNorm(nn.Module):
    """RMSNorm — faster than LayerNorm, used in LLaMA/Wan2.1/most modern DiTs."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # rsqrt over the last dim, cast to float for numerical stability then back
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


def apply_rope(x, freqs_cis):
    """
    Apply rotary position embeddings.
    x: (B, H, N, D_head), freqs_cis: (N, D_head/2) complex tensor.
    """
    # Pair up the last dim into complex numbers, multiply by precomputed rotations
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
    return x_out.type_as(x)


class Attention(nn.Module):
    """
    Multi-head self-attention with:
      - Fused QKV projection (standard)
      - QK-normalization (prevents logit blow-ups in long-context / bf16)
      - Rotary position embeddings (optional — pass freqs_cis to enable)
      - Flash attention via F.scaled_dot_product_attention
      - Optional grouped-query attention (n_kv_heads < n_heads saves KV memory)
    """
    def __init__(self, dim, n_heads=8, n_kv_heads=None, qk_norm=True, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        # Separate Q vs KV projections so we can have fewer KV heads (GQA/MQA).
        # When n_kv_heads == n_heads this is equivalent to standard MHA with a fused qkv.
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # QK-norm: normalize Q and K before the dot product.
        # Critical for training stability at scale with bf16.
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x, freqs_cis=None, attn_mask=None):
        B, N, _ = x.shape

        q = self.wq(x).view(B, N, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, N, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, N, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # (B, N, H, D) -> (B, H, N, D) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Rotary position embeddings applied to Q and K (not V)
        if freqs_cis is not None:
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)

        # Repeat KV heads to match Q heads if using GQA
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Flash attention — uses FA2/FA3 kernels when available, falls back otherwise.
        # Handles scaling, softmax, dropout, and masking internally.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.proj(out)


class SwiGLU(nn.Module):
    """
    SwiGLU MLP — used in LLaMA, PaLM, Wan2.1, and most modern transformers.
    Replaces the standard GeLU MLP. Slightly more params per layer but
    consistently better loss curves at the same FLOP budget.
    """
    def __init__(self, dim, hidden_dim, multiple_of=256, dropout=0.0):
        super().__init__()
        # Scale hidden_dim by 2/3 to keep FLOPs comparable to a 2-linear GeLU MLP,
        # then round up to a multiple of 256 for hardware-friendly shapes.
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # value
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # out
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))
    
def modulate(x, shift, scale):
    # x: (B, N, D), shift/scale: (B, D) -> broadcast over tokens
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    Pre-norm transformer block with adaLN-Zero conditioning.
    Conditioning vector `c` (e.g. timestep + text pooled embedding) produces
    6 modulation signals: shift/scale/gate for attention and MLP sub-layers.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, cond_dim=None):
        super().__init__()
        cond_dim = cond_dim or dim

        # elementwise_affine=False because adaLN supplies scale/shift
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, n_heads, qk_norm=True)  # RMSNorm on Q/K

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))  # ideally SwiGLU

        # adaLN-Zero: projects conditioning -> 6 * dim modulation params
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True),
        )
        # Zero-init so the block starts as an identity function.
        # This is the "Zero" in adaLN-Zero — critical for stable diffusion training.
        nn.init.zeros_(self.ada_mod[1].weight)
        nn.init.zeros_(self.ada_mod[1].bias)

    def forward(self, x, c):
        # x: (B, N, D) tokens; c: (B, cond_dim) conditioning vector
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            self.ada_mod(c).chunk(6, dim=-1)
        )

        # Attention sub-layer: modulate -> attn -> gate -> residual
        x = x + gate_a.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_a, scale_a)
        )
        # MLP sub-layer: same pattern
        x = x + gate_m.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_m, scale_m)
        )
        return x


### Step 5 — Advanced Transformer Block (no conditioning — pure ViT path)
"""
This block replaces the basic TransformerBlock with all the modern tricks:
  - RMSNorm instead of LayerNorm (faster, no mean computation)
  - Advanced Attention with QK-norm, Flash Attention, optional GQA & RoPE
  - SwiGLU MLP instead of GeLU MLP (better loss curves at same FLOPs)
This is the architecture used in EVA-02, InternVL, SigLIP-SO400M, and
similar state-of-the-art vision encoders. The DiTBlock above adds adaLN-Zero
conditioning on top of this for diffusion models.
"""
class AdvancedTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, mlp_ratio=4.0,
                 qk_norm=True, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads=n_kv_heads,
                              qk_norm=qk_norm, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x, freqs_cis=None):
        x = x + self.attn(self.norm1(x), freqs_cis=freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x


### Step 6 — 2D Rotary Position Embeddings
"""
Standard RoPE is 1D (for text). For images we need 2D: one set of frequency
bands for the row index and another for the column index. We precompute these
once and pass them into every attention layer.

For video (Wan2.1, CogVideoX) you'd extend this to 3D by adding a temporal
axis — the principle is the same: concatenate independent 1D RoPE tensors
along the head dimension.
"""
def precompute_freqs_cis_2d(dim, grid_size, theta=10000.0):
    """
    Precompute 2D rotary embeddings for a grid_size x grid_size spatial grid.
    Returns: (grid_size*grid_size, dim//2) complex tensor.
    Each row position gets rotation from its (row, col) coordinate.
    """
    half = dim // 2
    # Split head dim in half: first half encodes row, second half encodes col
    freqs_per_axis = half // 2
    freqs = 1.0 / (theta ** (torch.arange(0, freqs_per_axis, dtype=torch.float32) / freqs_per_axis))

    coords = torch.arange(grid_size, dtype=torch.float32)
    # row freqs: each row index × each frequency
    row_freqs = torch.outer(coords, freqs)  # (grid_size, freqs_per_axis)
    # col freqs: same
    col_freqs = torch.outer(coords, freqs)  # (grid_size, freqs_per_axis)

    # Expand to full grid: (grid_size, grid_size, freqs_per_axis) for each axis
    row_freqs = row_freqs[:, None, :].expand(-1, grid_size, -1)
    col_freqs = col_freqs[None, :, :].expand(grid_size, -1, -1)

    # Concatenate row and col frequencies, flatten spatial dims
    freqs_2d = torch.cat([row_freqs, col_freqs], dim=-1)  # (H, W, half)
    freqs_2d = freqs_2d.reshape(grid_size * grid_size, half)

    # Convert to complex for rotation: e^(i*theta)
    freqs_cis = torch.polar(torch.ones_like(freqs_2d), freqs_2d)
    return freqs_cis


### Step 7 — Full Advanced Vision Transformer
"""
Putting it all together. Compared to the basic ViT in Vit_tutorial.py:
  1. Learnable CLS token + 2D RoPE (instead of fixed absolute position embeddings)
     — RoPE generalizes better to different resolutions at inference time.
     — We still keep a small learnable pos_embed as a fallback / warmup signal.
  2. RMSNorm everywhere instead of LayerNorm.
  3. SwiGLU MLP instead of GeLU MLP.
  4. QK-normalization in every attention layer.
  5. Flash Attention (via F.scaled_dot_product_attention).
  6. Optional GQA for memory-efficient inference.
  7. No bias in any linear layer inside transformer blocks.

These are the exact design choices used in recent large-scale vision models:
  - EVA-02 (BAAI): RoPE + SwiGLU + RMSNorm
  - InternVL-2.5 (Shanghai AI Lab): same recipe + GQA
  - SigLIP-SO400M (Google): QK-norm + no bias
  - Wan2.1 (Alibaba): all of the above + adaLN-Zero for diffusion
"""
class AdvancedVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_ch=3,
        num_classes=10,
        embed_dim=768,
        depth=12,
        n_heads=12,
        n_kv_heads=None,
        mlp_ratio=4.0,
        qk_norm=True,
        dropout=0.0,
        use_rope=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_rope = use_rope
        self.grid_size = img_size // patch_size

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_ch, embed_dim
        )
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable absolute position embedding — kept even with RoPE.
        # RoPE handles relative position; this gives a small absolute signal
        # and helps the CLS token (which has no spatial position for RoPE).
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            AdvancedTransformerBlock(
                embed_dim, n_heads, n_kv_heads=n_kv_heads,
                mlp_ratio=mlp_ratio, qk_norm=qk_norm, dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Precompute 2D RoPE frequencies for the spatial grid
        if use_rope:
            head_dim = embed_dim // n_heads
            freqs_cis = precompute_freqs_cis_2d(head_dim, self.grid_size)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        self._init_weights()

    def _init_weights(self):
        # Truncated normal init for embeddings — standard ViT recipe
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Zero-init the classification head for stable early training
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)

        x = x + self.pos_embed

        # Build RoPE freqs for this forward pass.
        # CLS token gets zero rotation (no spatial position).
        freqs_cis = None
        if self.use_rope:
            N_patches = x.shape[1] - 1  # exclude CLS
            # Prepend a zero-rotation entry for the CLS token
            cls_freq = torch.ones(1, self.freqs_cis.shape[1],
                                  dtype=self.freqs_cis.dtype,
                                  device=self.freqs_cis.device)
            freqs_cis = torch.cat([cls_freq, self.freqs_cis[:N_patches]], dim=0)

        for block in self.blocks:
            x = block(x, freqs_cis=freqs_cis)

        x = self.norm(x)

        cls_output = x[:, 0]
        return self.head(cls_output)


if __name__ == "__main__":
    options = argparse.ArgumentParser()
    options.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    options.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    options.add_argument("--data_dir", type=str, default="./data", help="Directory for storing the dataset")
    options.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    options.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    options.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    options.add_argument("--patch_size", type=int, default=16, help="Size of each patch")
    args = options.parse_args()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    trainset = datasets.CIFAR10(
        root='/home/basheer/Signapse/Codes/DeepLearning-Performance-Lab/data',
        train=True,
        download=True,
        transform=transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True
    )

    model = AdvancedVisionTransformer(num_classes=10).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(args.num_epochs):
        for images, labels in trainloader:
            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss {loss.item():.4f}")
