"""
DiT vs UNet: Large-Batch Diffusion Throughput Comparison
========================================================

Why this tutorial exists
------------------------
Diffusion models historically used UNets (Stable Diffusion 1.x / 2.x). The
current generation (DiT, Stable Diffusion 3, Flux) replaced the UNet with a
Diffusion *Transformer* (DiT). The architectural shift is partly a research
choice -- transformers scale better with data and compute -- but it is also
a *performance* choice. Transformer blocks are almost entirely matmul-bound
(LayerNorm -> QKV -> SDPA -> MLP), feeding tensor cores well at large batch.
UNets spend meaningful time in memory-bound ops (small-channel convs,
GroupNorm, transposed-conv upsamplers) that scale less gracefully.

This tutorial makes that shift observable on real hardware: we build two
small models with *matched* parameter counts (within 5%), wrap each in a
minimal one-step diffusion training workload (sample noise -> sample
timestep -> predict noise -> MSE loss -> optimizer.step()), and sweep batch
size to compare throughput (samples/sec) and peak GPU memory.

Why synthetic latents (no VAE, no real images)
----------------------------------------------
Modern diffusion runs in latent space: a VAE encodes a 256x256 RGB image
(196,608 values) down to a 32x32x4 latent (4,096 values), roughly 48x
smaller. Attention cost is O(N^2) in token count, so latent space is
*orders of magnitude* cheaper than pixel space at this spatial scale.

For a *performance* tutorial we don't need a real VAE -- we just need
tensors of the right shape. We feed both models synthetic
``[B, 4, 32, 32]`` latents from ``torch.randn`` and skip the VAE entirely.
This keeps the tutorial standalone (no ``diffusers`` install) and isolates
the architectural performance comparison from VAE encoding overhead.

What this tutorial measures
---------------------------
1. Param count for SimpleDiT and SimpleUNet (asserted matched within 5%).
2. For each batch size in [8, 16, 32, 64, 128]:
   - Wall-clock time over NUM_ITERATIONS one-step diffusion training steps.
   - Peak GPU memory.
   - Derived throughput = (B * NUM_ITERATIONS) / time_seconds (samples/sec).
3. OOMs at large batch are caught and logged (not a crash) -- the failure
   itself is part of the perf story.

Constraints
-----------
- No diffusers, no VAE, no real images, no full sampling loop.
- Exactly one optimizer step per iteration (D-08).
- Models are inline in this file (D-09 + RESEARCH.md Open Question 1).
- Batch sweep uses synthetic latents shaped [B, 4, 32, 32] (D-11).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    setup_logging,
    benchmark,
    print_benchmark_table,
    get_device,
    print_device_info,
    get_gpu_capability,
)

logger = setup_logging("dit_vs_unet")

# ----------------------------------------------------------------------------
# Constants -- locked per RESEARCH.md / D-12
# ----------------------------------------------------------------------------
IN_CH = 4
LATENT_HW = 32
T_STEPS = 1000
BATCHES = [8, 16, 32, 64, 128]
NUM_ITERATIONS = 30
WARMUP_ITERATIONS = 5
PARAM_MATCH_TOLERANCE = 0.05  # 5%


# ----------------------------------------------------------------------------
# Sinusoidal time embedding -- shared by both models
# ----------------------------------------------------------------------------
def sinusoidal_time_embed(t, dim):
    """Standard sinusoidal positional embedding of integer timesteps.

    Args:
        t: [B] int tensor of diffusion timesteps in [0, T_STEPS).
        dim: embedding dimension (must be even).

    Returns:
        [B, dim] float tensor.
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=t.device)
        * (torch.log(torch.tensor(10000.0)) / half)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ----------------------------------------------------------------------------
# SimpleDiT -- minimal Diffusion Transformer
# ----------------------------------------------------------------------------
# Architecture summary:
#   patchify [B,4,32,32] -> [B, N=256, D] tokens via Conv2d(stride=patch)
#   + learnable positional embedding
#   stack of DiT blocks (LayerNorm -> adaLN scale/shift from t_emb
#                        -> QKV linear -> SDPA -> MLP)
#   final LayerNorm + linear head -> unpatchify back to [B, 4, 32, 32]
class DiTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        # adaLN-style scale/shift conditioned on the time embedding.
        # Simplified to (scale, shift) -- the original DiT paper uses 6 values
        # for separate gates on attn / mlp; this keeps the tutorial readable.
        self.ada = nn.Linear(dim, 2 * dim)
        self.heads = heads

    def forward(self, x, t_emb):
        # x:     [B, N, D]
        # t_emb: [B, D]
        scale, shift = self.ada(t_emb).chunk(2, dim=-1)  # [B, D] each
        h = self.norm1(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        B, N, D = h.shape
        qkv = (
            self.qkv(h)
            .view(B, N, 3, self.heads, D // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # each [B, H, N, d_h]
        # Use SDPA -- this is fair (same algorithm both models would use in
        # practice) and fast (flash/efficient backends on modern GPUs).
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleDiT(nn.Module):
    def __init__(self, in_ch=4, patch=2, H=32, W=32, dim=384, depth=6, heads=6):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.num_patches = (H // patch) * (W // patch)  # 16*16 = 256 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        self.t_embed_dim = dim
        self.t_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        # Predict noise per patch -- output shape per token is patch*patch*in_ch.
        self.head = nn.Linear(dim, patch * patch * in_ch)
        self.patch, self.in_ch, self.H, self.W = patch, in_ch, H, W

    def forward(self, x, t):
        # x: [B, 4, 32, 32], t: [B] int timesteps
        B = x.size(0)
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        t_emb = self.t_mlp(sinusoidal_time_embed(t, self.t_embed_dim))
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = self.norm(h)
        h = self.head(h)  # [B, N, patch*patch*C]
        # Unpatchify back to [B, 4, 32, 32]
        P = self.patch
        h = h.view(B, self.H // P, self.W // P, P, P, self.in_ch)
        h = h.permute(0, 5, 1, 3, 2, 4).contiguous().view(
            B, self.in_ch, self.H, self.W
        )
        return h


# ----------------------------------------------------------------------------
# SimpleUNet -- minimal convolutional UNet with time conditioning
# ----------------------------------------------------------------------------
# Architecture summary:
#   in_conv -> ResBlock down1 -> pool(2x) -> ResBlock down2 -> pool(2x)
#   -> ResBlock mid
#   -> upsample(2x) -> concat skip h3 -> ResBlock dec2
#   -> upsample(2x) -> concat skip h1 -> ResBlock dec1
#   -> out_conv -> [B, 4, 32, 32]
class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch_in)
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, ch_out)
        self.norm2 = nn.GroupNorm(8, ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.skip = (
            nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """[B, 4, 32, 32] -> down 2x -> down 2x -> bottleneck -> mirror up -> [B, 4, 32, 32]."""

    def __init__(self, in_ch=4, base=192, t_dim=384):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.t_dim = t_dim

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down1 = ResBlock(base, base * 2, t_dim)
        self.pool1 = nn.Conv2d(base * 2, base * 2, 2, stride=2)
        self.down2 = ResBlock(base * 2, base * 4, t_dim)
        self.pool2 = nn.Conv2d(base * 4, base * 4, 2, stride=2)
        self.mid = ResBlock(base * 4, base * 4, t_dim)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 4, 2, stride=2)
        self.dec2 = ResBlock(base * 8, base * 2, t_dim)  # skip-concat doubles input ch
        self.up1 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.dec1 = ResBlock(base * 4, base, t_dim)  # skip-concat doubles input ch
        self.out_conv = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.t_mlp(sinusoidal_time_embed(t, self.t_dim))
        h0 = self.in_conv(x)                                # [B, base,    32, 32]
        h1 = self.down1(h0, t_emb)                          # [B, base*2,  32, 32]
        h2 = self.pool1(h1)                                 # [B, base*2,  16, 16]
        h3 = self.down2(h2, t_emb)                          # [B, base*4,  16, 16]
        h4 = self.pool2(h3)                                 # [B, base*4,   8,  8]
        m = self.mid(h4, t_emb)                             # [B, base*4,   8,  8]
        u2 = self.up2(m)                                    # [B, base*4,  16, 16]
        d2 = self.dec2(torch.cat([u2, h3], 1), t_emb)       # [B, base*2,  16, 16]
        u1 = self.up1(d2)                                   # [B, base*2,  32, 32]
        d1 = self.dec1(torch.cat([u1, h1], 1), t_emb)       # [B, base,    32, 32]
        return self.out_conv(d1)                            # [B, 4,       32, 32]


# ----------------------------------------------------------------------------
# Param count + matching assertion
# ----------------------------------------------------------------------------
def param_count(m):
    return sum(p.numel() for p in m.parameters())


def find_matching_unet(target_params, t_dim=384, tol=PARAM_MATCH_TOLERANCE):
    """Empirically tune SimpleUNet `base` until param count matches DiT within tol.

    Per RESEARCH.md Assumption A2 the default `base=192` should land near 20M
    params -- close to the default DiT. If the match is already within `tol`
    we just return the default. Otherwise we sweep `base` in steps of 16 to
    find the closest valid configuration. Sweep keeps `base` divisible by 8
    (GroupNorm constraint) and >= 32.
    """
    candidates = [192]
    # Walk outward from 192 in 16-step increments, alternating sides.
    for delta in range(16, 161, 16):
        candidates.append(192 + delta)
        candidates.append(192 - delta)
    candidates = [c for c in candidates if c >= 32 and c % 8 == 0]

    best_base = 192
    best_drift = float("inf")
    for base in candidates:
        try:
            m = SimpleUNet(in_ch=IN_CH, base=base, t_dim=t_dim)
        except Exception:
            continue
        n = param_count(m)
        drift = abs(n - target_params) / target_params
        if drift < best_drift:
            best_drift = drift
            best_base = base
        del m
        if drift < tol:
            return base, drift
    return best_base, best_drift


# ----------------------------------------------------------------------------
# Diffusion schedule + one-step training
# ----------------------------------------------------------------------------
def make_schedule(T, device):
    """Standard linear beta schedule -> cumulative alpha-bars."""
    betas = torch.linspace(1e-4, 2e-2, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alpha_bars


def diffusion_step(model, x0, optimizer, alpha_bars):
    """One diffusion training step: noise + timestep + MSE + optimizer.step."""
    B = x0.size(0)
    t = torch.randint(0, T_STEPS, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    ab = alpha_bars[t].view(B, 1, 1, 1)
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
    optimizer.zero_grad()
    pred = model(x_t, t)
    loss = F.mse_loss(pred, noise)
    loss.backward()
    optimizer.step()
    return loss.item()


# ----------------------------------------------------------------------------
# Latent-space pedagogy logging block
# ----------------------------------------------------------------------------
def log_latent_space_pedagogy():
    print("\n--- Why latent space? ---\n")
    pixel_values = 256 * 256 * 3
    latent_values = LATENT_HW * LATENT_HW * IN_CH
    spatial_ratio = pixel_values / latent_values
    # Attention cost is O(N^2) where N is token count. If we naively treat
    # one channel-cell as one token, the attention work ratio is the square
    # of the spatial ratio.
    attn_ratio = (256 * 256) ** 2 / (LATENT_HW * LATENT_HW) ** 2
    logger.info(
        f"Pixel-space diffusion on a 256x256 RGB image operates on "
        f"256*256*3 = {pixel_values:,} values."
    )
    logger.info(
        f"Latent-space diffusion on a {LATENT_HW}x{LATENT_HW}x{IN_CH} VAE latent "
        f"operates on {LATENT_HW}*{LATENT_HW}*{IN_CH} = {latent_values:,} values "
        f"-- ~{spatial_ratio:.0f}x smaller."
    )
    logger.info(
        f"Attention cost is O(N^2) in token count, so at one-token-per-cell "
        f"the attention work scales as ({256*256}/{LATENT_HW*LATENT_HW})^2 "
        f"= ~{attn_ratio:,.0f}x cheaper -- orders of magnitude."
    )
    logger.info(
        f"This tutorial uses synthetic [B, {IN_CH}, {LATENT_HW}, {LATENT_HW}] "
        f"tensors to represent the latent -- no VAE is loaded."
    )


# ----------------------------------------------------------------------------
# Batch-size sweep
# ----------------------------------------------------------------------------
def sweep_model(
    model_name,
    model_factory,
    device,
    results,
    throughput_by_config,
    oom_configs,
):
    """Sweep BATCHES for one model class, appending rows to results."""
    print(f"\n--- Batch-size sweep: {model_name} ---\n")
    for B in BATCHES:
        config_name = f"{model_name} B={B:>3}"
        logger.info(f"[{model_name}] preparing batch size {B}...")

        try:
            model = model_factory()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            alpha_bars = make_schedule(T_STEPS, device)
            latents = torch.randn(B, IN_CH, LATENT_HW, LATENT_HW, device=device)

            # Warmup OUTSIDE the timed region so kernel selection / autotuning
            # do not pollute the measured throughput.
            for _ in range(WARMUP_ITERATIONS):
                diffusion_step(model, latents, optimizer, alpha_bars)
            if device.type == "cuda":
                torch.cuda.synchronize()

            @benchmark
            def run():
                for _ in range(NUM_ITERATIONS):
                    diffusion_step(model, latents, optimizer, alpha_bars)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            r = run()
            throughput = (B * NUM_ITERATIONS) / r["time_seconds"]
            mem_str = (
                f"{r['memory_mb']:.1f} MB" if r["memory_mb"] is not None else "N/A"
            )
            logger.info(
                f"[{model_name}] B={B:<3} time={r['time_seconds']:.3f}s "
                f"peak_mem={mem_str} throughput={throughput:.1f} samples/sec"
            )
            results.append(
                {
                    "name": config_name,
                    "time_seconds": r["time_seconds"],
                    "memory_mb": r["memory_mb"],
                }
            )
            throughput_by_config[(model_name, B)] = throughput
        except torch.cuda.OutOfMemoryError:
            # OOM is a teaching moment, not a crash. Log and continue.
            logger.warning(
                f"[{model_name}] B={B} OOM -- skipping (this is part of the "
                f"perf story; try a smaller batch on this GPU)."
            )
            results.append(
                {
                    "name": f"{model_name} B={B:>3} (OOM)",
                    "time_seconds": 0.0,
                    "memory_mb": None,
                }
            )
            oom_configs.append((model_name, B))
            # Best-effort cleanup -- locals may not all exist depending on
            # where the OOM hit.
            for var in ("model", "optimizer", "latents", "alpha_bars"):
                if var in dir():
                    pass
        finally:
            # Drop everything we made for this config so memory does not bleed
            # into the next row (Pitfall 4 in RESEARCH.md).
            for var in ("model", "optimizer", "latents", "alpha_bars", "r"):
                if var in locals():
                    del locals()[var]
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


def main():
    print("\n" + "=" * 70)
    print("  DiT vs UNet: Large-Batch Diffusion Throughput Comparison")
    print("=" * 70)

    device = get_device()
    print_device_info()

    cap = get_gpu_capability()
    logger.info(f"PyTorch version: {torch.__version__}")
    if cap is not None:
        logger.info(f"GPU compute capability: SM {cap[0]}.{cap[1]}")
    else:
        logger.info("No CUDA GPU detected -- running on CPU (will be slow).")

    # ------------------------------------------------------------------
    # Build models and verify param-match (D-10)
    # ------------------------------------------------------------------
    print("\n--- Model construction + param-match verification ---\n")
    dit = SimpleDiT(
        in_ch=IN_CH, patch=2, H=LATENT_HW, W=LATENT_HW, dim=384, depth=6, heads=6
    ).to(device)
    n_dit = param_count(dit)
    logger.info(f"SimpleDiT params:  {n_dit:,}")

    # First try the default UNet sizing.
    unet_default = SimpleUNet(in_ch=IN_CH, base=192, t_dim=384)
    n_unet_default = param_count(unet_default)
    drift_default = abs(n_dit - n_unet_default) / n_dit
    logger.info(
        f"SimpleUNet (base=192) params: {n_unet_default:,} "
        f"(drift {drift_default:.2%})"
    )
    del unet_default

    if drift_default < PARAM_MATCH_TOLERANCE:
        chosen_base = 192
        chosen_drift = drift_default
        logger.info(
            f"Default UNet sizing is within {PARAM_MATCH_TOLERANCE:.0%} -- "
            f"using base={chosen_base}."
        )
    else:
        logger.info(
            f"Default UNet sizing is outside {PARAM_MATCH_TOLERANCE:.0%}. "
            f"Searching for a better `base`..."
        )
        chosen_base, chosen_drift = find_matching_unet(
            n_dit, t_dim=384, tol=PARAM_MATCH_TOLERANCE
        )
        logger.info(
            f"Selected UNet base={chosen_base} (drift {chosen_drift:.2%})."
        )

    unet = SimpleUNet(in_ch=IN_CH, base=chosen_base, t_dim=384).to(device)
    n_unet = param_count(unet)
    drift = abs(n_dit - n_unet) / n_dit
    logger.info(f"SimpleUNet params: {n_unet:,}")
    logger.info(f"Param-count drift: {drift:.2%} (tolerance {PARAM_MATCH_TOLERANCE:.0%})")

    # Hard assertion -- misconfiguration must fail loudly (D-10).
    assert drift < PARAM_MATCH_TOLERANCE, (
        f"Param counts drift by {drift:.1%} -- tune UNet `base` or DiT `dim/depth`"
    )

    # Free the verification UNet -- we will rebuild fresh per sweep config so
    # optimizer state from earlier rows does not bleed memory between rows.
    del dit, unet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Pedagogy: why latent space matters
    # ------------------------------------------------------------------
    log_latent_space_pedagogy()

    # ------------------------------------------------------------------
    # Run the batch-size sweep for both models
    # ------------------------------------------------------------------
    logger.info(
        f"Sweep config: BATCHES={BATCHES}, NUM_ITERATIONS={NUM_ITERATIONS}, "
        f"WARMUP_ITERATIONS={WARMUP_ITERATIONS}"
    )

    results: list[dict] = []
    throughput_by_config: dict[tuple[str, int], float] = {}
    oom_configs: list[tuple[str, int]] = []

    sweep_model(
        "DiT",
        lambda: SimpleDiT(
            in_ch=IN_CH,
            patch=2,
            H=LATENT_HW,
            W=LATENT_HW,
            dim=384,
            depth=6,
            heads=6,
        ).to(device),
        device,
        results,
        throughput_by_config,
        oom_configs,
    )
    sweep_model(
        "UNet",
        lambda: SimpleUNet(in_ch=IN_CH, base=chosen_base, t_dim=384).to(device),
        device,
        results,
        throughput_by_config,
        oom_configs,
    )

    # ------------------------------------------------------------------
    # Final benchmark table (time + peak memory) -- per D-15
    # ------------------------------------------------------------------
    print("\n--- Final benchmark table (time + peak memory per config) ---\n")
    print_benchmark_table(results)

    # ------------------------------------------------------------------
    # Throughput summary -- the headline metric
    # ------------------------------------------------------------------
    print("\n--- Throughput summary (samples/sec) ---\n")
    print(f"+{'-'*10}+{'-'*16}+{'-'*16}+{'-'*12}+")
    print(f"| {'Batch':>8} | {'DiT (samp/s)':>14} | {'UNet (samp/s)':>14} | {'DiT/UNet':>10} |")
    print(f"+{'-'*10}+{'-'*16}+{'-'*16}+{'-'*12}+")
    for B in BATCHES:
        d = throughput_by_config.get(("DiT", B))
        u = throughput_by_config.get(("UNet", B))
        d_str = f"{d:>14.1f}" if d is not None else f"{'OOM':>14}"
        u_str = f"{u:>14.1f}" if u is not None else f"{'OOM':>14}"
        if d is not None and u is not None and u > 0:
            ratio_str = f"{d/u:>10.2f}"
        else:
            ratio_str = f"{'N/A':>10}"
        print(f"| {B:>8} | {d_str} | {u_str} | {ratio_str} |")
    print(f"+{'-'*10}+{'-'*16}+{'-'*16}+{'-'*12}+")

    if oom_configs:
        print("\n--- OOM configs ---\n")
        for model_name, B in oom_configs:
            logger.info(f"  {model_name} at B={B} hit CUDA OOM (caught, sweep continued).")

    # ------------------------------------------------------------------
    # Honest qualitative read -- do not force the conclusion (Assumption A3)
    # ------------------------------------------------------------------
    print("\n--- Observations ---\n")
    logger.info(
        "The headline question for this tutorial: does DiT pull ahead of UNet "
        "as the batch grows? Read the throughput table above -- the DiT/UNet "
        "ratio column is the answer for this hardware. A ratio > 1 at large B "
        "means DiT is faster (transformer matmul bound + tensor cores). "
        "A ratio < 1 means UNet is faster (likely small-N regime where conv "
        "kernels still dominate)."
    )
    logger.info(
        "Peak memory tells the second half of the story: DiT typically has a "
        "flatter memory curve at this latent size because attention runs "
        "through SDPA (flash/efficient backends), while UNet activations grow "
        "with batch * channels * spatial."
    )

    print("\n" + "=" * 70)
    print("  Tutorial Complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
