import torch
import torch.nn as nn
from typing import Optional

"""
Improved SRCNN-like models for image super-resolution (image-to-image refinement)
-------------------------------------------------------------------------------
Key design goals implemented here:
1) Residual learning inside the network via ResidualBlocks (+ residual scaling) for stable deep training.
2) Keep the model architecture independent of scale factors (x2/x3/x4/x6 ...). 
   -> You can train on bicubically upscaled inputs to HR size; the network maps HR->HR residuals.
3) Use larger receptive field at the head (k=9) then 3x3 residual blocks; no BatchNorm (SR detail preservation).
4) Prefer PReLU activations; optional lightweight Channel Attention (SE) every N blocks for medium/high.
5) Global skip from input to output; final clamp-to-[0,1] optional based on input normalization.

Variants:
- SRCNN_low:    64 channels, 4 residual blocks, no channel attention.
- SRCNN_medium: 64 channels, 10 residual blocks, channel attention every 2 blocks.
- SRCNN_high:   64 channels, 20 residual blocks, channel attention every 2 blocks.

All three networks keep width=64 so depth is the main complexity driver. This aligns with the goal to
compare simple vs. complex CNNs without entangling with scale-specific layers.

Training notes (summary):
- Train on LR patches upscaled to HR size (e.g., bicubic). Use 128x128 or 192x192 HR patches.
- Loss: L1 or Charbonnier; consider perceptual/LPIPS for qualitative comparisons.
- Optim: AdamW or Adam (lr=1e-4), cosine decay, gradient clipping (e.g., 1.0), EMA of weights.
- Deep nets: keep residual_scaling=0.1 in blocks to stabilize. No BatchNorm.
- Evaluate PSNR/SSIM on Y-channel; report LPIPS for perceptual.
"""


# -----------------------------
# Building blocks
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, use_se: bool = False, residual_scaling: float = 0.1):
        super().__init__()
        self.residual_scaling = residual_scaling
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.PReLU(num_parameters=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act(y)
        y = self.conv2(y)
        y = self.se(y)
        return x + self.residual_scaling * y


class SRCNNBase(nn.Module):
    """Backbone for scale-agnostic SR refinement networks (HR->HR mapping).

    Args:
        num_blocks: number of residual blocks in the trunk
        channels:   feature width
        se_period:  apply SE attention every se_period blocks (0 disables SE)
        residual_scaling: scaling inside residual blocks
        clamp_output: clamp output to [0,1] (set True if inputs are normalized to [0,1])
    """
    def __init__(
        self,
        num_blocks: int,
        channels: int = 64,
        se_period: int = 0,
        residual_scaling: float = 0.1,
        clamp_output: bool = True,
    ):
        super().__init__()
        self.clamp_output = clamp_output

        self.head = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=9, padding=4),
            nn.PReLU(num_parameters=channels),
        )

        blocks = []
        for i in range(num_blocks):
            use_se = (se_period > 0) and ((i + 1) % se_period == 0)
            blocks.append(ResidualBlock(channels, use_se=use_se, residual_scaling=residual_scaling))
        self.trunk = nn.Sequential(*blocks)

        self.tail = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

        self.global_skip = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.global_skip(x)
        y = self.head(x)
        y = self.trunk(y)
        y = self.tail(y)
        out = y + residual
        if self.clamp_output:
            out = torch.clamp(out, 0.0, 1.0)
        return out


# -----------------------------
# Model Variants
# -----------------------------
class SRCNN_low(SRCNNBase):
    def __init__(self, clamp_output: bool = True):
        super().__init__(num_blocks=4, channels=64, se_period=0, residual_scaling=0.1, clamp_output=clamp_output)


class SRCNN_medium(SRCNNBase):
    def __init__(self, clamp_output: bool = True):
        super().__init__(num_blocks=10, channels=64, se_period=2, residual_scaling=0.1, clamp_output=clamp_output)


class SRCNN_high(SRCNNBase):
    def __init__(self, clamp_output: bool = True):
        super().__init__(num_blocks=20, channels=64, se_period=2, residual_scaling=0.1, clamp_output=clamp_output)


# -----------------------------
# Utilities
# -----------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(variant: str = "low", clamp_output: bool = True) -> nn.Module:
    """Factory to instantiate a variant: 'low' | 'medium' | 'high'."""
    variant = variant.lower()
    if variant == "low":
        return SRCNN_low(clamp_output=clamp_output)
    if variant == "medium":
        return SRCNN_medium(clamp_output=clamp_output)
    if variant == "high":
        return SRCNN_high(clamp_output=clamp_output)
    raise ValueError(f"Unknown variant: {variant}")


# -----------------------------
# Example
# -----------------------------
if __name__ == "__main__":
    model = build_model("medium")
    print(model)
    print("Trainable params:", count_parameters(model))


    x = torch.rand(1, 3, 128, 128) # assume inputs in [0,1]
    with torch.no_grad():
        y = model(x)
    print("Output shape:", y.shape)