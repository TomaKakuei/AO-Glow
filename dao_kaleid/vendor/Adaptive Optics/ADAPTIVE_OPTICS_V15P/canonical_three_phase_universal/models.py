"""Three Canonical Universal 2D Neural Architectures for Autonomous Alignment.

All 3 architectures share identical depth across systems, receive N-field 2D continuous
maps (speckle or OPD), and output decoupled modular lens group poses (5 DOFs per group).

Architectures:
  1. Arch-A (Canonical Multi-Scale ResNet): Multi-scale stem + 3-stage DW-Separable ResNet.
  2. Arch-B (Mobile-SE Inverted Residual Net): Patchify stem + 3-stage MBConv with SE channel attention.
  3. Arch-C (Cross-Field Axial Attention Net): Per-field spatial CNN + 2-layer Cross-Field Multi-Head Attention.
"""

from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Common Building Blocks
# -----------------------------------------------------------------------------

class GroupFactorizedHead(nn.Module):
    """Independent MLP sub-heads per physical lens group (5 DOFs per group)."""

    def __init__(self, in_features: int, group_dofs: Sequence[int]):
        super().__init__()
        self.group_dofs = list(group_dofs)
        self.sub_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, 64),
                nn.GELU(),
                nn.Linear(64, dofs),
            )
            for dofs in self.group_dofs
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [head(x) for head in self.sub_heads]
        return torch.cat(outputs, dim=-1)


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable convolution with LayerNorm and GELU."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        p = kernel_size // 2
        self.dw = nn.Conv2d(in_c, in_c, kernel_size=kernel_size, stride=stride, padding=p, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.fc2(self.act(self.fc1(w)))).view(b, c, 1, 1)
        return x * w


# -----------------------------------------------------------------------------
# Architecture A: Canonical Multi-Scale ResNet (Arch-A)
# -----------------------------------------------------------------------------

class MultiScaleStem(nn.Module):
    """Multi-scale parallel convolution stem (3x3, 5x5, 7x7)."""

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        branch_c = out_channels // 4
        self.b1 = nn.Conv2d(in_channels, branch_c, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(in_channels, branch_c, kernel_size=5, padding=2)
        self.b3 = nn.Conv2d(in_channels, branch_c, kernel_size=7, padding=3)
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_channels, branch_c, kernel_size=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        out = torch.cat(feats, dim=1)
        return self.act(self.bn(out))


class ArchA_CanonicalResNet(nn.Module):
    """Arch-A: Multi-Scale Stem + 3-stage DW-Separable ResNet + Group-Factorized Head."""

    def __init__(self, in_fields: int, group_dofs: Sequence[int]):
        super().__init__()
        self.stem = MultiScaleStem(in_fields, out_channels=64)
        
        # 3 Stages of Depthwise-Separable ResNet
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv2d(64, 128, kernel_size=3, stride=2),
            DepthwiseSeparableConv2d(128, 128, kernel_size=3, stride=1),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv2d(128, 256, kernel_size=3, stride=2),
            DepthwiseSeparableConv2d(256, 256, kernel_size=3, stride=1),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv2d(256, 256, kernel_size=3, stride=2),
            DepthwiseSeparableConv2d(256, 256, kernel_size=3, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = GroupFactorizedHead(256, group_dofs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.pool(feat).flatten(1)
        return self.head(feat)


# -----------------------------------------------------------------------------
# Architecture B: Mobile-SE Inverted Residual Net (Arch-B)
# -----------------------------------------------------------------------------

class InvertedResidualMBConv(nn.Module):
    """Inverted residual block with SE channel attention and expansion ratio 2.0."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1, expand_ratio: float = 2.0):
        super().__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_c == out_c
        hidden_c = int(in_c * expand_ratio)

        layers = []
        if expand_ratio != 1.0:
            layers.extend([nn.Conv2d(in_c, hidden_c, kernel_size=1, bias=False), nn.BatchNorm2d(hidden_c), nn.GELU()])
        
        layers.extend([
            nn.Conv2d(hidden_c, hidden_c, kernel_size=3, stride=stride, padding=1, groups=hidden_c, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.GELU(),
            SqueezeExcitation(hidden_c),
            nn.Conv2d(hidden_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class ArchB_MobileSENet(nn.Module):
    """Arch-B: Patchify Stem + 3-stage Inverted Residual MBConv + SE Attention + Group-Factorized Head."""

    def __init__(self, in_fields: int, group_dofs: Sequence[int]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_fields, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            InvertedResidualMBConv(64, 128, stride=2),
            InvertedResidualMBConv(128, 128, stride=1),
        )
        self.stage2 = nn.Sequential(
            InvertedResidualMBConv(128, 256, stride=2),
            InvertedResidualMBConv(256, 256, stride=1),
        )
        self.stage3 = nn.Sequential(
            InvertedResidualMBConv(256, 256, stride=2),
            InvertedResidualMBConv(256, 256, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = GroupFactorizedHead(256, group_dofs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.pool(feat).flatten(1)
        return self.head(feat)


# -----------------------------------------------------------------------------
# Architecture C: Cross-Field Axial Attention Net (Arch-C)
# -----------------------------------------------------------------------------

class FieldSpatialEncoder(nn.Module):
    """Encodes each 2D field map into a single field embedding vector (128-d)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, single_field_map: torch.Tensor) -> torch.Tensor:
        return self.conv(single_field_map)


class ArchC_CrossFieldAttentionNet(nn.Module):
    """Arch-C: Independent Field Spatial Encoders + 2-layer Cross-Field Multi-Head Attention."""

    def __init__(self, in_fields: int, group_dofs: Sequence[int], embed_dim: int = 128):
        super().__init__()
        self.in_fields = in_fields
        self.embed_dim = embed_dim
        self.spatial_encoder = FieldSpatialEncoder(out_dim=embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, in_fields, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4, dim_feedforward=256, dropout=0.05, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = GroupFactorizedHead(embed_dim, group_dofs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, h, w = x.shape
        # Process each field independently through shared spatial encoder
        x_reshaped = x.view(b * n, 1, h, w)
        field_tokens = self.spatial_encoder(x_reshaped).view(b, n, self.embed_dim)
        field_tokens = field_tokens + self.pos_embed

        # Cross-Field Transformer Attention
        attn_out = self.transformer(field_tokens)
        pooled = attn_out.mean(dim=1)  # Pool across fields
        return self.head(pooled)


# -----------------------------------------------------------------------------
# Architecture 4: TolNet Literature Baseline (Plain CNN + Coupled Dense Head)
# -----------------------------------------------------------------------------

class TolNet_PlainCNN(nn.Module):
    """TolNet (Physics-Informed Alignment Baseline, Sun et al. / Optica 2024).
    
    Architecture Characteristics:
      - Stem: Single-scale 3x3 convolution (no multi-scale branch parallel paths).
      - Backbone: 4-layer plain feedforward CNN with BatchNorm and ReLU (no residual connections).
      - Head: Single coupled Dense MLP (all physical DOFs entangled in one linear projection, no group factorization).
      - Parameter footprint: ~4.2M parameters.
    """

    def __init__(self, in_channels: int, group_dofs: Sequence[int]):
        super().__init__()
        self.group_dofs = list(group_dofs)
        total_dofs = sum(group_dofs)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, total_dofs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.head(feat)


# -----------------------------------------------------------------------------
# Factory Builder
# -----------------------------------------------------------------------------

def build_universal_model(arch_name: str, in_fields: int, group_dofs: Sequence[int]) -> nn.Module:
    arch = arch_name.lower()
    if "arch_a" in arch or "resnet" in arch:
        return ArchA_CanonicalResNet(in_fields, group_dofs)
    elif "arch_b" in arch or "mobile" in arch:
        return ArchB_MobileSENet(in_fields, group_dofs)
    elif "arch_c" in arch or "attention" in arch or "attn" in arch:
        return ArchC_CrossFieldAttentionNet(in_fields, group_dofs)
    elif "tolnet" in arch or "plain_cnn" in arch:
        return TolNet_PlainCNN(in_fields, group_dofs)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")


if __name__ == "__main__":
    print("Testing Three Universal Architectures on synthetic batch:")
    b, n, h, w = 2, 9, 64, 64
    x = torch.randn(b, n, h, w)
    dofs = [5, 5, 5]  # 3 groups x 5 DOFs = 15 DOFs

    for name in ("Arch-A (ResNet)", "Arch-B (Mobile-SE)", "Arch-C (CrossField-Attn)"):
        m = build_universal_model(name, n, dofs)
        out = m(x)
        p_cnt = sum(p.numel() for p in m.parameters())
        print(f"  {name:<24} Params: {p_cnt:,} | Output shape: {list(out.shape)}")
