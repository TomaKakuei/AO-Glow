"""Model definitions for the user-selected Trepan2p ablations.

The baseline layers are imported from ``train_speckle_cnn.py`` so that the
ablation models differ only in the declared factor (input fields, latent width,
MLP width, or MLP depth).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from train_speckle_cnn import (
    DepthwiseSeparableBlock,
    MonolithicResMLP,
    MultiScaleStem,
    SpeckleCNN,
)


class FlexibleSpeckleCNN(nn.Module):
    """Baseline speckle CNN with configurable channel count and head capacity."""

    def __init__(
        self,
        *,
        in_channels: int,
        output_dim: int = 20,
        latent_dim: int = 1024,
        mlp_width: int = 2048,
        mlp_blocks: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = MultiScaleStem(in_channels=in_channels, out_channels=64)
        self.stage1 = DepthwiseSeparableBlock(64, 128, stride=2)
        self.stage2 = DepthwiseSeparableBlock(128, 256, stride=2)
        self.stage3 = DepthwiseSeparableBlock(256, 512, stride=2)
        self.stage4 = DepthwiseSeparableBlock(512, latent_dim, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = MonolithicResMLP(
            input_dim=latent_dim,
            output_dim=output_dim,
            width=mlp_width,
            blocks=mlp_blocks,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.mlp(torch.flatten(x, 1))


class ShwfsResMLP(nn.Module):
    """Capacity-matched regression head for cached SHWFS coefficient vectors."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int = 20,
        mlp_width: int = 2048,
        mlp_blocks: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = MonolithicResMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            width=mlp_width,
            blocks=mlp_blocks,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.flatten(x, 1))


def build_model(config: dict) -> nn.Module:
    """Build either the frozen paper baseline or a declared ablation model."""

    model_kind = str(config.get("model_kind", "flexible_cnn"))
    if model_kind == "paper_baseline":
        return SpeckleCNN(output_dim=int(config.get("output_dim", 20)))
    if model_kind == "shwfs_mlp":
        return ShwfsResMLP(
            input_dim=int(config["input_dim"]),
            output_dim=int(config.get("output_dim", 20)),
            mlp_width=int(config.get("mlp_width", 2048)),
            mlp_blocks=int(config.get("mlp_blocks", 8)),
            dropout=float(config.get("dropout", 0.1)),
        )
    if model_kind != "flexible_cnn":
        raise ValueError(f"Unknown model_kind={model_kind!r}")
    return FlexibleSpeckleCNN(
        in_channels=int(config["in_channels"]),
        output_dim=int(config.get("output_dim", 20)),
        latent_dim=int(config.get("latent_dim", 1024)),
        mlp_width=int(config.get("mlp_width", 2048)),
        mlp_blocks=int(config.get("mlp_blocks", 8)),
        dropout=float(config.get("dropout", 0.1)),
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def state_dict_to_half(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Store floating weights as FP16; loading into an FP32 module restores FP32."""

    result: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu()
        result[key] = tensor.half() if tensor.is_floating_point() else tensor
    return result
