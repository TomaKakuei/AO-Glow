"""Versioned trained runtime architectures; AST-extracted from original sources."""
import numpy as np
import torch
from torch import nn
from ._canonical import ArchA_CanonicalResNet, GroupFactorizedHead, build_universal_model
BRANCHES=("front","rear")
GROUPS=("G1","G2","G3")
def build(branch): return ModularNetwork(3 if branch=="front" else 4)

class ModularNetwork(ArchA_CanonicalResNet):

    def __init__(self, groups):
        super().__init__(9, [5] * groups)
        y, x = torch.meshgrid((torch.arange(128) + 0.5) / 64 - 1, (torch.arange(128) + 0.5) / 64 - 1, indexing='ij')
        basis = torch.stack([torch.ones_like(x), x, y, x * y, x * x, y * y, torch.sin(torch.pi * x), torch.sin(torch.pi * y)])
        self.register_buffer('sensor_moments', basis)
        self.head = GroupFactorizedHead(256 + 9 * len(basis), [5] * groups)

    def forward(self, x):
        feat = self.stage3(self.stage2(self.stage1(self.stem(x))))
        global_features = self.pool(feat).flatten(1)
        moments = torch.einsum('bfxy,kxy->bfk', x, self.sensor_moments) / 16384.0
        return self.head(torch.cat([global_features, moments.flatten(1)], dim=1))



class CoordinatedSpeckle(torch.nn.Module):

    def __init__(self, branch, payload):
        super().__init__()
        if branch not in BRANCHES:
            raise ValueError(branch)
        self.branch = branch
        self.groups = 3 if branch == 'front' else 4
        self.base = build(branch)
        if 'state_dict' in payload:
            self.base.load_state_dict(payload['state_dict'])
        for key in ('target_mean', 'target_std', 'reference', 'noise_std'):
            self.register_buffer(key, torch.as_tensor(payload[key], dtype=torch.float32).clone())
        width = 128
        self.node_encoder = torch.nn.Sequential(torch.nn.Linear(68, 64), torch.nn.GELU(), torch.nn.Linear(64, width), torch.nn.GELU())
        self.raw_global = torch.nn.Sequential(torch.nn.Linear(9 * 16 * 16, width), torch.nn.GELU(), torch.nn.Linear(width, width))
        self.feature_projection = torch.nn.Linear(256 + 72, width)
        self.pose_projections = torch.nn.ModuleList([torch.nn.Linear(5, width) for _ in range(self.groups)])
        self.module_identity = torch.nn.Parameter(torch.zeros(1, self.groups, width))
        self.cross_norm = torch.nn.LayerNorm(width)
        self.node_attention = torch.nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.module_norm = torch.nn.LayerNorm(width)
        self.module_attention = torch.nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.ff_norm = torch.nn.LayerNorm(width)
        self.ff = torch.nn.Sequential(torch.nn.Linear(width, 256), torch.nn.GELU(), torch.nn.Linear(256, width))
        self.correction_heads = torch.nn.ModuleList([torch.nn.Linear(width, 5) for _ in range(self.groups)])
        for head in self.correction_heads:
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
        y, x = torch.meshgrid((torch.arange(16) + 0.5) / 8 - 1, (torch.arange(16) + 0.5) / 8 - 1, indexing='ij')
        xy = torch.stack((x.flatten(), y.flatten()), 1).repeat(9, 1)
        field = torch.repeat_interleave(torch.linspace(-1, 1, 9), 256)[:, None]
        self.register_buffer('coordinates', torch.cat((xy, field, xy.square().sum(1, keepdim=True)), 1))

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (9, 128, 128):
            raise ValueError('Expected batch x9x128x128 noisy sensor ADU')
        x = torch.asinh((images - self.reference) / self.noise_std / 3.0)
        feat = self.base.stage3(self.base.stage2(self.base.stage1(self.base.stem(x))))
        moments = torch.einsum('bfxy,kxy->bfk', x, self.base.sensor_moments) / 16384.0
        feat = torch.cat((self.base.pool(feat).flatten(1), moments.flatten(1)), 1)
        estimate = self.base.head(feat)
        raw = (images - self.reference) / 16384.0
        patches = raw.unfold(2, 8, 8).unfold(3, 8, 8).reshape(len(raw), 9 * 16 * 16, 64)
        nodes = self.node_encoder(torch.cat((patches, self.coordinates[None].expand(len(raw), -1, -1)), 2))
        context = torch.stack([proj(estimate[:, 5 * i:5 * i + 5]) for i, proj in enumerate(self.pose_projections)], 1)
        context = context + self.feature_projection(feat)[:, None] + self.raw_global(patches.mean(2))[:, None] + self.module_identity
        context = context + self.node_attention(self.cross_norm(context), nodes, nodes, need_weights=False)[0]
        normalized = self.module_norm(context)
        context = context + self.module_attention(normalized, normalized, normalized, need_weights=False)[0]
        context = context + self.ff(self.ff_norm(context))
        correction = torch.cat([head(context[:, i]) for i, head in enumerate(self.correction_heads)], 1)
        return estimate + correction

    def physical_prediction(self, images):
        return self(images) * self.target_std + self.target_mean

    def command(self, images):
        return -self.physical_prediction(images).clamp(-1.0, 1.0)

    def train(self, mode=True):
        super().train(mode)
        for layer in self.base.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.eval()
        return self



class JointModel(torch.nn.Module):

    def __init__(self, payloads):
        super().__init__()
        self.networks = torch.nn.ModuleList()
        for payload in payloads:
            network = build_universal_model('arch_a', payload['in_fields'], payload['group_dofs'])
            network.load_state_dict(payload['state_dict'])
            self.networks.append(network)

    def forward(self, image):
        return torch.cat([network(image) for network in self.networks], dim=-1)



class CoordinatedNikon(torch.nn.Module):
    """Original encoders plus native-node attention and cross-module residuals.

    All three module tokens attend to the SAME noisy node stack and to each
    other. The final zero-initialized heads preserve the archived model exactly
    at initialization. Seven 5-DOF bodies remain grouped as15/15/5 outputs.
    """

    def __init__(self, payloads, raw_mean, raw_std, nodes_coordinates):
        super().__init__()
        self.base = JointModel(payloads)
        self.register_buffer('raw_mean', torch.as_tensor(raw_mean, dtype=torch.float32))
        self.register_buffer('raw_std', torch.as_tensor(raw_std, dtype=torch.float32).clamp_min(0.015))
        self.register_buffer('target_mean', torch.tensor(np.concatenate([p['target_mean'] for p in payloads]), dtype=torch.float32))
        self.register_buffer('target_std', torch.tensor(np.concatenate([p['target_std'] for p in payloads]), dtype=torch.float32))
        xy = np.tile(np.asarray(nodes_coordinates, dtype=np.float32), (3, 1))
        field = np.repeat([0.0, -1.0, 1.0], 49)[:, None]
        coords = np.c_[xy, field, (xy * xy).sum(1)]
        self.register_buffer('node_coordinates', torch.tensor(coords, dtype=torch.float32))
        width = 128
        self.node_encoder = torch.nn.Sequential(torch.nn.Linear(5, 64), torch.nn.GELU(), torch.nn.Linear(64, width), torch.nn.GELU())
        self.raw_global = torch.nn.Sequential(torch.nn.Linear(147, width), torch.nn.GELU(), torch.nn.Linear(width, width))
        self.feature_projections = torch.nn.ModuleList([torch.nn.Linear(256, width) for _ in GROUPS])
        self.pose_projections = torch.nn.ModuleList([torch.nn.Linear(n, width) for n in (15, 15, 5)])
        self.module_identity = torch.nn.Parameter(torch.zeros(1, 3, width))
        self.cross_norm = torch.nn.LayerNorm(width)
        self.node_attention = torch.nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.module_norm = torch.nn.LayerNorm(width)
        self.module_attention = torch.nn.MultiheadAttention(width, 4, dropout=0.0, batch_first=True)
        self.ff_norm = torch.nn.LayerNorm(width)
        self.ff = torch.nn.Sequential(torch.nn.Linear(width, 256), torch.nn.GELU(), torch.nn.Linear(256, width))
        self.correction_heads = torch.nn.ModuleList([torch.nn.Linear(width, n) for n in (15, 15, 5)])
        for head in self.correction_heads:
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)

    def forward(self, image, raw):
        features = []
        estimates = []
        for network in self.base.networks:
            f = network.stage3(network.stage2(network.stage1(network.stem(image))))
            f = network.pool(f).flatten(1)
            features.append(f)
            estimates.append(network.head(f))
        native = ((raw - self.raw_mean) / self.raw_std).flatten(1)
        coordinates = self.node_coordinates[None].expand(len(image), -1, -1)
        nodes = self.node_encoder(torch.cat((native[:, :, None], coordinates), dim=-1))
        context = torch.stack([fp(f) + pp(p) for fp, pp, f, p in zip(self.feature_projections, self.pose_projections, features, estimates)], dim=1)
        context = context + self.raw_global(native)[:, None] + self.module_identity
        context = context + self.node_attention(self.cross_norm(context), nodes, nodes, need_weights=False)[0]
        normalized = self.module_norm(context)
        context = context + self.module_attention(normalized, normalized, normalized, need_weights=False)[0]
        context = context + self.ff(self.ff_norm(context))
        correction = torch.cat([head(context[:, i]) for i, head in enumerate(self.correction_heads)], dim=-1)
        return torch.cat(estimates, dim=-1) + correction

    def physical_prediction(self, image, raw):
        return self(image, raw) * self.target_std + self.target_mean
