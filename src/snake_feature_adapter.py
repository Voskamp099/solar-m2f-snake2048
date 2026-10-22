"""Bottleneck residual adapters inspired by Qi et al., DSConv (ICCV 2023).

Reference: https://arxiv.org/abs/2307.08388 (Sec. 3.1).
This is an independent 2D implementation, not a reproduction of DSCNet:
we use learned dense kernel weights, center-out cumulative transverse offsets,
bilinear sampling, and a zero-initialized residual gate around M2F features.
No DSCNet fusion strategy, topology loss, or medical data is included.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SnakeKernel2D(nn.Module):
    """Two connected K-point sampling paths, initialized as horizontal/vertical."""

    def __init__(self, channels, kernel_size=9, max_step=1., dynamic=True):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd and >= 3')
        self.channels = channels
        self.kernel_size = kernel_size
        self.max_step = float(max_step)
        self.dynamic = dynamic
        self.offset = nn.Conv2d(channels, 2 * (kernel_size - 1), 3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.weight = nn.Parameter(torch.empty(2, channels, channels, kernel_size))
        nn.init.normal_(self.weight, std=1. / math.sqrt(channels * kernel_size))
        if not dynamic:
            self.offset.requires_grad_(False)

    def transverse_offsets(self, x):
        b, _, h, w = x.shape
        k, radius = self.kernel_size, self.kernel_size // 2
        if not self.dynamic:
            return x.new_zeros((b, 2, k, h, w))
        steps = self.max_step * torch.tanh(self.offset(x))
        steps = steps.view(b, 2, k - 1, h, w)
        # On either side, steps are stored from the center towards the edge.
        left = steps[:, :, :radius].cumsum(2).flip(2)
        right = steps[:, :, radius:].cumsum(2)
        center = steps[:, :, :1] * 0.
        return torch.cat((left, center, right), dim=2)

    def forward(self, x):
        # Coordinates/sampling use FP32 for CUDA AMP compatibility and stability.
        # The inner 1x1 contractions may still be autocast by the caller.
        x = x.float()
        b, _, h, w = x.shape
        if min(h, w) <= 1:
            raise ValueError('Snake sampling requires H,W > 1')
        offsets = self.transverse_offsets(x).float()
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                                torch.arange(w, device=x.device, dtype=x.dtype), indexing='ij')
        output = []
        radius = self.kernel_size // 2
        for axis in range(2):
            value = None
            for idx in range(self.kernel_size):
                fixed = idx - radius
                if axis == 0:
                    gx, gy = xx + fixed, yy + offsets[:, axis, idx]
                else:
                    gx, gy = xx + offsets[:, axis, idx], yy + fixed
                gx = gx.expand(b, h, w)
                gy = gy.expand(b, h, w)
                grid = torch.stack((2 * gx / (w - 1) - 1, 2 * gy / (h - 1) - 1), dim=-1)
                # Sequential samples avoid a materialized B*C*K*H*W tensor.
                sampled = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
                weighted = F.conv2d(sampled, self.weight[axis, :, :, idx, None, None])
                value = weighted if value is None else value + weighted
            output.append(value)
        return torch.cat(output, dim=1)


class ResidualFeatureAdapter(nn.Module):
    def __init__(self, channels=256, bottleneck=32, mode='snake', kernel_size=9):
        super().__init__()
        if mode not in ('snake', 'fixed_snake', 'conv'):
            raise ValueError(mode)
        self.mode = mode
        self.reduce = nn.Sequential(nn.Conv2d(channels, bottleneck, 1),
                                    nn.GroupNorm(math.gcd(8, bottleneck), bottleneck), nn.GELU())
        self.spatial = (nn.Conv2d(bottleneck, 2 * bottleneck, 3, padding=1)
                        if mode == 'conv' else SnakeKernel2D(bottleneck, kernel_size,
                                                             dynamic=mode == 'snake'))
        self.expand = nn.Conv2d(2 * bottleneck, channels, 1)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        correction = self.expand(F.gelu(self.spatial(self.reduce(x))))
        return x + self.alpha * correction.to(x.dtype)


def attach_feature_adapters(model, *, mode='snake', mask_features=True, scales=(), bottleneck=32, kernel_size=9):
    """Attach after loading original weights; expanded checkpoints attach before load.

    One mask adapter is shared by all query-decoder prediction heads; scale 0/1/2
    address the existing coarse-to-fine 1/32,1/16,1/8 features. The hook changes
    predictor inputs only. It never convolves query tokens or class logits.
    """
    head = model.sem_seg_head
    if hasattr(head, '_snake_feature_handle'):
        raise ValueError('Adapters already installed; remove before reinstalling')
    selected = tuple(sorted(set(scales)))
    if any(level not in (0, 1, 2) for level in selected):
        raise ValueError('scales must be a subset of (0,1,2)')
    if not mask_features and not selected:
        raise ValueError('Select at least one feature site')
    device = next(model.parameters()).device
    head.snake_mask_adapter = (ResidualFeatureAdapter(mode=mode, bottleneck=bottleneck, kernel_size=kernel_size)
                               if mask_features else nn.Identity()).to(device)
    head.snake_scale_adapters = nn.ModuleDict({str(i): ResidualFeatureAdapter(mode=mode, bottleneck=bottleneck,
                                                                           kernel_size=kernel_size)
                                              for i in selected}).to(device)
    config = dict(mode=mode, mask_features=mask_features, scales=list(selected),
                  channels=256, bottleneck=bottleneck, kernel_size=kernel_size,
                  reference='https://arxiv.org/abs/2307.08388', implementation='independent_bottleneck_residual_v1')
    head._snake_adapter_config = config

    def adapt_inputs(_module, args):
        features, masks, *rest = args
        if len(features) != 3:
            raise ValueError('Expected the original three M2F spatial feature levels')
        enhanced = [head.snake_scale_adapters[str(i)](feature) if str(i) in head.snake_scale_adapters else feature
                    for i, feature in enumerate(features)]
        return (enhanced, head.snake_mask_adapter(masks), *rest)

    head._snake_feature_handle = head.predictor.register_forward_pre_hook(adapt_inputs)
    return config
