"""CPU checks for connected sampling, gradients, checkpointing and M2F hooks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from snake_feature_adapter import SnakeKernel2D, ResidualFeatureAdapter, attach_feature_adapters

torch.set_num_threads(2)
torch.manual_seed(20261017)
checks = {}
x = torch.randn(1, 4, 11, 13)
kernel = SnakeKernel2D(4, kernel_size=5)
actual = kernel(x)
horizontal = F.conv2d(F.pad(x, (2, 2, 0, 0), mode='replicate'), kernel.weight[0].unsqueeze(2))
vertical = F.conv2d(F.pad(x, (0, 0, 2, 2), mode='replicate'), kernel.weight[1].unsqueeze(3))
assert torch.allclose(actual, torch.cat((horizontal, vertical), dim=1), atol=2e-6)
checks['zero_offset_equals_independent_strip_convolutions'] = True
with torch.no_grad():
    kernel.offset.bias.fill_(.4)
offset = kernel.transverse_offsets(x)
step = torch.tanh(torch.tensor(.4))
expected = step * torch.tensor([2., 1., 0., 1., 2.])
assert torch.allclose(offset[0, 0, :, 3, 4], expected)
assert offset[:, :, 2].count_nonzero() == 0
checks['center_fixed_and_offsets_accumulate_outwards'] = True
kernel(x).square().mean().backward()
assert torch.isfinite(kernel.offset.weight.grad).all() and kernel.offset.weight.grad.abs().sum() > 0
assert torch.isfinite(kernel.weight.grad).all() and kernel.weight.grad.abs().sum() > 0
checks['offset_and_kernel_gradients_nonzero'] = True

adapter = ResidualFeatureAdapter(channels=16, bottleneck=4, kernel_size=5)
features = torch.randn(1, 16, 12, 14)
assert torch.equal(adapter(features), features)
optimizer = torch.optim.AdamW(adapter.parameters(), lr=.01)
for _ in range(4):
    optimizer.zero_grad()
    adapter(features).square().mean().backward()
    optimizer.step()
assert adapter.alpha.detach().abs() > 0
assert adapter.spatial.offset.weight.grad.abs().sum() > 0
checks['initial_identity_then_branch_learns'] = True

class Predictor(nn.Module):
    def forward(self, features, masks, padding=None):
        return dict(features=features, masks=masks, padding=padding)

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = Predictor()
        self.original = nn.Parameter(torch.ones(()))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sem_seg_head = Head()

model = Model()
base = copy.deepcopy(model.state_dict())
config = attach_feature_adapters(model, mask_features=True, scales=(0, 1, 2), bottleneck=4, kernel_size=5)
maps = [torch.randn(1, 256, s, s) for s in (3, 6, 12)]
masks = torch.randn(1, 256, 24, 24)
result = model.sem_seg_head.predictor(maps, masks, None)
assert all(torch.equal(a, b) for a, b in zip(maps, result['features']))
assert torch.equal(masks, result['masks'])
assert all(torch.equal(value, model.state_dict()[key]) for key, value in base.items())
for value in model.sem_seg_head.snake_scale_adapters.values():
    value.alpha.data.fill_(.1)
model.sem_seg_head.snake_mask_adapter.alpha.data.fill_(.1)
result = model.sem_seg_head.predictor(maps, masks, None)
assert all(not torch.equal(a, b) for a, b in zip(maps, result['features']))
assert not torch.equal(masks, result['masks'])
checks['three_scales_and_shared_mask_hook_identity_then_active'] = True
other = Model()
attach_feature_adapters(other, mask_features=True, scales=(0, 1, 2), bottleneck=4, kernel_size=5)
other.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
other_result = other.sem_seg_head.predictor(maps, masks, None)
assert torch.equal(result['masks'], other_result['masks'])
assert all(torch.equal(a, b) for a, b in zip(result['features'], other_result['features']))
checks['expanded_state_dict_strict_roundtrip'] = True
report = dict(status='PASS', device='CPU', checks=checks, config=config,
              limitations=['Actual M2F CUDA forward/backward and peak VRAM have not been tested.',
                           'No training accuracy/PQ claim is implied by these implementation checks.'])
Path(__file__).with_name('snake_adapter_cpu_checks.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
