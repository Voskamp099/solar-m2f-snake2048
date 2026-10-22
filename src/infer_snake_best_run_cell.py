import os, sys, time, json, shutil, subprocess
from pathlib import Path

RUN_STARTED = time.monotonic()
import zipfile
archives = list(Path('/kaggle/input').rglob('solar_payload.bin'))
assert len(archives) == 1, archives
ASSETS = Path('/kaggle/working/solar_assets')
ASSETS.mkdir(exist_ok=True)
with zipfile.ZipFile(archives[0]) as archive:
    archive.extractall(ASSETS)
INPUTS = ASSETS
RUNTIME = Path('/kaggle/working/solar_m2f_runtime')
RUNTIME.mkdir(exist_ok=True)
for name in ['Mask2Former', 'detectron2']:
    target = RUNTIME / name
    if not target.exists():
        shutil.copytree(ASSETS / 'vendor' / name, target)
print('ASSETS:', ASSETS, flush=True)
print('Python:', sys.version, flush=True)
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
assert torch.cuda.is_available(), 'A free Kaggle GPU is required; no CPU full run.'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'ninja', 'pycocotools', 'timm==0.9.16', 'fvcore==0.1.5.post20221221',
    'iopath==0.1.9', 'yacs', 'omegaconf>=2.1,<2.4', 'hydra-core',
    'termcolor', 'tabulate', 'cloudpickle', 'tensorboard', 'shapely',
    'scipy', 'submitit', 'black', 'packaging'], check=True, timeout=900)
# The Mask2Former R50 path needs no Detectron2 CUDA-specific operators.
# Build its normal CPU extension; deformable attention uses the existing
# project's differentiable PyTorch reference implementation during inference.
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MAX_JOBS='2')
log_path = Path('/kaggle/working/detectron2_install.log')
print('Building Detectron2 CPU extension; log:', log_path, flush=True)
with log_path.open('w') as log:
    result = subprocess.run([sys.executable, '-m', 'pip', 'install',
        '--no-build-isolation', '--no-deps', str(RUNTIME / 'detectron2')],
        env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
print(log_path.read_text()[-8000:], flush=True)
assert result.returncode == 0, 'See detectron2_install.log'
sys.path[:0] = [str(ASSETS), str(RUNTIME / 'Mask2Former')]
from solar_runtime import install_msda_backend
BACKEND = install_msda_backend('torch')
import detectron2, mask2former
print('Runtime ready:', detectron2.__version__, BACKEND, flush=True)
print('SETUP_SECONDS:', round(time.monotonic() - RUN_STARTED), flush=True)

import csv, gc, hashlib, itertools, json, time, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from pycocotools import mask as mu
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.data import detection_utils as utils
from mask2former import add_maskformer2_config
from solar_inference import install_chunked_instance_inference
from solar_evaluation import mask_rle, pq_counts, summarize_counts

OUT = Path('/kaggle/working/snake_best_submission_20261018')
OUT.mkdir(exist_ok=True)
LIMIT_SECONDS = 3600
torch.set_grad_enabled(False)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
provenance = json.loads((ASSETS / 'provenance.json').read_text())
weight = ASSETS / 'model_best.pth'
assert hashlib.sha256(weight.read_bytes()).hexdigest() == provenance['checkpoint_sha256']
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.set_new_allowed(True)
cfg.merge_from_file(str(ASSETS / 'config.yaml'))
cfg.MODEL.DEVICE = 'cuda'
cfg.DATASETS.TRAIN = ('solar_calibration_metadata',)
cfg.DATASETS.TEST = ()
model = build_model(cfg)
checkpoint = torch.load(weight, map_location='cpu', weights_only=True)
assert checkpoint['iteration'] == 36999
model.load_state_dict(checkpoint['model'], strict=True)
del checkpoint
gc.collect()
model.requires_grad_(False).eval()
install_chunked_instance_inference(model, .15)
records = json.loads((ASSETS / 'validation_records.json').read_text())
dates = sorted({r['file_name'][:8] for r in records},
               key=lambda x: hashlib.sha256(('calibration-v1-' + x).encode()).hexdigest())
cal_dates = set(dates[:len(dates)//2])
part = {r['file_name']: ('calibration' if r['file_name'][:8] in cal_dates else 'holdout') for r in records}
assert {r['file_name'][:8] for r in records if part[r['file_name']]=='calibration'}.isdisjoint(
       {r['file_name'][:8] for r in records if part[r['file_name']]=='holdout'})
(OUT/'split.json').write_text(json.dumps(part, indent=2))
root = next(ASSETS.rglob('test_images')).parent.parent
train_dir, test_dir = root/'train/train_images', root/'test/test_images'
raw = json.loads((root/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
gt = defaultdict(list)
for a in raw['annotations']:
    gt[a['image_id']].append(a)
VIEWS = ('native','hflip','vflip','rot180')

def encode(mask):
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    r['counts'] = r['counts'].decode('ascii')
    return r

def candidates(path, view):
    assert time.monotonic()-RUN_STARTED < LIMIT_SECONDS, 'Calibration time cap reached'
    assert not model.training and not torch.is_grad_enabled()
    image = utils.read_image(str(path), format=cfg.INPUT.FORMAT)
    assert image.shape[:2] == (2048,2048)
    if view in ('hflip','rot180'): image = image[:,::-1]
    if view in ('vflip','rot180'): image = image[::-1]
    x = torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1)),dtype=torch.float32)
    with torch.inference_mode():
        instances = model([{'image':x,'height':2048,'width':2048}])[0]['instances']
    result = []
    for idx in instances.scores.argsort(descending=True).tolist():
        mask = instances.pred_masks[idx].numpy()
        if view in ('hflip','rot180'): mask = mask[:,::-1]
        if view in ('vflip','rot180'): mask = mask[::-1]
        if mask.any(): result.append(dict(rle=encode(mask),score=float(instances.scores[idx]),view=view))
    del x, instances
    return result

def collect(directory, names, views, filename):
    path = OUT/filename
    cache = json.loads(path.read_text()) if path.exists() else {}
    started=time.monotonic()
    for i,name in enumerate(names,1):
        cache.setdefault(name,{})
        for view in views:
            if view not in cache[name]:
                cache[name][view] = candidates(directory/name,view)
                temp=path.with_suffix('.tmp');temp.write_text(json.dumps(cache));temp.replace(path)
        if i%10==0 or i==len(names):
            print('CANDIDATES',filename,i,'/',len(names),'elapsed_s',round(time.monotonic()-started),flush=True)
    return cache

def consensus(per_view):
    pool=[p for v in VIEWS for p in per_view[v]]
    if not pool: return []
    iou=mu.iou([p['rle'] for p in pool],[p['rle'] for p in pool],[0]*len(pool))
    order=sorted(range(len(pool)),key=lambda i:pool[i]['score'],reverse=True)
    used=set();result=[]
    for i in order:
        if i in used: continue
        matched=[i]
        for v in VIEWS:
            if v==pool[i]['view']: continue
            options=[j for j in order if j not in used and pool[j]['view']==v and iou[i,j]>.5]
            if options: matched.append(max(options,key=lambda j:iou[i,j]))
        used.update(matched)
        if len(matched)>=2:
            result.append(dict(rle=pool[i]['rle'],score=sum(pool[j]['score'] for j in matched)/len(matched),
                               support=len(matched)))
    return result

def disjoint(pool):
    occupied=np.zeros((2048,2048),bool); result=[]
    for p in sorted(pool,key=lambda p:p['score'],reverse=True):
        mask=mu.decode(p['rle']).astype(bool)
        mask &= ~occupied
        if not mask.any(): continue
        occupied |= mask
        rle=encode(mask)
        result.append(dict(rle=rle,score=p['score'],area=int(mu.area(rle))))
    return result





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

import datetime, traceback

SOURCE = Path('/kaggle/working/finetune_results/model_best_finetune.pth')
SOURCE_SHA = '6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1'
DELTA = Path('/kaggle/working/snake2048_resume_20261018/snake_step1200.pth')
DELTA_SHA = 'ca87077f721356018594b3f0fe4cf3bea562991c5f3e484347782cfa7946ad75'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
assert hashlib.sha256(DELTA.read_bytes()).hexdigest() == DELTA_SHA
base = torch.load(SOURCE, map_location='cpu', weights_only=True)
model.load_state_dict(base['model'], strict=True)
cp = torch.load(DELTA, map_location='cpu', weights_only=False)
assert cp['arm'] == 'snake' and cp['effective_updates'] == 1200 and cp['input_size'] == 2048
assert cp['source_checkpoint_sha256'] == SOURCE_SHA
saved_config = cp['adapter_config']
config = attach_feature_adapters(model, **{k:saved_config[k] for k in ('mode','mask_features','scales','bottleneck','kernel_size')})
assert config == saved_config
assert cp['delta']._metadata['sem_seg_head']['version'] == 2
expected_missing = set(model.state_dict()) - set(cp['delta'])
loaded = model.load_state_dict(cp['delta'], strict=False)
assert not loaded.unexpected_keys and set(loaded.missing_keys) == expected_missing
assert all(torch.equal(model.state_dict()[k].detach().cpu(), v) for k,v in cp['delta'].items())
assert all(torch.equal(v.detach().cpu(), base['model'][k]) for k,v in model.state_dict().items() if k not in cp['delta'])
del base, cp
gc.collect()
model.requires_grad_(False).eval()
assert all(not m.training for m in model.modules())

def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)

def emit(marker, data):
    print(marker, json.dumps(data), flush=True)
    with (OUT/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(marker=marker, data=data, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')

def pack_outputs():
    archive = OUT.with_suffix('.zip')
    temp = archive.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.suffix != '.tmp': z.write(p,p.name)
    temp.replace(archive)

summary = dict(experiment='best_snake1200_test_inference', status='preflight',
               base_checkpoint_sha256=SOURCE_SHA, delta_checkpoint_sha256=DELTA_SHA,
               adapter_config=config, selected_step=1200, local_calibration_pq=.46036319475580045,
               previous_best_public_score=.36, new_public_score=None, submitted=False,
               input_size=2048, test_policy='whole native2048; FP32; native/hflip/vflip/rot180',
               postprocessing='>=2 views IoU>.5; mean score>=.9; area>=256; greedy nonoverlap before filtering',
               training_performed=False, free_gpu_only=True, wall_cap_seconds=LIMIT_SECONDS,
               device=torch.cuda.get_device_name(0), started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
write_json(OUT/'provenance.json', summary)
try:
    reference = json.loads(Path('/kaggle/working/snake2048_resume_20261018/snake_step1200_candidates.json').read_text())
    probe_name = sorted(reference)[0]
    actual = candidates(train_dir/probe_name, 'native')
    expected = reference[probe_name]['native']
    assert len(actual) == len(expected)
    assert all(a['rle']==b['rle'] and a['view']==b['view'] and abs(a['score']-b['score'])<1e-6 for a,b in zip(actual,expected))
    summary['preflight'] = dict(status='PASS', restored_prediction_cache_match=True, image=probe_name, candidates=len(actual))
    emit('SNAKE_TEST_PREFLIGHT_PASS',summary['preflight'])
    del reference, actual, expected
    names = sorted(p.name for p in test_dir.glob('*.jpeg'))
    assert len(names) == 180
    cache_path = OUT/'test_candidates.json'
    if cache_path.exists():
        prior=json.loads((OUT/'cache_identity.json').read_text())
        assert prior == dict(base_sha256=SOURCE_SHA,delta_sha256=DELTA_SHA,views=list(VIEWS))
        cache=json.loads(cache_path.read_text())
    else: cache={}
    write_json(OUT/'cache_identity.json',dict(base_sha256=SOURCE_SHA,delta_sha256=DELTA_SHA,views=list(VIEWS)))
    assert set(cache) <= set(names)
    summary['status']='inference'
    write_json(OUT/'summary.json',summary)
    for i,name in enumerate(names,1):
        cache.setdefault(name,{})
        for view in VIEWS:
            if view not in cache[name]:
                cache[name][view]=candidates(test_dir/name,view)
                write_json(cache_path,cache)
        if i%10==0 or i==len(names):
            summary.update(processed_test_images=i,elapsed_seconds=round(time.monotonic()-RUN_STARTED))
            write_json(OUT/'summary.json',summary)
            emit('SNAKE_TEST_PROGRESS',dict(images=i,total=180,elapsed_seconds=summary['elapsed_seconds']))
            pack_outputs()
    assert set(cache)==set(names) and all(set(v)==set(VIEWS) for v in cache.values())
    rows=[];image_counts={};predictions={}
    for i,name in enumerate(names,1):
        selected=[p for p in disjoint(consensus(cache[name])) if p['score']>=.9 and p['area']>=256]
        predictions[name]=selected;image_counts[name]=len(selected)
        rows.extend((Path(name).stem+'_'+str(k),p['rle']['counts']) for k,p in enumerate(selected,1))
        if i%30==0:emit('SNAKE_TEST_POSTPROCESS',dict(images=i,total=180))
    assert rows and len({r[0] for r in rows})==len(rows)
    with (OUT/'submission_snake1200.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['filament_id','segmentation_rle']);w.writerows(rows)
    write_json(OUT/'test_predictions.json',predictions)
    write_json(OUT/'test_names.json',names)
    summary.update(status='complete', processed_test_images=180,predicted_instances=len(rows),
                   empty_test_images=sum(n==0 for n in image_counts.values()),test_image_counts=image_counts,
                   submission_sha256=hashlib.sha256((OUT/'submission_snake1200.csv').read_bytes()).hexdigest(),
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
except Exception as error:
    summary.update(status='error',error_type=type(error).__name__,error=str(error))
    traceback.print_exc()
    emit('SNAKE_TEST_ERROR',dict(type=type(error).__name__,message=str(error)))
finally:
    summary.update(elapsed_seconds=round(time.monotonic()-RUN_STARTED),completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(OUT/'summary.json',summary)
    emit('SNAKE_TEST_COMPLETE',summary)
    gc.collect();torch.cuda.empty_cache();pack_outputs()
    print('SNAKE_TEST_ARCHIVE_READY',str(OUT.with_suffix('.zip')),flush=True)
