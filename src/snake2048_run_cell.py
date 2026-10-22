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

OUT = Path('/kaggle/working/snake2048_controls_20261018')
OUT.mkdir(exist_ok=True)
LIMIT_SECONDS = 5400
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

import copy, random, warnings, traceback, datetime
from collections import OrderedDict
from detectron2.structures import Instances
warnings.filterwarnings('ignore', category=FutureWarning)
EXPERIMENT_START=time.monotonic()
# Conservative allowance: <=85min GPU wall, <=170 GPU-hours minutes if both T4s charged.
WALL_CAP_SECONDS=5100
TRAIN_SEED=20261018
INIT_SEED=20261017
SOURCE=Path('/kaggle/working/finetune_results/model_best_finetune.pth')
SOURCE_SHA='6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
base=torch.load(SOURCE,map_location='cpu',weights_only=True)
base_weights=base['model'];del base
model.load_state_dict(base_weights,strict=True)
training_root=Path('/kaggle/working/training_images')
if not (training_root/'training_manifest.json').exists():
    archives=list(Path('/kaggle/input').rglob('training_images.bin'));assert len(archives)==1
    training_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archives[0]) as z:z.extractall(training_root)
manifest=json.loads((training_root/'training_manifest.json').read_text())
train_names=manifest['train_images'];assert len(train_names)==566 and not set(train_names)&set(part)
train_records=defaultdict(list)
for image in raw['images']:
    if image['file_name'] in set(train_names):train_records[image['file_name']].append(image['id'])
MASK_PREFIX='sem_seg_head.predictor.mask_embed.'
assert any(n.startswith(MASK_PREFIX) for n,_ in model.named_parameters())
def emit(marker,data):
    line=marker+' '+json.dumps(data,ensure_ascii=False)
    print(line,flush=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(dict(marker=marker,data=data,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def pack_outputs():
    dst=OUT.with_suffix('.zip');tmp=dst.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file() and f.suffix!='.tmp':z.write(f,f.name)
    tmp.replace(dst)
def cleanup():
    h=model.sem_seg_head
    if hasattr(h,'_snake_feature_handle'):
        h._snake_feature_handle.remove();del h._snake_feature_handle
        del h.snake_mask_adapter,h.snake_scale_adapters,h._snake_adapter_config
    model.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.empty_cache()
def create_arm(mode):
    cleanup();model.load_state_dict(base_weights,strict=True);model.requires_grad_(False)
    torch.manual_seed(INIT_SEED)
    config=dict(mode='none',mask_features=False,scales=[])
    if mode!='head_only':config=attach_feature_adapters(model,mode=mode,mask_features=True,scales=(),bottleneck=32,kernel_size=9)
    model.sem_seg_head.predictor.mask_embed.requires_grad_(True)
    return config
def optimizer_for_arm():
    groups=[]
    for label,lr,predicate in [('mask_embed',1e-6,lambda n:n.startswith(MASK_PREFIX)),('gate',1e-3,lambda n:n.endswith('.alpha')),('adapter',1e-4,lambda n:'.snake_' in n and not n.endswith('.alpha'))]:
        params=[p for n,p in model.named_parameters() if p.requires_grad and predicate(n)]
        if params:groups.append(dict(params=params,lr=lr,group_name=label))
    params=[p for p in model.parameters() if p.requires_grad]
    assert len(params)==sum(len(g['params']) for g in groups)
    return torch.optim.AdamW(groups,weight_decay=.01),params
schedule=[];schedule_rng=random.Random(TRAIN_SEED)
for step in range(600):
    name=schedule_rng.choice(train_names);sid=schedule_rng.choice(train_records[name])
    schedule.append(dict(file_name=name,source_image_id=sid,hflip=schedule_rng.random()<.5,vflip=schedule_rng.random()<.5))
write_json(OUT/'training_schedule.json',schedule)
def training_item(spec):
    image=utils.read_image(str(training_root/'train_images'/spec['file_name']),format=cfg.INPUT.FORMAT)
    assert image.shape[:2]==(2048,2048)
    masks=[mu.decode(mask_rle(a['segmentation'],2048,2048)) for a in gt[spec['source_image_id']]]
    masks=np.stack(masks) if masks else np.zeros((0,2048,2048),np.uint8)
    if spec['hflip']:image=image[:,::-1];masks=masks[:,:,::-1]
    if spec['vflip']:image=image[::-1];masks=masks[:,::-1,:]
    target=Instances((2048,2048));target.gt_classes=torch.zeros(len(masks),dtype=torch.int64)
    target.gt_masks=torch.from_numpy(np.ascontiguousarray(masks)).bool()
    return dict(image=torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1))),instances=target,height=2048,width=2048)
def save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history,metrics=None):
    state=model.state_dict()
    delta=OrderedDict((k,v.detach().cpu()) for k,v in state.items() if '.snake_' in k or k.startswith(MASK_PREFIX))
    delta._metadata=copy.deepcopy(state._metadata)
    data=dict(format='mask_embed_and_adapter_delta_v1',delta=delta,adapter_config=config,arm=mode,
              source_checkpoint_sha256=SOURCE_SHA,source_checkpoint=str(SOURCE),effective_updates=updates,attempts=attempts,
              optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
              torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history,metrics=metrics,
              schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),input_size=2048,
              trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
              recovery='Load original base, attach adapter_config when mode!=none, load delta strict=False with exact allowed missing keys, restore optimizer/scaler/RNG, resume schedule[effective_updates:].')
    path=OUT/(mode+'_last.pth');tmp=path.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(path)
    write_json(OUT/(mode+'_history.json'),history)
def verify_frozen():
    state=model.state_dict()
    assert all(torch.equal(v.detach().cpu(),base_weights[k]) for k,v in state.items() if '.snake_' not in k and not k.startswith(MASK_PREFIX)), 'Unexpected frozen-base mutation'
    return dict(frozen_parameters_identical=True,mask_embed_changed=any(not torch.equal(v.detach().cpu(),base_weights[k]) for k,v in state.items() if k.startswith(MASK_PREFIX)),
                offset_nonzero={n:int(torch.count_nonzero(p.detach()).item()) for n,p in model.named_parameters() if '.spatial.offset.weight' in n},
                gates={n:float(p.detach()) for n,p in model.named_parameters() if n.endswith('.alpha')})
def evaluate_arm(mode):
    model.eval();cache={};selected=sorted(n for n in part if part[n]=='calibration');assert len(selected)==70
    with torch.inference_mode():
        for idx,name in enumerate(selected,1):
            assert time.monotonic()-EXPERIMENT_START<WALL_CAP_SECONDS-120,'Evaluation wall cap reached'
            cache[name]={v:candidates(train_dir/name,v) for v in VIEWS}
            write_json(OUT/(mode+'_candidates.json'),cache)
            if idx%10==0:emit('FULL2048_EVAL_PROGRESS',dict(arm=mode,images=idx,total=70,elapsed_seconds=round(time.monotonic()-EXPERIMENT_START)))
    pred={name:[p['rle'] for p in disjoint(consensus(views)) if p['score']>=.9 and p['area']>=256] for name,views in cache.items()}
    counts=[]
    for r in records:
        name=r['file_name']
        if name not in pred:continue
        truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[r['source_image_id']]]
        ps=pred[name];mat=mu.iou(truth,ps,[0]*len(ps)) if ps else np.zeros((len(truth),0))
        counts.append(pq_counts(mat))
    return summarize_counts(counts)
summary=dict(experiment='shared_mask_adapter_plus_mask_embed_full2048',source_sha256=SOURCE_SHA,input_size=2048,
             input_policy='whole native2048, no crop/no downsample, matched H/V flips',planned_effective_updates=600,
             arms=[],preflights=[],baseline_calibration_pq=.4563696594893772,baseline_public_score=.36,
             mask_embed_lr=1e-6,adapter_lr=1e-4,gate_lr=1e-3,weight_decay=.01,grad_clip=1.,
             train_seed=TRAIN_SEED,init_seed=INIT_SEED,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
             calibration_images=70,training_images=566,review_evaluated=False,submitted=False,
             evaluation='full2048FP32,4views,2-viewIoU>.5 consensus,mean score>=.90,area>=256,greedy nonoverlap',
             wall_cap_seconds=WALL_CAP_SECONDS,conservative_max_two_device_quota_hours=(WALL_CAP_SECONDS+300)/1800,
             status='preflight_running',device=torch.cuda.get_device_name(0),visible_devices=torch.cuda.device_count())
write_json(OUT/'summary.json',summary);emit('FULL2048_START',summary)
try:
    # Compare actual full-image inference at initialization, before any updates.
    model.eval();torch.set_grad_enabled(False)
    identity_name=sorted(n for n in part if part[n]=='calibration')[0]
    with torch.inference_mode():before=candidates(train_dir/identity_name,'native')
    create_arm('snake');model.eval()
    with torch.inference_mode():after=candidates(train_dir/identity_name,'native')
    assert before==after,'Zero-gate baseline identity failed'
    summary['actual_zero_gate_full2048_identity']=True
    # Two batches per arm, including densest training annotation; Adam state allocated.
    dense_name,dense_sid=max(((name,sid) for name in train_names for sid in train_records[name]),key=lambda item:len(gt[item[1]]))
    probe_specs=[schedule[0],dict(file_name=dense_name,source_image_id=dense_sid,hflip=False,vflip=False)]
    for mode in ('head_only','fixed_snake','snake'):
        config=create_arm(mode);model.train();model.backbone.eval()
        if mode!='head_only':model.sem_seg_head.snake_mask_adapter.alpha.data.fill_(.01)
        optimizer,params=optimizer_for_arm();scaler=torch.amp.GradScaler('cuda',init_scale=64.)
        torch.manual_seed(TRAIN_SEED);torch.cuda.reset_peak_memory_stats();probe_started=time.monotonic();probe_losses=[]
        for spec in probe_specs:
            item=training_item(spec);optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):losses=model([item]);loss=sum(losses.values())
            assert torch.isfinite(loss),'Nonfinite preflight loss'
            with torch.enable_grad():scaler.scale(loss).backward()
            scaler.unscale_(optimizer);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            probe_losses.append(float(loss.detach()));del item,losses,loss
        probe=dict(arm=mode,status='PASS',input_size=2048,max_gt_instances=len(gt[dense_sid]),losses=probe_losses,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                   free_gib=torch.cuda.mem_get_info()[0]/1024**3,probe_optimizer_steps=2,discarded_before_training=True,
                   elapsed_seconds=time.monotonic()-probe_started,trainable_parameters=sum(p.numel() for p in params))
        summary['preflights'].append(probe);write_json(OUT/'summary.json',summary);emit('FULL2048_PREFLIGHT_PASS',probe)
        assert probe['peak_reserved_gib']<.92*torch.cuda.get_device_properties(0).total_memory/1024**3,'Insufficient memory margin'
        del optimizer,scaler,params;cleanup()
    # Same effective update budget for all arms, chosen by preflight throughput only.
    worst=max(p['elapsed_seconds']/2 for p in summary['preflights'])
    available=WALL_CAP_SECONDS-(time.monotonic()-EXPERIMENT_START)-2100
    common_steps=min(600,int(available/(3*worst*1.20))//50*50)
    assert common_steps>=200,'Not enough bounded time for matched three-arm training'
    summary['effective_updates_per_arm']=common_steps;summary['status']='training';emit('FULL2048_MATCHED_BUDGET',dict(effective_updates=common_steps,estimated_worst_seconds_per_update=worst))
    for mode in ('head_only','fixed_snake','snake'):
        config=create_arm(mode);model.train();model.backbone.eval();torch.manual_seed(TRAIN_SEED);np.random.seed(TRAIN_SEED);random.seed(TRAIN_SEED)
        optimizer,params=optimizer_for_arm();scaler=torch.amp.GradScaler('cuda',init_scale=64.)
        updates=0;attempts=0;history=[];arm_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        while updates<common_steps:
            assert time.monotonic()-EXPERIMENT_START<WALL_CAP_SECONDS-900,'Training wall cap reached'
            assert attempts<common_steps+50,'Excessive AMP retries'
            spec=schedule[updates];item=training_item(spec);optimizer.zero_grad(set_to_none=True);attempts+=1
            with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):losses=model([item]);loss=sum(losses.values())
            assert torch.isfinite(loss),'Nonfinite training loss'
            old_scale=scaler.get_scale()
            with torch.enable_grad():scaler.scale(loss).backward()
            scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(optimizer);scaler.update()
            accepted=scaler.get_scale()>=old_scale;updates+=int(accepted)
            entry=dict(attempt=attempts,effective_update=updates,optimizer_updated=bool(accepted),loss=float(loss.detach()),gradient_norm=float(norm),scale=scaler.get_scale(),**spec)
            if mode!='head_only':entry['alpha']=float(model.sem_seg_head.snake_mask_adapter.alpha.detach())
            history.append(entry);del item,losses,loss
            if attempts==1 or attempts%25==0:emit('FULL2048_TRAIN',dict(arm=mode,**entry,elapsed_seconds=round(time.monotonic()-arm_started)))
            if accepted and updates%50==0:save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history)
        training_peak=torch.cuda.max_memory_allocated()/1024**3
        checks=verify_frozen();save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history)
        model.zero_grad(set_to_none=True);metrics=evaluate_arm(mode)
        save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history,metrics)
        result=dict(arm=mode,config=config,effective_updates=updates,attempts=attempts,calibration=metrics,
                    training_peak_allocated_gib=training_peak,trainable_parameters=sum(p.numel() for p in params),
                    integrity=checks,elapsed_seconds=round(time.monotonic()-arm_started))
        summary['arms'].append(result);write_json(OUT/'summary.json',summary);emit('FULL2048_ARM_COMPLETE',result)
        del optimizer,scaler,params;cleanup();pack_outputs()
    summary['status']='complete_three_arms'
except Exception as error:
    summary['status']='error';summary['error_type']=type(error).__name__;summary['error']=str(error);traceback.print_exc()
    emit('FULL2048_ERROR',dict(type=type(error).__name__,message=str(error)))
    if 'optimizer' in globals() and 'history' in globals():
        save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history)
finally:
    summary['elapsed_seconds']=round(time.monotonic()-EXPERIMENT_START)
    summary['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(OUT/'summary.json',summary);emit('FULL2048_COMPLETE',summary)
    model.eval();model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();pack_outputs()
    print('FULL2048_ARCHIVE_READY',str(OUT.with_suffix('.zip')),flush=True)
