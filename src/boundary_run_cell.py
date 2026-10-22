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

OUT = Path('/kaggle/working/calibration_results')
OUT.mkdir(exist_ok=True)
LIMIT_SECONDS = 5.5 * 3600
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

# Fixed-mask boundary experiment: no training; one forward reused for three contours.
from datetime import datetime, timezone
import math
import torch.nn.functional as F
from detectron2.structures import ImageList

CUTOFF = datetime.fromisoformat('2026-10-16T23:35:00+00:00').timestamp()
assert time.time() < CUTOFF, 'Current quota cycle cutoff reached'
BOUNDARY_LEVELS=(.35,.5,.65)
OUT=Path('/kaggle/working/boundary_results')
OUT.mkdir(exist_ok=True)
LIMIT_SECONDS=8*3600
source=Path('/kaggle/working/finetune_results/model_best_finetune.pth')
assert source.exists(), 'Previous best checkpoint must remain in the preserved working directory'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1'
state=torch.load(source,map_location='cpu',weights_only=True)
assert state['finetune_step']==300
model.load_state_dict(state['model'],strict=True)
del state
model.requires_grad_(False).eval()

@torch.inference_mode()
def boundary_candidates(path,view,levels):
    assert time.time()<CUTOFF, 'Current quota cycle cutoff reached'
    image=utils.read_image(str(path),format=cfg.INPUT.FORMAT)
    assert image.shape[:2]==(2048,2048)
    if view in ('hflip','rot180'):image=image[:,::-1]
    if view in ('vflip','rot180'):image=image[::-1]
    tensor=torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1)),dtype=torch.float32).to(model.device)
    images=ImageList.from_tensors([(tensor-model.pixel_mean)/model.pixel_std],model.size_divisibility)
    predictions=model.sem_seg_head(model.backbone(images.tensor))
    cls=predictions['pred_logits'][0];masks=predictions['pred_masks'][0]
    scores,indices=cls.softmax(-1)[:,:-1].flatten().topk(model.test_topk_per_image,sorted=False)
    queries=indices//(cls.shape[-1]-1)
    valid=scores>=.15;scores,queries=scores[valid],queries[valid]
    result={str(p):[] for p in levels}
    for start in range(0,len(queries),4):
        logits=F.interpolate(masks[queries[start:start+4]][None],size=images.tensor.shape[-2:],mode='bilinear',align_corners=False)[0]
        logits=logits[:,:images.image_sizes[0][0],:images.image_sizes[0][1]]
        logits=F.interpolate(logits[None],size=(2048,2048),mode='bilinear',align_corners=False)[0]
        canonical=logits>0
        quality=(logits.sigmoid().flatten(1)*canonical.flatten(1)).sum(1)/(canonical.flatten(1).sum(1)+1e-6)
        final_scores=scores[start:start+4]*quality
        for level in levels:
            binary=(logits>math.log(level/(1-level))).cpu().numpy()
            for mask,score in zip(binary,final_scores.cpu().tolist()):
                if score<.15 or not mask.any():continue
                if view in ('hflip','rot180'):mask=mask[:,::-1]
                if view in ('vflip','rot180'):mask=mask[::-1]
                result[str(level)].append(dict(rle=encode(mask),score=score,view=view))
    for key in result:result[key].sort(key=lambda p:p['score'],reverse=True)
    return result

def collect_boundary(directory,names,levels,filename):
    path=OUT/filename
    cache=json.loads(path.read_text()) if path.exists() else {}
    for i,name in enumerate(names,1):
        cache.setdefault(name,{})
        for view in VIEWS:
            if view not in cache[name]:
                cache[name][view]=boundary_candidates(directory/name,view,levels)
                temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(cache));temporary.replace(path)
        if i%10==0 or i==len(names):print('BOUNDARY_CACHE',filename,i,'/',len(names),flush=True)
    return cache

def boundary_predictions(cache,level):
    return {name:[p['rle'] for p in disjoint(consensus({v:cache[name][v][str(level)] for v in VIEWS}))
                  if p['score']>=.9 and p['area']>=256] for name in cache}

def boundary_score(predictions,split):
    counts=[]
    for r in records:
        if part[r['file_name']]!=split:continue
        truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[r['source_image_id']]]
        pred=predictions[r['file_name']]
        iou=mu.iou(truth,pred,[0]*len(pred)) if pred else np.zeros((len(truth),0))
        counts.append(pq_counts(iou))
    return summarize_counts(counts)

print('BOUNDARY_START',json.dumps(dict(levels=BOUNDARY_LEVELS,checkpoint=str(source),training=False,cutoff_utc='2026-10-16T23:35:00Z')),flush=True)
cal_names=sorted(n for n in part if part[n]=='calibration')
cal_cache=collect_boundary(train_dir,cal_names,BOUNDARY_LEVELS,'calibration_candidates.json')
grid=[dict(level=level,calibration=boundary_score(boundary_predictions(cal_cache,level),'calibration')) for level in BOUNDARY_LEVELS]
canonical=next(g for g in grid if g['level']==.5)
assert abs(canonical['calibration']['PQ']-.4563696594893772)<1e-7, canonical
chosen=max(grid,key=lambda x:x['calibration']['PQ'])
summary=dict(grid=grid,chosen=chosen,holdout_accepted=False,baseline_public_score=.36,
             baseline_calibration_pq=.4563696594893772,baseline_holdout_pq=.40760711778568315,
             trained=False,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
(OUT/'boundary_summary.json').write_text(json.dumps(summary,indent=2))
print('BOUNDARY_CALIBRATION',json.dumps(summary),flush=True)
if chosen['level']!=.5 and chosen['calibration']['PQ']>.4563696594893772+.003:
    hold_names=sorted(n for n in part if part[n]=='holdout')
    hold_cache=collect_boundary(train_dir,hold_names,(chosen['level'],),'holdout_candidates.json')
    hold=boundary_score(boundary_predictions(hold_cache,chosen['level']),'holdout')
    summary['holdout']=hold
    summary['holdout_accepted']=hold['PQ']>.40760711778568315+.003
    (OUT/'boundary_summary.json').write_text(json.dumps(summary,indent=2))
    print('BOUNDARY_HOLDOUT',json.dumps(hold),flush=True)
    if summary['holdout_accepted']:
        test_names=sorted(p.name for p in test_dir.glob('*.jpeg'));assert len(test_names)==180
        test_cache=collect_boundary(test_dir,test_names,(chosen['level'],),'test_candidates.json')
        predictions=boundary_predictions(test_cache,chosen['level']);rows=[]
        for name in test_names:
            rows.extend((Path(name).stem+'_'+str(i),r['counts']) for i,r in enumerate(predictions[name],1))
        with (OUT/'submission_boundary.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle']);writer.writerows(rows)
        summary.update(test_images=180,instances=len(rows))
summary.update(elapsed_seconds=round(time.monotonic()-RUN_STARTED),completed_at_utc=datetime.now(timezone.utc).isoformat())
(OUT/'boundary_summary.json').write_text(json.dumps(summary,indent=2))
with zipfile.ZipFile('/kaggle/working/boundary_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for p in OUT.iterdir():archive.write(p,p.name)
print('BOUNDARY_COMPLETE',json.dumps(summary),flush=True)
