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

# Recover interrupted final evaluation only. No optimizer updates or retraining.
from datetime import datetime,timezone
CUTOFF=datetime.fromisoformat('2026-10-16T23:35:00+00:00').timestamp()
assert time.time()<CUTOFF-1800
OUT=Path('/kaggle/working/context1536_results');assert OUT.is_dir()
LIMIT_SECONDS=8*3600
best_path=OUT/'model_best_context.pth';last_path=OUT/'model_last_context.pth'
best=torch.load(best_path,map_location='cpu',weights_only=True)
assert best['source_sha256']=='6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1'
assert best['context_step']==200 and best['crop_size']==1536
model.load_state_dict(best['model'],strict=True)
assert all(torch.equal(v.detach().cpu(),best['model'][k]) for k,v in model.state_dict().items())
best_metric=best['calibration'];best_step=best['context_step'];del best
last=torch.load(last_path,map_location='cpu',weights_only=True)
assert last['context_step']==600
assert last['early_stopping_count']==2
assert {'optimizer','scaler','random_state','torch_rng','cuda_rng'}<=last.keys()
training_steps=last['context_step'];train_elapsed=last['elapsed_seconds'];del last
history=json.loads((OUT/'training_history.json').read_text())
assert history[-1]['step']==600
print('CONTEXT_RECOVERY_STRICT_LOAD_PASS',json.dumps(dict(best_step=best_step,training_steps=training_steps,optimizer_updates_in_recovery=0)),flush=True)
model.requires_grad_(False).eval()

def recover_context_eval(split,step):
    assert time.time()<CUTOFF-1200
    names=sorted(n for n in part if part[n]==split)
    cache=collect(train_dir,names,VIEWS,f'{split}_step{step}_candidates.json')
    predictions={name:[p['rle'] for p in disjoint(consensus(cache[name])) if p['score']>=.9 and p['area']>=256] for name in names}
    counts=[]
    for record in records:
        if part[record['file_name']]!=split:continue
        truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[record['source_image_id']]];prediction=predictions[record['file_name']]
        iou=mu.iou(truth,prediction,[0]*len(prediction)) if prediction else np.zeros((len(truth),0))
        counts.append(pq_counts(iou))
    return summarize_counts(counts)

summary_path=OUT/'context_summary.json'
if summary_path.exists():
    summary=json.loads(summary_path.read_text());print('CONTEXT_ALREADY_COMPLETE',json.dumps(summary),flush=True)
else:
    summary=dict(crop=1536,lr=2e-6,training_steps=training_steps,best_step=best_step,best_calibration=best_metric,
                 baseline_calibration_pq=.4563696594893772,baseline_holdout_pq=.40760711778568315,
                 holdout_accepted=False,stop_reason='early_stop',source_sha256='6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1',
                 peak_vram_gib=10.61,training_elapsed_seconds=train_elapsed,recovery_performed=True,recovery_optimizer_updates=0)
    holdout=recover_context_eval('holdout',best_step);summary['holdout']=holdout
    summary['holdout_accepted']=holdout['PQ']>.40760711778568315+.003
    print('CONTEXT_HOLDOUT',json.dumps(holdout),flush=True)
    if summary['holdout_accepted']:
        assert time.time()<CUTOFF-1800
        names=sorted(p.name for p in test_dir.glob('*.jpeg'));assert len(names)==180
        cache=collect(test_dir,names,VIEWS,'test_candidates.json');rows=[]
        for name in names:
            prediction=[p['rle'] for p in disjoint(consensus(cache[name])) if p['score']>=.9 and p['area']>=256]
            rows.extend((Path(name).stem+'_'+str(i),p['counts']) for i,p in enumerate(prediction,1))
        with (OUT/'submission_context.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle']);writer.writerows(rows)
        summary.update(test_images=180,instances=len(rows))
    summary.update(recovery_elapsed_seconds=round(time.monotonic()-RUN_STARTED),completed_at_utc=datetime.now(timezone.utc).isoformat())
    summary_path.write_text(json.dumps(summary,indent=2))
with zipfile.ZipFile('/kaggle/working/context1536_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in OUT.iterdir():archive.write(path,path.name)
print('CONTEXT_COMPLETE',json.dumps(summary),flush=True)
