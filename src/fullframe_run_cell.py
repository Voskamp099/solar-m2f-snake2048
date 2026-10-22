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

# Whole-frame resize to1536; unfreeze only res5 at one tenth head LR. Never Run All.
from datetime import datetime, timezone
import warnings, copy, math, random
from collections import OrderedDict
import cv2
from detectron2.structures import Instances
warnings.filterwarnings('ignore',category=FutureWarning)
CUTOFF=datetime.fromisoformat('2026-10-16T23:35:00+00:00').timestamp()
assert time.time()+3600<CUTOFF, 'Insufficient time in this free quota cycle'
assert not model.training
OUT=Path('/kaggle/working/fullframe1536_res5_results');OUT.mkdir(exist_ok=True)
SOURCE=Path('/kaggle/working/finetune_results/model_best_finetune.pth')
SOURCE_SHA='6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
warm=torch.load(SOURCE,map_location='cpu',weights_only=True)
model.load_state_dict(warm['model'],strict=True);del warm
CROP_SIZE=1536;BASE_LR=2e-6;MAX_STEPS=1200;EVAL_EVERY=200;PATIENCE=2
TRAIN_SECONDS=min(7200,CUTOFF-time.time()-1200)
EXPERIMENT_STARTED=time.monotonic()
LIMIT_SECONDS=12*3600
archives=list(Path('/kaggle/input').rglob('training_images.bin'));assert len(archives)==1
training_root=Path('/kaggle/working/training_images');training_root.mkdir(exist_ok=True)
if not (training_root/'training_manifest.json').exists():
    with zipfile.ZipFile(archives[0]) as archive:archive.extractall(training_root)
manifest=json.loads((training_root/'training_manifest.json').read_text())
train_names=manifest['train_images'];assert len(train_names)==566 and not set(train_names)&set(part)
assert set(manifest['validation_images'])==set(part)
train_records=defaultdict(list)
for im in raw['images']:
    if im['file_name'] in set(train_names):train_records[im['file_name']].append(im['id'])
assert set(train_records)==set(train_names)
rng=random.Random(20261018)
model.requires_grad_(True);model.backbone.requires_grad_(False)
for name,parameter in model.named_parameters():
    if name.startswith('backbone.res5.'):parameter.requires_grad_(True)
parameters=[p for p in model.parameters() if p.requires_grad]
groups=[]
for stage,scale in [('head',1.),('res5',.1)]:
    for decay in (True,False):
        selected=[p for name,p in model.named_parameters() if p.requires_grad
                  and (name.startswith('backbone.res5.'))==(stage=='res5') and (p.ndim>1)==decay]
        if selected:groups.append(dict(params=selected,weight_decay=.05 if decay else 0.,lr=BASE_LR*scale,lr_scale=scale))
assert sum(p.numel() for group in groups for p in group['params'])==sum(p.numel() for p in parameters)
optimizer=torch.optim.AdamW(groups,lr=BASE_LR)
scaler=torch.amp.GradScaler('cuda',init_scale=256.)

def sample_context():
    name=rng.choice(train_names);sid=rng.choice(train_records[name])
    image=utils.read_image(str(training_root/'train_images'/name),format=cfg.INPUT.FORMAT)
    assert image.shape[:2]==(2048,2048)
    image=cv2.resize(image,(CROP_SIZE,CROP_SIZE),interpolation=cv2.INTER_LINEAR)
    masks=[]
    for a in gt[sid]:
        mask=cv2.resize(mu.decode(mask_rle(a['segmentation'],2048,2048)),(CROP_SIZE,CROP_SIZE),interpolation=cv2.INTER_NEAREST)
        if mask.any():masks.append(mask)
    masks=np.stack(masks) if masks else np.zeros((0,CROP_SIZE,CROP_SIZE),np.uint8)
    if rng.random()<.5:image=image[:,::-1];masks=masks[:,:,::-1]
    if rng.random()<.5:image=image[::-1];masks=masks[:,::-1,:]
    target=Instances((CROP_SIZE,CROP_SIZE));target.gt_classes=torch.zeros(len(masks),dtype=torch.int64)
    target.gt_masks=torch.from_numpy(np.ascontiguousarray(masks)).bool()
    return {'image':torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1))),
            'instances':target,'height':CROP_SIZE,'width':CROP_SIZE},name

def load_context(weights):
    assert set(weights)==set(model.state_dict())
    restored=OrderedDict(weights);restored._metadata=copy.deepcopy(model.state_dict()._metadata)
    return model.load_state_dict(restored,strict=True)

def save_context(path,step,metric):
    state=model.state_dict();weights=OrderedDict((k,v.detach().cpu()) for k,v in state.items())
    weights._metadata=copy.deepcopy(state._metadata)
    output=dict(model=weights,context_step=step,crop_size=CROP_SIZE,frame_mode='full_image_resized',unfrozen_backbone_stage='res5',source_sha256=SOURCE_SHA,
                calibration=metric,early_stopping_count=no_improvement,optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),random_state=rng.getstate(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),best_step=best_step,best_metric=best_metric,
                elapsed_seconds=prior_elapsed+time.monotonic()-EXPERIMENT_STARTED)
    temp=path.with_suffix('.tmp');torch.save(output,temp);temp.replace(path)

def context_eval(split,step):
    assert time.time()<CUTOFF-60
    model.eval()
    names=sorted(n for n in part if part[n]==split)
    cache=collect(train_dir,names,VIEWS,f'{split}_step{step}_candidates.json')
    predictions={name:[p['rle'] for p in disjoint(consensus(cache[name])) if p['score']>=.9 and p['area']>=256] for name in names}
    counts=[]
    for record in records:
        if part[record['file_name']]!=split:continue
        truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[record['source_image_id']]]
        prediction=predictions[record['file_name']]
        iou=mu.iou(truth,prediction,[0]*len(prediction)) if prediction else np.zeros((len(truth),0))
        counts.append(pq_counts(iou))
    return summarize_counts(counts)

no_improvement=0;best_step=0;best_metric=dict(PQ=.4563696594893772)
prior_elapsed=0.;starting_step=0;history=[]
last_path=OUT/'model_last_context.pth';best_path=OUT/'model_best_context.pth'
if last_path.exists():
    state=torch.load(last_path,map_location='cpu',weights_only=True)
    assert state['source_sha256']==SOURCE_SHA and state['crop_size']==CROP_SIZE
    assert state['frame_mode']=='full_image_resized' and state['unfrozen_backbone_stage']=='res5'
    load_context(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
    rng.setstate(state['random_state']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    starting_step=state['context_step'];no_improvement=state['early_stopping_count']
    best_step=state['best_step'];best_metric=state['best_metric'];prior_elapsed=state['elapsed_seconds']
    history=json.loads((OUT/'training_history.json').read_text()) if (OUT/'training_history.json').exists() else []
    del state
    print('FULLFRAME_RESUME',starting_step,flush=True)
else:
    # Dry preflight computes gradients but performs no optimizer update.
    rng_state=rng.getstate();torch_state=torch.get_rng_state();cuda_state=torch.cuda.get_rng_state_all()
    model.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
    record,name=sample_context()
    with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):
        losses=model([record]);loss=sum(losses.values())
    assert torch.isfinite(loss)
    with torch.enable_grad():scaler.scale(loss).backward()
    preflight=dict(crop=CROP_SIZE,optimizer_updates=0,loss=float(loss),peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
    assert preflight['peak_vram_gib']<14.,preflight
    optimizer.zero_grad(set_to_none=True);del loss,losses,record
    rng.setstate(rng_state);torch.set_rng_state(torch_state);torch.cuda.set_rng_state_all(cuda_state)
    torch.cuda.empty_cache();model.eval()
    (OUT/'preflight.json').write_text(json.dumps(preflight,indent=2));print('FULLFRAME_PREFLIGHT_PASS',json.dumps(preflight),flush=True)
    save_context(best_path,0,best_metric)
print('FULLFRAME_TRAIN_START',json.dumps(dict(crop=CROP_SIZE,lr=BASE_LR,max_steps=MAX_STEPS,evaluate_every=EVAL_EVERY,
       baseline_calibration=best_metric,selection='fixed four-view .9/256 calibration PQ',wall_cap=TRAIN_SECONDS,
       trainable_parameters=sum(p.numel() for p in parameters))),flush=True)
stop_reason='max_steps';training_steps=starting_step
try:
    for step in range(starting_step+1,MAX_STEPS+1):
        if time.time()>CUTOFF-1200 or prior_elapsed+time.monotonic()-EXPERIMENT_STARTED>TRAIN_SECONDS:
            stop_reason='cycle_or_wall_cutoff';break
        model.train();optimizer.zero_grad(set_to_none=True);record,name=sample_context()
        with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):
            losses=model([record]);loss=sum(losses.values())
        assert torch.isfinite(loss)
        with torch.enable_grad():scaler.scale(loss).backward()
        scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(parameters,.1)
        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        updated=scaler.get_scale()>=old_scale
        lr=BASE_LR*(.2+.8*(1+math.cos(math.pi*step/MAX_STEPS))/2)
        for group in optimizer.param_groups:group['lr']=lr*group['lr_scale']
        training_steps=step
        history.append(dict(step=step,loss=float(loss),lr=lr,optimizer_updated=updated,
                            grad_norm=float(norm) if torch.isfinite(norm) else None))
        if step in (1,5,10) or step%50==0:
            print('FULLFRAME_TRAIN',step,'loss',round(float(loss),4),'elapsed_s',round(time.monotonic()-EXPERIMENT_STARTED),
                  'peak_vram_gib',round(torch.cuda.max_memory_allocated()/1024**3,3),flush=True)
        del loss,losses,record
        if step%50==0:
            save_context(last_path,step,None);(OUT/'training_history.json').write_text(json.dumps(history))
        if step%EVAL_EVERY==0:
            metric=context_eval('calibration',step);history[-1]['calibration']=metric
            print('FULLFRAME_CHECKPOINT',step,json.dumps(metric),flush=True)
            no_improvement=0 if metric['PQ']>best_metric['PQ']+.0005 else no_improvement+1
            if metric['PQ']>best_metric['PQ']:
                best_metric=metric;best_step=step;save_context(best_path,step,metric)
            save_context(last_path,step,metric);(OUT/'training_history.json').write_text(json.dumps(history))
            if no_improvement>=PATIENCE:stop_reason='early_stop';break
except BaseException:
    save_context(last_path,training_steps,None);(OUT/'training_history.json').write_text(json.dumps(history));raise
save_context(last_path,training_steps,None);(OUT/'training_history.json').write_text(json.dumps(history))
state=torch.load(best_path,map_location='cpu',weights_only=True);load_context(state['model']);del state
model.requires_grad_(False).eval()
summary=dict(crop=CROP_SIZE,frame_mode='full_image_resized',unfrozen_backbone_stage='res5',backbone_lr=BASE_LR*.1,lr=BASE_LR,training_steps=training_steps,best_step=best_step,best_calibration=best_metric,
             baseline_calibration_pq=.4563696594893772,baseline_holdout_pq=.40760711778568315,
             holdout_accepted=False,stop_reason=stop_reason,source_sha256=SOURCE_SHA,
             peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
if best_step>0 and best_metric['PQ']>.4563696594893772+.003 and time.time()<CUTOFF-1500:
    holdout=context_eval('holdout',best_step);summary['holdout']=holdout
    summary['holdout_accepted']=holdout['PQ']>.40760711778568315+.003
    print('FULLFRAME_HOLDOUT',json.dumps(holdout),flush=True)
    if summary['holdout_accepted'] and time.time()<CUTOFF-1500:
        names=sorted(p.name for p in test_dir.glob('*.jpeg'));assert len(names)==180
        cache=collect(test_dir,names,VIEWS,'test_candidates.json');rows=[]
        for name in names:
            prediction=[p['rle'] for p in disjoint(consensus(cache[name])) if p['score']>=.9 and p['area']>=256]
            rows.extend((Path(name).stem+'_'+str(i),p['counts']) for i,p in enumerate(prediction,1))
        with (OUT/'submission_context.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle']);writer.writerows(rows)
        summary.update(test_images=180,instances=len(rows))
summary.update(elapsed_seconds=round(prior_elapsed+time.monotonic()-EXPERIMENT_STARTED),completed_at_utc=datetime.now(timezone.utc).isoformat())
(OUT/'context_summary.json').write_text(json.dumps(summary,indent=2))
with zipfile.ZipFile('/kaggle/working/fullframe1536_res5_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in OUT.iterdir():archive.write(path,path.name)
print('FULLFRAME_COMPLETE',json.dumps(summary),flush=True)
