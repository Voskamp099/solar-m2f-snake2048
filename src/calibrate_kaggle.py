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

names=sorted(part)
print('SPLIT',dict((p,sum(v==p for v in part.values())) for p in ('calibration','holdout')),flush=True)
cache=collect(train_dir,names,VIEWS,'validation_candidates.json')
pools={'native':{},'flip_consensus':{}}
for i,name in enumerate(names,1):
    pools['native'][name]=disjoint(cache[name]['native'])
    pools['flip_consensus'][name]=disjoint(consensus(cache[name]))
    if i%20==0: print('POSTPROCESS',i,'/',len(names),flush=True)
prepared=[]
for r in records:
    truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[r['source_image_id']]]
    assert len(truth)==r['n_gt']
    for mode in pools:
        p=pools[mode][r['file_name']]
        iou=mu.iou(truth,[x['rle'] for x in p],[0]*len(p)) if p else np.zeros((len(truth),0))
        prepared.append((mode,part[r['file_name']],iou,np.array([x['score'] for x in p]),np.array([x['area'] for x in p])))

def evaluate(mode, threshold, min_area, calibration_only=False):
    rows={'calibration':[],'holdout':[]}
    for m,split,iou,scores,areas in prepared:
        if m==mode and (not calibration_only or split=='calibration'): rows[split].append(pq_counts(iou[:,(scores>=threshold)&(areas>=min_area)]))
    return {**{s:summarize_counts(v) for s,v in rows.items()},'all':summarize_counts(rows['calibration']+rows['holdout'])}

baseline=evaluate('native',.5,0)
assert abs(baseline['all']['PQ']-.31056424399272314)<.002, baseline
grid=[]
for mode,threshold,min_area in itertools.product(pools,[round(x/100,2) for x in range(15,91,5)],[0,64,128,256]):
    grid.append(dict(mode=mode,threshold=threshold,min_area=min_area,metrics=evaluate(mode,threshold,min_area,calibration_only=True)))
grid.sort(key=lambda g:g['metrics']['calibration']['PQ'],reverse=True)
chosen=dict(grid[0])
chosen['metrics']=evaluate(chosen['mode'],chosen['threshold'],chosen['min_area'])
# Only this pre-selected candidate is tested for acceptance on the untouched holdout.
accept=chosen['metrics']['calibration']['PQ']>baseline['calibration']['PQ']+.003 and chosen['metrics']['holdout']['PQ']>baseline['holdout']['PQ']+.003
summary=dict(baseline=baseline,chosen=chosen,holdout_accepted=bool(accept),training_performed=False,
             selection='max calibration PQ; date-disjoint holdout acceptance gain >0.003',
             checkpoint_sha256=provenance['checkpoint_sha256'],grid_size=len(grid),test_predictions_generated=False)
(OUT/'calibration_summary.json').write_text(json.dumps(summary,indent=2))
(OUT/'calibration_grid.json').write_text(json.dumps(grid,indent=2))
print('CALIBRATION_RESULT',json.dumps(summary),flush=True)
if accept:
    test_names=sorted(p.name for p in test_dir.glob('*.jpeg'))
    assert len(test_names)==180
    test_views=VIEWS if chosen['mode']=='flip_consensus' else ('native',)
    tests=collect(test_dir,test_names,test_views,'test_candidates.json')
    rows=[]
    for name in test_names:
        p=tests[name]['native'] if chosen['mode']=='native' else consensus(tests[name])
        p=disjoint(p)
        p=[x for x in p if x['score']>=chosen['threshold'] and x['area']>=chosen['min_area']]
        rows.extend((Path(name).stem+'_'+str(i),x['rle']['counts']) for i,x in enumerate(p,1))
    assert rows and len({r[0] for r in rows})==len(rows)
    with (OUT/'submission_calibrated.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['filament_id','segmentation_rle']);w.writerows(rows)
    summary.update(test_predictions_generated=True,processed_test_images=180,predicted_instances=len(rows))
summary.update(elapsed_seconds=round(time.monotonic()-RUN_STARTED),gpu=torch.cuda.get_device_name(0),
               peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
(OUT/'calibration_summary.json').write_text(json.dumps(summary,indent=2))
with zipfile.ZipFile('/kaggle/working/calibration_results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir(): z.write(p,p.name)
print('CALIBRATION_COMPLETE',json.dumps(summary),flush=True)
