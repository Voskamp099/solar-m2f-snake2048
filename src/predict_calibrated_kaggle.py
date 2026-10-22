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



summary=json.loads('{"baseline": {"calibration": {"PQ": 0.3137662575115845, "SQ": 0.6571119116143566, "RQ": 0.4774928774928775, "tp": 419, "fp": 554, "fn": 363, "iou_sum": 275.3298909664154}, "holdout": {"PQ": 0.3075018277934851, "SQ": 0.6397572040828177, "RQ": 0.48065395095367847, "tp": 441, "fp": 551, "fn": 402, "iou_sum": 282.1329270005226}, "all": {"PQ": 0.31056424399272314, "SQ": 0.6482125790313232, "RQ": 0.479108635097493, "tp": 860, "fp": 1105, "fn": 765, "iou_sum": 557.462817966938}}, "chosen": {"mode": "flip_consensus", "threshold": 0.9, "min_area": 256, "metrics": {"calibration": {"PQ": 0.38339739457112637, "SQ": 0.65576362877591, "RQ": 0.5846579129232896, "tp": 423, "fp": 242, "fn": 359, "iou_sum": 277.38801497220993}, "holdout": {"PQ": 0.34730657114948477, "SQ": 0.646259321860482, "RQ": 0.5374105400130124, "tp": 413, "fp": 281, "fn": 430, "iou_sum": 266.90509992837906}, "all": {"PQ": 0.3648077177617889, "SQ": 0.6510683192590777, "RQ": 0.5603217158176944, "tp": 836, "fp": 523, "fn": 789, "iou_sum": 544.293114900589}}}, "holdout_accepted": true, "training_performed": false, "selection": "max calibration PQ; date-disjoint holdout acceptance gain >0.003", "checkpoint_sha256": "6b1e85267e63a29508501f154db87da65114ad3d826e4af4d5cf80420589854c", "grid_size": 128, "test_predictions_generated": false, "evaluation_device": "local CPU", "gpu_forward_repeated": false, "elapsed_seconds": 99}')
assert summary['holdout_accepted']
chosen=summary['chosen']
assert chosen['mode']=='flip_consensus' and chosen['threshold']==.9 and chosen['min_area']==256
assert summary['checkpoint_sha256']==provenance['checkpoint_sha256']
print('FROZEN_CALIBRATION',json.dumps(summary),flush=True)
test_names=sorted(p.name for p in test_dir.glob('*.jpeg'))
assert len(test_names)==180
tests=collect(test_dir,test_names,VIEWS,'test_candidates.json')
rows=[];image_counts={}
for i,name in enumerate(test_names,1):
    p=disjoint(consensus(tests[name]))
    p=[x for x in p if x['score']>=chosen['threshold'] and x['area']>=chosen['min_area']]
    image_counts[name]=len(p)
    rows.extend((Path(name).stem+'_'+str(k),x['rle']['counts']) for k,x in enumerate(p,1))
    if i%20==0:print('TEST_POSTPROCESS',i,'/180',flush=True)
assert rows and len({r[0] for r in rows})==len(rows)
with (OUT/'submission_calibrated.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['filament_id','segmentation_rle']);w.writerows(rows)
summary.update(test_predictions_generated=True,processed_test_images=180,predicted_instances=len(rows),
               empty_test_images=sum(n==0 for n in image_counts.values()),test_image_counts=image_counts,
               elapsed_seconds=round(time.monotonic()-RUN_STARTED),gpu=torch.cuda.get_device_name(0),
               peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
(OUT/'calibration_summary.json').write_text(json.dumps(summary,indent=2))
with zipfile.ZipFile('/kaggle/working/calibration_test_results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in ['submission_calibrated.csv','calibration_summary.json','test_candidates.json','split.json']:
        z.write(OUT/name,name)
print('CALIBRATED_TEST_COMPLETE',json.dumps(summary),flush=True)
