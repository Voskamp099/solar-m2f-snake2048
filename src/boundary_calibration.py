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
