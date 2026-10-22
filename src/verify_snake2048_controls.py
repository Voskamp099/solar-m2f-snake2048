"""Independently verify a downloaded pilot, without model inference or test GT."""
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'snake2048_controls_20261018'
sys.path.insert(0, str(ROOT / 'bundle'))
from solar_evaluation import mask_rle, pq_counts, summarize_counts

VIEWS = ('native', 'hflip', 'vflip', 'rot180')
tree = ast.parse((ROOT / 'predict_calibrated_kaggle.py').read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('encode', 'consensus', 'disjoint')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'locked_recipe', 'exec'))
summary = json.loads((OUT / 'summary.json').read_text())
records = json.loads((ROOT / 'bundle/validation_records.json').read_text())
part = json.loads((ROOT / 'calibration_results/split.json').read_text())
names = {name for name, split in part.items() if split == 'calibration'}
raw = json.loads(Path('/Users/raelynn/Downloads/solar_project/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
gt = defaultdict(list)
for ann in raw['annotations']:
    gt[ann['image_id']].append(ann)

def evaluate(cache):
    assert set(cache) == names
    predictions = {name: [p['rle'] for p in disjoint(consensus(views)) if p['score'] >= .9 and p['area'] >= 256]
                   for name, views in cache.items()}
    rows, extra, overlap_values = [], Counter(), []
    for record in records:
        name = record['file_name']
        if name not in names:
            continue
        truth = [mask_rle(a['segmentation'], 2048, 2048) for a in gt[record['source_image_id']]]
        pred = predictions[name]
        mat = np.asarray(mu.iou(truth, pred, [0] * len(pred)), dtype=np.float32) if pred else np.zeros((len(truth), 0), np.float32)
        rows.append(pq_counts(mat))
        overlap_values.extend(mat[mat > 0].astype(float).tolist())
        hit = mat > .5
        for idx, rle in enumerate(truth):
            if mu.area(rle) < 2000:
                extra['small_gt'] += 1
                extra['small_gt_matched'] += int(hit[idx].any())
        extra['gt_overlapping_multiple_predictions'] += int(((mat > 0).sum(1) > 1).sum())
        extra['predictions_overlapping_multiple_gt'] += int(((mat > 0).sum(0) > 1).sum())
    values = np.asarray(overlap_values)
    bins = np.linspace(0., 1., 51)
    return dict(metrics=summarize_counts(rows), diagnostic_counts=dict(extra),
                overlap_distributions=dict(
                    threshold='all pairs with IoU > 0, official v6 plotting semantics',
                    bins=bins.tolist(), iou_counts=np.histogram(values, bins)[0].tolist(),
                    dice_counts=np.histogram(2 * values / (1 + values), bins)[0].tolist()))


import hashlib
cache_path=OUT/'cpu_metric_cache.json'
metric_cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
def memo_eval(path):
 key=hashlib.sha256(path.read_bytes()).hexdigest()
 if key not in metric_cache:
  metric_cache[key]=evaluate(json.loads(path.read_text()))
  temp=cache_path.with_suffix('.tmp');temp.write_text(json.dumps(metric_cache));temp.replace(cache_path)
 return metric_cache[key]
import torch
base_path=ROOT/'finetune_results/model_best_finetune.pth'
assert hashlib.sha256(base_path.read_bytes()).hexdigest()==summary['source_sha256']
base=torch.load(base_path,map_location='cpu',weights_only=True)['model']
result={'baseline':memo_eval(ROOT/'finetune_results/finetune_calibration_candidates.json')}
schedule=json.loads((OUT/'training_schedule.json').read_text())
schedule_sha=hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
assert schedule_sha==summary['schedule_sha256']
sequences=[];checks={}
partial='--partial' in sys.argv
assert (len(summary['arms'])>=1) if partial else (summary['status']=='complete_three_arms' and len(summary['arms'])==3)
assert all(p['status']=='PASS' for p in summary['preflights'])
for arm in summary['arms']:
 label=arm['arm'];result[label]=memo_eval(OUT/(label+'_candidates.json'))
 for k,v in arm['calibration'].items():assert abs(result[label]['metrics'][k]-v)<1e-7,(label,k)
 hist=json.loads((OUT/(label+'_history.json')).read_text());accepted=[r for r in hist if r['optimizer_updated']]
 assert len(accepted)==arm['effective_updates']==summary['effective_updates_per_arm']
 assert all(np.isfinite(r['loss']) for r in hist)
 specs=[{k:r[k] for k in ('file_name','source_image_id','hflip','vflip')} for r in accepted]
 assert specs==schedule[:len(specs)];sequences.append(specs)
 checkpoint=torch.load(OUT/(label+'_last.pth'),map_location='cpu',weights_only=False)
 assert checkpoint['source_checkpoint_sha256']==summary['source_sha256'] and checkpoint['input_size']==2048
 assert checkpoint['effective_updates']==len(accepted) and checkpoint['schedule_sha256']==schedule_sha
 delta=checkpoint['delta'];assert hasattr(delta,'_metadata')
 assert delta._metadata.get('sem_seg_head',{}).get('version')==2
 assert len(checkpoint['optimizer']['state'])>0
 assert sum(len(g['params']) for g in checkpoint['optimizer']['param_groups'])==len(checkpoint['trainable_names'])
 assert all('.snake_' in k or k.startswith('sem_seg_head.predictor.mask_embed.') for k in delta)
 assert all(torch.isfinite(v).all() for v in delta.values())
 changed=[k for k,v in delta.items() if k.startswith('sem_seg_head.predictor.mask_embed.') and not torch.equal(v,base[k])]
 assert changed
 assert all(k in checkpoint for k in ('optimizer','scaler','python_rng','numpy_rng','torch_rng','cuda_rng'))
 assert arm['integrity']['frozen_parameters_identical'] and arm['integrity']['mask_embed_changed']
 if label=='fixed_snake':assert all(v==0 for v in arm['integrity']['offset_nonzero'].values())
 if label=='snake':assert sum(arm['integrity']['offset_nonzero'].values())>0
 checks[label]=dict(checkpoint_sha256=hashlib.sha256((OUT/(label+'_last.pth')).read_bytes()).hexdigest(),metadata_preserved=True,finite_delta=True,mask_embed_changed_keys=changed,recovery_state_present=True,actual_updates=len(accepted),sampling_sequence_matches=True)
assert all(seq==sequences[0] for seq in sequences)
report=dict(status='PARTIAL_PASS' if partial else 'PASS',results=result,checkpoint_checks=checks,identical_training_sampling=True,actual_full2048_zero_gate_identity=summary['actual_zero_gate_full2048_identity'],scope='70 reused calibration physical images, not independent validation; three matched full2048 arms',submission_generated=False)
(OUT/('independent_verification_partial.json' if partial else 'independent_verification.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
