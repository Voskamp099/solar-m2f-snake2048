"""Independently verify a downloaded pilot, without model inference or test GT."""
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'snake2048_resume_20261018'
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
import torch
PARENT=ROOT/'snake2048_controls_20261018'
cache_path=OUT/'cpu_metric_cache.json'
if cache_path.exists():metric_cache=json.loads(cache_path.read_text())
else:metric_cache=json.loads((PARENT/'cpu_metric_cache.json').read_text())
def memo_eval(path):
 key=hashlib.sha256(path.read_bytes()).hexdigest()
 if key not in metric_cache:
  metric_cache[key]=evaluate(json.loads(path.read_text()))
  tmp=cache_path.with_suffix('.tmp');tmp.write_text(json.dumps(metric_cache));tmp.replace(cache_path)
 return metric_cache[key]
partial='--partial' in sys.argv
assert len(summary['preflights'])==3 and all(x['status']=='PASS' and x['prediction_cache_match'] for x in summary['preflights'])
if not partial:assert summary['status'] in ('complete','stopped_no_calibration_improvement')
assert hashlib.sha256((ROOT/'finetune_results/model_best_finetune.pth').read_bytes()).hexdigest()==summary['source_sha256']
base=torch.load(ROOT/'finetune_results/model_best_finetune.pth',map_location='cpu',weights_only=True)['model']
schedule=json.loads((OUT/'training_schedule.json').read_text())
assert len(schedule)==1200 and schedule[:600]==json.loads((PARENT/'training_schedule.json').read_text())
assert all(row['file_name'] not in part for row in schedule)
assert hashlib.sha256(json.dumps(schedule).encode()).hexdigest()==summary['schedule_sha256']
parents={mode:torch.load(PARENT/(mode+'_last.pth'),map_location='cpu',weights_only=False) for mode in summary['origin_step250_sha256']}
for mode,sha in summary['origin_step250_sha256'].items():assert hashlib.sha256((PARENT/(mode+'_last.pth')).read_bytes()).hexdigest()==sha
results={'baseline':memo_eval(ROOT/'finetune_results/finetune_calibration_candidates.json')}
checks={};rounds=summary['rounds'][:]
if partial and summary.get('incomplete_round'):rounds+=[summary['incomplete_round']]
assert rounds
for phase in rounds:
 step=phase['target'];sequences=[]
 if not partial:assert len(phase['arms'])==3
 for a in phase['arms']:
  mode=a['arm'];label=mode+'_step'+str(step)
  results[label]=memo_eval(OUT/(label+'_candidates.json'))
  for key,value in a['calibration'].items():assert abs(results[label]['metrics'][key]-value)<1e-7,(label,key)
  checkpoint_path=OUT/(label+'.pth');cp=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
  assert cp['effective_updates']==step and cp['input_size']==2048 and cp['source_checkpoint_sha256']==summary['source_sha256']
  assert cp['origin_step250_sha256']==summary['origin_step250_sha256'][mode]
  assert cp['history'][:len(parents[mode]['history'])]==parents[mode]['history']
  h=cp['history'];assert h==json.loads((OUT/(label+'_history.json')).read_text())
  accepted=[r for r in h if r['optimizer_updated']];assert len(accepted)==step and len(h)==cp['attempts']
  specs=[{k:r[k] for k in ('file_name','source_image_id','hflip','vflip')} for r in accepted]
  assert specs==schedule[:step];sequences.append(specs)
  assert all(np.isfinite(r['loss']) for r in h)
  delta=cp['delta'];assert delta._metadata['sem_seg_head']['version']==2
  assert all('.snake_' in k or k.startswith('sem_seg_head.predictor.mask_embed.') for k in delta)
  assert all(torch.isfinite(t).all() for t in delta.values())
  assert any(not torch.equal(t,base[k]) for k,t in delta.items() if k.startswith('sem_seg_head.predictor.mask_embed.'))
  assert all(k in cp for k in ('optimizer','scaler','python_rng','numpy_rng','torch_rng','cuda_rng'))
  assert sum(len(g['params']) for g in cp['optimizer']['param_groups'])==len(cp['trainable_names'])
  assert len(cp['optimizer']['state'])==len(cp['trainable_names'])
  assert a['integrity']['frozen_parameters_identical'] and a['integrity']['mask_embed_changed']
  if mode=='fixed_snake':assert all(x==0 for x in a['integrity']['offset_nonzero'].values())
  if mode=='snake':assert sum(a['integrity']['offset_nonzero'].values())>0
  checks[label]=dict(checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),actual_updates=step,origin250_history_unchanged=True,matching_sampling_sequence=True,metadata_preserved=True,recovery_states_present=True)
 assert all(seq==sequences[0] for seq in sequences)
report=dict(status='PARTIAL_PASS' if partial else 'PASS',results=results,checkpoint_checks=checks,source_sha256=summary['source_sha256'],resume_predictions_match_step250=True,training_sampling_matched=True,scope='70 reused calibration images; no independent test or leaderboard score',submitted=False)
(OUT/('independent_verification_partial.json' if partial else 'independent_verification.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
