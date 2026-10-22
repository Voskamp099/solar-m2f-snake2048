"""Small CPU-only fusion experiment; select on calibration before holdout."""
import ast
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'bundle'))
from solar_evaluation import mask_rle, pq_counts, summarize_counts
OUT = ROOT / 'fusion_results'
OUT.mkdir(exist_ok=True)
VIEWS = ('native', 'hflip', 'vflip', 'rot180')
tree = ast.parse((ROOT / 'calibrate_kaggle.py').read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name in ('encode', 'consensus', 'disjoint')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'original_postprocessor', 'exec'))
part = json.loads((ROOT / 'calibration_results/split.json').read_text())
records = json.loads((ROOT / 'bundle/validation_records.json').read_text())
raw = json.loads(Path('/Users/raelynn/Downloads/solar_project/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
annotations = defaultdict(list)
for ann in raw['annotations']:
    annotations[ann['image_id']].append(ann)
truth = [(r['file_name'], [mask_rle(a['segmentation'], 2048, 2048)
                         for a in annotations[r['source_image_id']]]) for r in records]
old_pools = json.loads((ROOT / 'calibration_results/processed_pools.json').read_text())['flip_consensus']
old = {n: [x['rle'] for x in ps if x['score'] >= .9 and x['area'] >= 256]
       for n, ps in old_pools.items()}


def metrics(predictions):
    rows = []
    for name, gt in truth:
        if name not in predictions:
            continue
        p = predictions[name]
        matrix = mu.iou(gt, p, [0] * len(p)) if p else np.zeros((len(gt), 0))
        rows.append(pq_counts(matrix))
    return summarize_counts(rows)


def prepare_new(split):
    cache = json.loads((ROOT / 'finetune_results' / f'finetune_{split}_candidates.json').read_text())
    assert set(cache) == {n for n, s in part.items() if s == split}
    assert all(set(v) == set(VIEWS) for v in cache.values())
    result = {}
    for i, name in enumerate(sorted(cache), 1):
        result[name] = [x['rle'] for x in disjoint(consensus(cache[name]))
                        if x['score'] >= .9 and x['area'] >= 256]
        if i % 20 == 0:
            print('FUSION_PREPARE', split, i, '/', len(cache), flush=True)
    (OUT / f'new_{split}_predictions.json').write_text(json.dumps(result))
    return result


def normalized(rle):
    return {'size': rle['size'], 'counts': rle['counts'].decode('ascii')
            if isinstance(rle['counts'], bytes) else rle['counts']}


def fuse(new, original, mode):
    matrix = mu.iou(new, original, [0] * len(original)) if new and original else np.zeros((len(new), len(original)))
    hits = matrix > .5
    assert (hits.sum(0) <= 1).all() and (hits.sum(1) <= 1).all()
    result = []
    for i, mask in enumerate(new):
        js = np.flatnonzero(hits[i])
        if mode == 'reciprocal' and len(js) == 0:
            continue
        if len(js) and mode in ('paired_union', 'paired_intersection'):
            mask = normalized(mu.merge([mask, original[int(js[0])]], intersect=(mode == 'paired_intersection')))
        if mu.area(mask) >= 256:
            result.append(mask)
    if mode == 'add_unmatched_old':
        result.extend(p for j, p in enumerate(original) if not hits[:, j].any())
    if mode in ('paired_union', 'add_unmatched_old'):
        # Keep new-model confidence order and give its masks pixel ownership.
        result = [x['rle'] for x in disjoint([{'rle': p, 'score': 1 - i * 1e-6}
                                            for i, p in enumerate(result)]) if x['area'] >= 256]
    return result


started = time.monotonic()
new_cal = prepare_new('calibration')
base_cal = metrics(new_cal)
old_cal = metrics({n: old[n] for n in new_cal})
assert abs(old_cal['PQ'] - .38339739457112637) < 1e-7
modes = ('paired_union', 'paired_intersection', 'reciprocal', 'add_unmatched_old')
trials = []
for mode in modes:
    m = metrics({n: fuse(new_cal[n], old[n], mode) for n in new_cal})
    trials.append({'mode': mode, 'calibration': m})
    print('FUSION_CALIBRATION', mode, m['PQ'], flush=True)
chosen = max(trials, key=lambda x: x['calibration']['PQ'])
summary = {'baseline_new_calibration': base_cal, 'baseline_old_calibration': old_cal,
           'trials_calibration_only': trials, 'chosen': chosen, 'holdout_evaluated': False,
           'holdout_accepted': False, 'extra_gpu_hours': 0}
if chosen['calibration']['PQ'] > max(base_cal['PQ'], old_cal['PQ']) + .003:
    new_hold = prepare_new('holdout')
    base_hold = metrics(new_hold)
    hold = metrics({n: fuse(new_hold[n], old[n], chosen['mode']) for n in new_hold})
    summary.update(holdout_evaluated=True, baseline_new_holdout=base_hold, chosen_holdout=hold,
                   holdout_accepted=hold['PQ'] > max(base_hold['PQ'], .34730657114948477) + .003)
summary['elapsed_seconds'] = round(time.monotonic() - started)
(OUT / 'fusion_summary.json').write_text(json.dumps(summary, indent=2))
print('FUSION_RESULT', json.dumps(summary), flush=True)

if summary['holdout_accepted'] and (ROOT / 'finetune_results/submission_finetune.csv').exists():
    def read_predictions(path):
        result = defaultdict(list)
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                name = row['filament_id'].rsplit('_', 1)[0] + '.jpeg'
                result[name].append({'size': [2048, 2048], 'counts': row['segmentation_rle']})
        return result
    nt = read_predictions(ROOT / 'finetune_results/submission_finetune.csv')
    ot = read_predictions(ROOT / 'calibration_results/submission_calibrated.csv')
    expected = sorted(json.loads((ROOT / 'calibration_results/test_candidates.json').read_text()))
    rows = []
    for name in expected:
        rows.extend((Path(name).stem + '_' + str(i), p['counts'])
                    for i, p in enumerate(fuse(nt[name], ot[name], chosen['mode']), 1))
    with (OUT / 'submission_fusion.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['filament_id', 'segmentation_rle']); writer.writerows(rows)
    print('FUSION_TEST_COMPLETE', len(expected), 'images', len(rows), 'instances', flush=True)
