import os
"""Independently verify a downloaded pilot, without model inference or test GT."""
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'snake_pilot_20261017'
sys.path.insert(0, str(ROOT / 'bundle'))
from solar_evaluation import mask_rle, pq_counts, summarize_counts

# Local dataset root. Override with SOLAR_DATA_ROOT=<path> on your machine.
DATA_ROOT = Path(os.environ.get('SOLAR_DATA_ROOT', 'solar_project'))

VIEWS = ('native', 'hflip', 'vflip', 'rot180')
tree = ast.parse((ROOT / 'predict_calibrated_kaggle.py').read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('encode', 'consensus', 'disjoint')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'locked_recipe', 'exec'))
summary = json.loads((OUT / 'pilot_summary.json').read_text())
records = json.loads((ROOT / 'bundle/validation_records.json').read_text())
part = json.loads((ROOT / 'calibration_results/split.json').read_text())
names = {name for name, split in part.items() if split == 'calibration'}
raw = json.loads((DATA_ROOT / 'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
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

result = {'baseline': evaluate(json.loads((ROOT / 'finetune_results/finetune_calibration_candidates.json').read_text()))}
assert abs(result['baseline']['metrics']['PQ'] - .4563696594893772) < 1e-7
sample_sequences = []
for arm in summary['arms']:
    label = arm['mode']
    result[label] = evaluate(json.loads((OUT / (label + '_candidates.json')).read_text()))
    for key, value in arm['calibration'].items():
        assert abs(result[label]['metrics'][key] - value) < 1e-7, (label, key)
    history = json.loads((OUT / (label + '_history.json')).read_text())
    assert len(history) == 200 and history[-1]['step'] == 200
    assert all(np.isfinite(row['loss']) and np.isfinite(row['alpha']) for row in history)
    sample_sequences.append([row['file_name'] for row in history])
assert all(sequence == sample_sequences[0] for sequence in sample_sequences)
report = dict(status='PASS', results=result, identical_training_image_sequences=True,
              actual_2048_zero_gate_identity=summary.get('actual_2048_zero_gate_identity'),
              scope='70 calibration images; reused development split, not independent validation',
              submission_generated=False)
(OUT / 'independent_verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
