import os
"""Audit the locked 0.36 recipe on development/calibration images only."""
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'bundle'))
from solar_evaluation import mask_rle, pq_counts, summarize_counts

# Local dataset root. Override with SOLAR_DATA_ROOT=<path> on your machine.
DATA_ROOT = Path(os.environ.get('SOLAR_DATA_ROOT', 'solar_project'))

VIEWS = ('native', 'hflip', 'vflip', 'rot180')
tree = ast.parse((ROOT / 'predict_calibrated_kaggle.py').read_text())
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name in ('encode', 'consensus', 'disjoint')]
assert len(nodes) == 3
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'locked_postprocessing', 'exec'))

cache = json.loads((ROOT / 'finetune_results/finetune_calibration_candidates.json').read_text())
part = json.loads((ROOT / 'calibration_results/split.json').read_text())
assert set(cache) == {name for name, split in part.items() if split == 'calibration'}
records = [r for r in json.loads((ROOT / 'bundle/validation_records.json').read_text())
           if r['file_name'] in cache]
raw = json.loads((DATA_ROOT / 'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
gt = defaultdict(list)
for ann in raw['annotations']:
    gt[ann['image_id']].append(ann)
predictions = {}
raw_predictions = {}
for name, views in cache.items():
    predictions[name] = [p['rle'] for p in disjoint(consensus(views))
                         if p['score'] >= .90 and p['area'] >= 256]
    raw_predictions[name] = [p['rle'] for view in VIEWS for p in views[view]]

rows = []
counts = Counter()
gt_degrees, pred_degrees = Counter(), Counter()
ious, dices = [], []
misses = []
for record in records:
    name = record['file_name']
    truth = [mask_rle(a['segmentation'], 2048, 2048) for a in gt[record['source_image_id']]]
    pred = predictions[name]
    mat = np.asarray(mu.iou(truth, pred, [0] * len(pred)), dtype=np.float32) if pred else np.zeros((len(truth), 0), np.float32)
    candidate = raw_predictions[name]
    raw_mat = np.asarray(mu.iou(truth, candidate, [0] * len(candidate)), dtype=np.float32) if candidate else np.zeros((len(truth), 0), np.float32)
    row = pq_counts(mat)
    rows.append(row)
    hit = mat > .5
    gt_degree = (mat > 0).sum(1)
    pred_degree = (mat > 0).sum(0)
    gt_degrees.update(map(int, gt_degree))
    pred_degrees.update(map(int, pred_degree))
    overlap = mat[mat > 0].astype(float)
    ious.extend(overlap.tolist())
    dices.extend((2 * overlap / (1 + overlap)).tolist())
    for idx in np.flatnonzero(~hit.any(1)):
        best = float(mat[idx].max()) if pred else 0.
        raw_best = float(raw_mat[idx].max()) if candidate else 0.
        category = 'no_overlap' if best == 0 else ('iou_0_to_03' if best < .3 else 'iou_03_to_05')
        counts['fn_' + category] += 1
        if raw_best > .5:
            counts['fn_with_qualifying_raw_candidate'] += 1
        area = int(mu.area(truth[idx]))
        counts['fn_area_lt2000' if area < 2000 else 'fn_area_ge2000'] += 1
        misses.append(dict(file_name=name, source_image_id=record['source_image_id'], gt_index=int(idx),
                           best_iou=best, best_raw_candidate_iou=raw_best, area=area))
    for idx in np.flatnonzero(~hit.any(0)):
        best = float(mat[:, idx].max()) if truth else 0.
        category = 'no_overlap' if best == 0 else ('iou_0_to_03' if best < .3 else 'iou_03_to_05')
        counts['fp_' + category] += 1

metrics = summarize_counts(rows)
assert abs(metrics['PQ'] - .4563696594893772) < 1e-7, metrics
assert (metrics['tp'], metrics['fp'], metrics['fn']) == (498, 204, 284)
bins = np.linspace(0., 1., 51)
output = dict(
    source_model='public0.36_step300', evaluation_split='calibration_only',
    physical_images=len(cache), annotation_records=len(records), metrics=metrics,
    error_counts=dict(counts),
    official_overlap_degree_histograms=dict(gt_to_prediction=dict(gt_degrees), prediction_to_gt=dict(pred_degrees), threshold='IoU > 0'),
    pairwise_distributions=dict(threshold='IoU > 0, all positive pairs (not TP-only)',
                               bins=bins.tolist(), iou_counts=np.histogram(ious, bins)[0].tolist(),
                               dice_counts=np.histogram(dices, bins)[0].tolist()),
    missed_instances=misses,
    caveats=['Repeated annotation records are counted separately, as in official v6.',
             'IoU>0 overlap degree is sensitive to tiny intersections; it is not a physical continuity judgement.',
             'Qualifying raw candidates are an oracle diagnostic, not deployable recall or a proposed tuned threshold.',
             'No internal-review or test labels inspected by this audit.',
             'Calibration is reused development data; independent date-group validation remains necessary.'])
path = ROOT / 'finetune_results/current_rubric_calibration_audit.json'
path.write_text(json.dumps(output, indent=2))
print(json.dumps({key: output[key] for key in ['physical_images', 'annotation_records', 'metrics', 'error_counts', 'official_overlap_degree_histograms']}), flush=True)
