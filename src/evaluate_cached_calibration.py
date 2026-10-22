"""Resume the fixed calibration experiment from saved GPU predictions, on CPU."""
import ast
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'bundle'))
from solar_evaluation import mask_rle, pq_counts, summarize_counts

OUT = ROOT / 'calibration_results'
started = time.monotonic()
source = (ROOT / 'calibrate_kaggle.py').read_text()
# Reuse the exact mask transformations from the pre-registered GPU experiment.
tree = ast.parse(source)
selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in ('encode', 'consensus', 'disjoint', 'evaluate')]
assert len(selected) == 4
VIEWS = ('native', 'hflip', 'vflip', 'rot180')
exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / 'calibrate_kaggle.py'), 'exec'))
cache_path = OUT / 'validation_candidates.json'
assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == '2d0cff852172173194db541eabbf240124775ec1b404186aa8ec6ddfaa1f7399'
cache = json.loads(cache_path.read_text())
part = json.loads((OUT / 'split.json').read_text())
records = json.loads((ROOT / 'bundle/validation_records.json').read_text())
assert set(cache) == set(part) == {r['file_name'] for r in records}
assert len(cache) == 141 and all(set(v) == set(VIEWS) for v in cache.values())
raw = json.loads(Path('/Users/raelynn/Downloads/solar_project/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
gt = defaultdict(list)
for a in raw['annotations']:
    gt[a['image_id']].append(a)
pools = {'native': {}, 'flip_consensus': {}}
pool_path = OUT / 'processed_pools.json'
if pool_path.exists():
    pools = json.loads(pool_path.read_text())
for i, name in enumerate(sorted(part), 1):
    if name not in pools['native']:
        pools['native'][name] = disjoint(cache[name]['native'])
        pools['flip_consensus'][name] = disjoint(consensus(cache[name]))
        tmp = pool_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(pools))
        tmp.replace(pool_path)
    if i % 20 == 0 or i == len(part):
        print('POSTPROCESS', i, '/', len(part), 'seconds', round(time.monotonic() - started), flush=True)
prepared = []
for r in records:
    truth = [mask_rle(a['segmentation'], 2048, 2048) for a in gt[r['source_image_id']]]
    assert len(truth) == r['n_gt']
    for mode in pools:
        p = pools[mode][r['file_name']]
        iou = mu.iou(truth, [x['rle'] for x in p], [0] * len(p)) if p else np.zeros((len(truth), 0))
        prepared.append((mode, part[r['file_name']], iou,
                         np.array([x['score'] for x in p]), np.array([x['area'] for x in p])))
baseline = evaluate('native', .5, 0)
assert abs(baseline['all']['PQ'] - .31056424399272314) < .00001, baseline
grid = []
for mode, threshold, min_area in itertools.product(pools, [round(x / 100, 2) for x in range(15, 91, 5)], [0, 64, 128, 256]):
    # Persist only actually measured calibration metrics; no dummy holdout values.
    metrics = {'calibration': evaluate(mode, threshold, min_area, calibration_only=True)['calibration']}
    grid.append(dict(mode=mode, threshold=threshold, min_area=min_area, metrics=metrics))
grid.sort(key=lambda g: g['metrics']['calibration']['PQ'], reverse=True)
chosen = dict(grid[0])
chosen['metrics'] = evaluate(chosen['mode'], chosen['threshold'], chosen['min_area'])
accept = (chosen['metrics']['calibration']['PQ'] > baseline['calibration']['PQ'] + .003
          and chosen['metrics']['holdout']['PQ'] > baseline['holdout']['PQ'] + .003)
summary = dict(baseline=baseline, chosen=chosen, holdout_accepted=bool(accept),
               training_performed=False, selection='max calibration PQ; date-disjoint holdout acceptance gain >0.003',
               checkpoint_sha256=json.loads((ROOT / 'bundle/provenance.json').read_text())['checkpoint_sha256'],
               grid_size=len(grid), test_predictions_generated=False, evaluation_device='local CPU',
               gpu_forward_repeated=False, elapsed_seconds=round(time.monotonic()-started))
(OUT / 'calibration_grid.json').write_text(json.dumps(grid, indent=2))
(OUT / 'calibration_summary.json').write_text(json.dumps(summary, indent=2))
print('CALIBRATION_RESULT', json.dumps(summary), flush=True)
