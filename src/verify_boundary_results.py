"""Independently recompute calibration PQ from downloaded boundary candidates."""
from pathlib import Path
import ast,json,sys
from collections import defaultdict
import numpy as np
from pycocotools import mask as mu
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'bundle'))
from solar_evaluation import mask_rle,pq_counts,summarize_counts
VIEWS=('native','hflip','vflip','rot180')
tree=ast.parse((ROOT/'calibrate_kaggle.py').read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('encode','consensus','disjoint')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_postprocessing','exec'))
OUT=ROOT/'boundary_results';cache=json.loads((OUT/'calibration_candidates.json').read_text())
part=json.loads((ROOT/'calibration_results/split.json').read_text())
assert set(cache)=={n for n,p in part.items() if p=='calibration'}
assert len(cache)==70 and all(set(v)==set(VIEWS) for v in cache.values())
assert all(set(by_level)=={'0.35','0.5','0.65'} for by_view in cache.values() for by_level in by_view.values())
raw=json.loads(Path('/Users/raelynn/Downloads/solar_project/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
gt=defaultdict(list)
for a in raw['annotations']:gt[a['image_id']].append(a)
records=json.loads((ROOT/'bundle/validation_records.json').read_text())
summary=json.loads((OUT/'boundary_summary.json').read_text());checks=[]
for level in (.35,.5,.65):
    predictions={n:[p['rle'] for p in disjoint(consensus({v:by_view[v][str(level)] for v in VIEWS}))
                    if p['score']>=.9 and p['area']>=256] for n,by_view in cache.items()}
    rows=[]
    for r in records:
        if part[r['file_name']]!='calibration':continue
        truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[r['source_image_id']]]
        p=predictions[r['file_name']]
        iou=mu.iou(truth,p,[0]*len(p)) if p else np.zeros((len(truth),0))
        rows.append(pq_counts(iou))
    metric=summarize_counts(rows)
    recorded=next(g['calibration'] for g in summary['grid'] if g['level']==level)
    assert abs(metric['PQ']-recorded['PQ'])<1e-10,(level,metric,recorded)
    checks.append(dict(level=level,calibration=metric))
    print('BOUNDARY_VERIFY',level,metric['PQ'],flush=True)
assert not summary['holdout_accepted'] and 'holdout' not in summary
assert summary['chosen']['calibration']['PQ']-summary['baseline_calibration_pq']<.003
result=dict(status='PASS',cache_images=70,all_three_calibration_metrics_match=True,holdout_not_evaluated=True,
            no_submission_generated=True,decision='reject_below_predeclared_gain',metrics=checks)
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2))
print('BOUNDARY_INDEPENDENT_VERIFY_PASS',flush=True)
