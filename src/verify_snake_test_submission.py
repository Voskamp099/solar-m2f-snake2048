"""Independently rebuild the CSV from cached candidates and validate every mask."""
import ast, csv, hashlib, json
from pathlib import Path
import numpy as np
from pycocotools import mask as mu

P=Path(__file__).resolve().parent
O=P/'snake_best_submission_20261018'
VIEWS=('native','hflip','vflip','rot180')
tree=ast.parse((P/'predict_calibrated_kaggle.py').read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('encode','consensus','disjoint')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'locked_recipe','exec'))
s=json.loads((O/'summary.json').read_text())
assert s['status']=='complete' and s['processed_test_images']==180 and not s['training_performed']
assert s['preflight']['status']=='PASS' and s['preflight']['restored_prediction_cache_match']
assert s['delta_checkpoint_sha256']=='ca87077f721356018594b3f0fe4cf3bea562991c5f3e484347782cfa7946ad75'
assert s['base_checkpoint_sha256']=='6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1'
names=json.loads((O/'test_names.json').read_text())
local_names=sorted(x.name for x in Path('/Users/raelynn/Downloads/solar_project/MAGFiLO_1.0_Kaggle_2026/test/test_images').glob('*.jpeg'))
assert names==local_names and len(names)==180
cache=json.loads((O/'test_candidates.json').read_text())
predictions=json.loads((O/'test_predictions.json').read_text())
assert set(cache)==set(predictions)==set(names)
rows=[];counts={};area_sum=0
for name in names:
    views=cache[name]
    assert set(views)==set(VIEWS)
    for view in VIEWS:
        for p in views[view]:
            assert p['view']==view and p['rle']['size']==[2048,2048]
            assert np.isfinite(p['score']) and 0<=p['score']<=1
            assert int(mu.area(p['rle']))>0
    selected=[p for p in disjoint(consensus(views)) if p['score']>=.9 and p['area']>=256]
    assert selected==predictions[name]
    counts[name]=len(selected)
    occupied=np.zeros((2048,2048),dtype=bool)
    for k,p in enumerate(selected,1):
        a=mu.decode(p['rle']).astype(bool)
        assert a.shape==(2048,2048) and not np.any(occupied&a)
        assert int(a.sum())==p['area']>=256
        occupied|=a;area_sum+=p['area']
        rows.append([Path(name).stem+'_'+str(k),p['rle']['counts']])
with (O/'submission_snake1200.csv').open(newline='') as f:
    reader=csv.reader(f);assert next(reader)==['filament_id','segmentation_rle'];actual=list(reader)
assert actual==rows and len({r[0] for r in actual})==len(actual)>0
assert counts==s['test_image_counts'] and len(actual)==s['predicted_instances']
sha=hashlib.sha256((O/'submission_snake1200.csv').read_bytes()).hexdigest()
assert sha==s['submission_sha256']
report=dict(status='PASS',processed_test_images=180,predicted_instances=len(rows),empty_test_images=sum(v==0 for v in counts.values()),
            submission_sha256=sha,all_masks_decode=True,all_masks_nonoverlapping=True,all_ids_unique=True,
            exact_rebuild_from_locked_recipe=True,weight_recovery_preflight='PASS',total_mask_pixels=area_sum,
            scope='CSV integrity only; does not estimate hidden-test score',submitted=False)
(O/'csv_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
