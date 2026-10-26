import os
from pathlib import Path
import ast,hashlib,json,sys,zipfile
from collections import defaultdict
import numpy as np
from pycocotools import mask as mu
import torch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'context1536_results';OUT.mkdir(exist_ok=True)
archive=DATA_ROOT / 'context1536_results.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all('/' not in n for n in z.namelist())
    names=z.namelist();z.extractall(OUT)
summary=json.loads((OUT/'context_summary.json').read_text());history=json.loads((OUT/'training_history.json').read_text())
assert summary['training_steps']==600 and summary['best_step']==200
assert history[-1]['step']==600 and len(history)==600
actual=sum(x['optimizer_updated'] for x in history)
last=torch.load(OUT/'model_last_context.pth',map_location='cpu',weights_only=True)
assert last['context_step']==600 and last['early_stopping_count']==2
assert last['model']._metadata['sem_seg_head']['version']==2
assert {'optimizer','scaler','random_state','torch_rng','cuda_rng'}<=last.keys()
assert max(float(s['step']) for s in last['optimizer']['state'].values())==actual
best=torch.load(OUT/'model_best_context.pth',map_location='cpu',weights_only=True)
assert best['context_step']==200 and best['model']._metadata['sem_seg_head']['version']==2
assert hashlib.sha256((OUT/'model_best_context.pth').read_bytes()).hexdigest()=='cf4d61853501107e2bdc908f555508b2e540381da120056b58a75b7f237aa591'
del last,best
sys.path.insert(0,str(ROOT/'bundle'))
from solar_evaluation import mask_rle,pq_counts,summarize_counts

# Local dataset root. Override with SOLAR_DATA_ROOT=<path> on your machine.
DATA_ROOT = Path(os.environ.get('SOLAR_DATA_ROOT', 'solar_project'))
VIEWS=('native','hflip','vflip','rot180')
tree=ast.parse((ROOT/'calibrate_kaggle.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('encode','consensus','disjoint')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'official_postprocess','exec'))
cache=json.loads((OUT/'holdout_step200_candidates.json').read_text());part=json.loads((ROOT/'calibration_results/split.json').read_text())
assert len(cache)==71 and set(cache)=={n for n,p in part.items() if p=='holdout'}
assert all(set(v)==set(VIEWS) for v in cache.values())
predictions={n:[p['rle'] for p in disjoint(consensus(v)) if p['score']>=.9 and p['area']>=256] for n,v in cache.items()}
raw=json.loads((DATA_ROOT / 'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text());gt=defaultdict(list)
for a in raw['annotations']:gt[a['image_id']].append(a)
records=json.loads((ROOT/'bundle/validation_records.json').read_text());counts=[]
for record in records:
    if part[record['file_name']]!='holdout':continue
    truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[record['source_image_id']]];prediction=predictions[record['file_name']]
    iou=mu.iou(truth,prediction,[0]*len(prediction)) if prediction else np.zeros((len(truth),0))
    counts.append(pq_counts(iou))
metric=summarize_counts(counts);assert abs(metric['PQ']-summary['holdout']['PQ'])<1e-10
assert not summary['holdout_accepted']
assert metric['PQ']-summary['baseline_holdout_pq']<.003
assert not any(n.startswith('submission') or n=='test_candidates.json' for n in names)
result=dict(status='PASS',archive_crc=True,files=names,steps=600,optimizer_updates=actual,best_step=200,
            metadata_version=2,full_resume_state=True,independent_holdout=metric,accepted=False,
            holdout_gain=metric['PQ']-summary['baseline_holdout_pq'],public_best_unchanged=.36)
(OUT/'final_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
