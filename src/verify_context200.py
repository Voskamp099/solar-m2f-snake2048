import os
from pathlib import Path
import ast,hashlib,json,sys,shutil
from collections import defaultdict
import numpy as np
from pycocotools import mask as mu
import torch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'context1536_results';OUT.mkdir(exist_ok=True)
shutil.copyfile(str(DATA_ROOT / 'model_best_context.pth'),OUT/'model_best_context_step200.pth')
shutil.copyfile(str(DATA_ROOT / 'calibration_step200_candidates.json'),OUT/'calibration_step200_candidates.json')
state=torch.load(OUT/'model_best_context_step200.pth',map_location='cpu',weights_only=True)
assert state['context_step']==200 and state['best_step']==200 and state['crop_size']==1536
assert state['source_sha256']=='6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1'
assert state['model']._metadata['sem_seg_head']['version']==2
assert {'optimizer','scaler','random_state','torch_rng','cuda_rng'}<=state.keys()
source=torch.load(ROOT/'finetune_results/model_best_finetune.pth',map_location='cpu',weights_only=True)
assert set(state['model'])==set(source['model'])
backbone=[k for k in source['model'] if k.startswith('backbone.')]
assert all(torch.equal(state['model'][k],source['model'][k]) for k in backbone)
changed=sum(not torch.equal(state['model'][k],source['model'][k]) for k in source['model'] if not k.startswith('backbone.'))
assert changed>0
metric=state['calibration'];sha=hashlib.sha256((OUT/'model_best_context_step200.pth').read_bytes()).hexdigest()
del source,state
sys.path.insert(0,str(ROOT/'bundle'))
from solar_evaluation import mask_rle,pq_counts,summarize_counts

# Local dataset root. Override with SOLAR_DATA_ROOT=<path> on your machine.
DATA_ROOT = Path(os.environ.get('SOLAR_DATA_ROOT', 'solar_project'))
VIEWS=('native','hflip','vflip','rot180')
tree=ast.parse((ROOT/'calibrate_kaggle.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('encode','consensus','disjoint')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_postprocessor','exec'))
cache=json.loads((OUT/'calibration_step200_candidates.json').read_text());part=json.loads((ROOT/'calibration_results/split.json').read_text())
assert set(cache)=={n for n,p in part.items() if p=='calibration'} and len(cache)==70
assert all(set(x)==set(VIEWS) for x in cache.values())
predictions={name:[p['rle'] for p in disjoint(consensus(by_view)) if p['score']>=.9 and p['area']>=256] for name,by_view in cache.items()}
raw=json.loads((DATA_ROOT / 'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text());gt=defaultdict(list)
for a in raw['annotations']:gt[a['image_id']].append(a)
records=json.loads((ROOT/'bundle/validation_records.json').read_text());rows=[]
for record in records:
    if part[record['file_name']]!='calibration':continue
    truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[record['source_image_id']]];pred=predictions[record['file_name']]
    iou=mu.iou(truth,pred,[0]*len(pred)) if pred else np.zeros((len(truth),0));rows.append(pq_counts(iou))
recomputed=summarize_counts(rows);assert abs(recomputed['PQ']-metric['PQ'])<1e-10
result=dict(status='PASS',step=200,calibration=metric,recomputed_calibration=recomputed,backbone_tensors_unchanged=len(backbone),head_tensors_changed=changed,state_dict_metadata_version=2,full_resume_state=True,holdout_evaluated=False,checkpoint_sha256=sha)
(OUT/'step200_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
