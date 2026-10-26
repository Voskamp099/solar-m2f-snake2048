import os
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from pycocotools import mask as mu

# Local dataset root. Override with SOLAR_DATA_ROOT=<path> on your machine.
DATA_ROOT = Path(os.environ.get('SOLAR_DATA_ROOT', 'solar_project'))

R=Path(__file__).resolve().parent
raw=json.loads((DATA_ROOT / 'MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
records=json.loads((R/'bundle/validation_records.json').read_text())
pred=json.loads((R/'results/validation_predictions.json').read_text())
annotations=defaultdict(list)
for a in raw['annotations']: annotations[a['image_id']].append(a)
by_name=defaultdict(list)
for r in records:by_name[r['file_name']].append(r)
counts=defaultdict(int);rows=[]
for name,rs in by_name.items():
    p=pred[name];truth=[];slices=[]
    for r in rs:
        start=len(truth)
        for a in annotations[r['source_image_id']]:
            s=a['segmentation']
            if isinstance(s,list):s=mu.merge(mu.frPyObjects(s,2048,2048))
            elif isinstance(s['counts'],list):s=mu.frPyObjects(s,2048,2048)
            truth.append(s)
        slices.append((start,len(truth)))
    iou=mu.iou(truth,p,[0]*len(p)).astype(np.float32) if p else np.zeros((len(truth),0),np.float32)
    matches_any_record=(iou>.5).any(0)
    for r,(start,end) in zip(rs,slices):
        mat=iou[start:end];hit=mat>.5
        fp=~hit.any(0);fn=~hit.any(1)
        values=dict(tp=int(hit.sum()),fp=int(fp.sum()),fn=int(fn.sum()),
                    fp_matching_other_annotation=int((fp&matches_any_record).sum()),
                    fp_no_gt_overlap=int((fp&~(mat>0).any(0)).sum()),
                    fn_no_prediction_overlap=int((fn&~(mat>0).any(1)).sum()),
                    fn_best_iou_03_to_05=int((fn&(mat>=.3).any(1)).sum()),
                    fp_best_iou_03_to_05=int((fp&(mat>=.3).any(0)).sum()))
        for k,v in values.items():counts[k]+=v
        rows.append(dict(file_name=name,source_image_id=r['source_image_id'],**values))
assert (counts['tp'],counts['fp'],counts['fn'])==(860,1105,765),counts
out=dict(counts=dict(counts),unit='independent annotation record; same image may be scored repeatedly',
         interpretation='Matching another annotation indicates annotation disagreement, not proof the prediction is physically correct.',
         worst_records=sorted(rows,key=lambda r:r['fp']+r['fn'],reverse=True)[:30])
(R/'calibration_results/baseline_error_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(counts)))
