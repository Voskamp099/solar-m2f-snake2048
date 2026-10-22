# Continue final evaluation from preserved training state; no optimizer updates.
import copy
from collections import OrderedDict
assert 'model' in globals() and 'OUT' in globals(), 'Keep the completed training kernel.'
best=torch.load(OUT/'model_best_finetune.pth',map_location='cpu',weights_only=True)
assert best['finetune_step']==300
assert best['source_checkpoint_sha256']==provenance['checkpoint_sha256']
assert set(best['model'])==set(model.state_dict())
restored=OrderedDict(best['model'])
restored._metadata=copy.deepcopy(model.state_dict()._metadata)
loaded=model.load_state_dict(restored,strict=True)
assert all(torch.equal(v.detach().cpu(),best['model'][k]) for k,v in model.state_dict().items())
print('RECOVERY_STRICT_LOAD_PASS',str(loaded),'step',best['finetune_step'],'all tensors identical',flush=True)
# Upgrade best/last disk files to preserve metadata for future independent loads.
for filename in ['model_best_finetune.pth','model_last_finetune.pth']:
    path=OUT/filename
    state=torch.load(path,map_location='cpu',weights_only=True)
    weights=OrderedDict(state['model'])
    weights._metadata=copy.deepcopy(model.state_dict()._metadata)
    state['model']=weights
    temporary=path.with_suffix('.tmp')
    torch.save(state,temporary);temporary.replace(path)
del best,restored,state,weights
model.requires_grad_(False).eval()
def evaluate_fixed_consensus(split):
    selected=sorted(n for n in part if part[n]==split)
    cached=collect(train_dir,selected,VIEWS,'finetune_'+split+'_candidates.json')
    predictions={n:[x['rle'] for x in disjoint(consensus(cached[n]))
                    if x['score']>=.9 and x['area']>=256] for n in selected}
    rows=[]
    for r in records:
        if part[r['file_name']]!=split:continue
        truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[r['source_image_id']]]
        pred=predictions[r['file_name']]
        iou=mu.iou(truth,pred,[0]*len(pred)) if pred else np.zeros((len(truth),0))
        rows.append(pq_counts(iou))
    return summarize_counts(rows)

# Weights use the original native-view proxy for selection; the final candidate
# must beat the current best pipeline with its already frozen postprocessor.
baseline_consensus_calibration_pq=.38339739457112637
baseline_consensus_holdout_pq=.34730657114948477
final_calibration=None;holdout=None;accept=False
if best_step>0 and best_metric['PQ']>baseline_cal['PQ']+.003:
    final_calibration=evaluate_fixed_consensus('calibration')
    if final_calibration['PQ']>baseline_consensus_calibration_pq+.003:
        holdout=evaluate_fixed_consensus('holdout')
        accept=holdout['PQ']>baseline_consensus_holdout_pq+.003
summary=dict(training_performed=True,training_steps=training_steps,best_step=best_step,
             baseline_calibration=baseline_cal,best_calibration=best_metric,holdout=holdout,
             baseline_consensus_calibration_pq=baseline_consensus_calibration_pq,
             baseline_consensus_holdout_pq=baseline_consensus_holdout_pq,
             fixed_consensus_calibration=final_calibration,holdout_accepted=bool(accept),
             stop_reason=stop_reason,early_stopping_patience=PATIENCE,
             current_best_public_score=.30,
             crop_size=CROP_SIZE,backbone_frozen=True,physical_image_uniform_sampling=True,
             original_checkpoint_sha256=provenance['original_checkpoint_sha256'],
             source_checkpoint_sha256=provenance['checkpoint_sha256'],
             peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
             elapsed_seconds=round(time.monotonic()-RUN_STARTED),gpu=torch.cuda.get_device_name(0))
if accept:
    model.eval();rows=[]
    tests=sorted(p.name for p in test_dir.glob('*.jpeg'));assert len(tests)==180
    cached=collect(test_dir,tests,VIEWS,'finetune_test_candidates.json')
    for i,name in enumerate(tests,1):
        pred=[x['rle'] for x in disjoint(consensus(cached[name])) if x['score']>=.9 and x['area']>=256]
        rows.extend((Path(name).stem+'_'+str(k),x['counts']) for k,x in enumerate(pred,1))
        if i%10==0:print('FINETUNE_TEST',i,'/180',flush=True)
    with (OUT/'submission_finetune.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['filament_id','segmentation_rle']);w.writerows(rows)
    summary.update(processed_test_images=180,predicted_instances=len(rows))
(OUT/'training_history.json').write_text(json.dumps(history))
(OUT/'finetune_summary.json').write_text(json.dumps(summary,indent=2))
with zipfile.ZipFile('/kaggle/working/finetune_results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir():z.write(p,p.name)
print('FINETUNE_COMPLETE',json.dumps(summary),flush=True)
