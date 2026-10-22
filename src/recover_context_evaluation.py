# Recover interrupted final evaluation only. No optimizer updates or retraining.
from datetime import datetime,timezone
CUTOFF=datetime.fromisoformat('2026-10-16T23:35:00+00:00').timestamp()
assert time.time()<CUTOFF-1800
OUT=Path('/kaggle/working/context1536_results');assert OUT.is_dir()
LIMIT_SECONDS=8*3600
best_path=OUT/'model_best_context.pth';last_path=OUT/'model_last_context.pth'
best=torch.load(best_path,map_location='cpu',weights_only=True)
assert best['source_sha256']=='6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1'
assert best['context_step']==200 and best['crop_size']==1536
model.load_state_dict(best['model'],strict=True)
assert all(torch.equal(v.detach().cpu(),best['model'][k]) for k,v in model.state_dict().items())
best_metric=best['calibration'];best_step=best['context_step'];del best
last=torch.load(last_path,map_location='cpu',weights_only=True)
assert last['context_step']==600
assert last['early_stopping_count']==2
assert {'optimizer','scaler','random_state','torch_rng','cuda_rng'}<=last.keys()
training_steps=last['context_step'];train_elapsed=last['elapsed_seconds'];del last
history=json.loads((OUT/'training_history.json').read_text())
assert history[-1]['step']==600
print('CONTEXT_RECOVERY_STRICT_LOAD_PASS',json.dumps(dict(best_step=best_step,training_steps=training_steps,optimizer_updates_in_recovery=0)),flush=True)
model.requires_grad_(False).eval()

def recover_context_eval(split,step):
    assert time.time()<CUTOFF-1200
    names=sorted(n for n in part if part[n]==split)
    cache=collect(train_dir,names,VIEWS,f'{split}_step{step}_candidates.json')
    predictions={name:[p['rle'] for p in disjoint(consensus(cache[name])) if p['score']>=.9 and p['area']>=256] for name in names}
    counts=[]
    for record in records:
        if part[record['file_name']]!=split:continue
        truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[record['source_image_id']]];prediction=predictions[record['file_name']]
        iou=mu.iou(truth,prediction,[0]*len(prediction)) if prediction else np.zeros((len(truth),0))
        counts.append(pq_counts(iou))
    return summarize_counts(counts)

summary_path=OUT/'context_summary.json'
if summary_path.exists():
    summary=json.loads(summary_path.read_text());print('CONTEXT_ALREADY_COMPLETE',json.dumps(summary),flush=True)
else:
    summary=dict(crop=1536,lr=2e-6,training_steps=training_steps,best_step=best_step,best_calibration=best_metric,
                 baseline_calibration_pq=.4563696594893772,baseline_holdout_pq=.40760711778568315,
                 holdout_accepted=False,stop_reason='early_stop',source_sha256='6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1',
                 peak_vram_gib=10.61,training_elapsed_seconds=train_elapsed,recovery_performed=True,recovery_optimizer_updates=0)
    holdout=recover_context_eval('holdout',best_step);summary['holdout']=holdout
    summary['holdout_accepted']=holdout['PQ']>.40760711778568315+.003
    print('CONTEXT_HOLDOUT',json.dumps(holdout),flush=True)
    if summary['holdout_accepted']:
        assert time.time()<CUTOFF-1800
        names=sorted(p.name for p in test_dir.glob('*.jpeg'));assert len(names)==180
        cache=collect(test_dir,names,VIEWS,'test_candidates.json');rows=[]
        for name in names:
            prediction=[p['rle'] for p in disjoint(consensus(cache[name])) if p['score']>=.9 and p['area']>=256]
            rows.extend((Path(name).stem+'_'+str(i),p['counts']) for i,p in enumerate(prediction,1))
        with (OUT/'submission_context.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['filament_id','segmentation_rle']);writer.writerows(rows)
        summary.update(test_images=180,instances=len(rows))
    summary.update(recovery_elapsed_seconds=round(time.monotonic()-RUN_STARTED),completed_at_utc=datetime.now(timezone.utc).isoformat())
    summary_path.write_text(json.dumps(summary,indent=2))
with zipfile.ZipFile('/kaggle/working/context1536_results.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
    for path in OUT.iterdir():archive.write(path,path.name)
print('CONTEXT_COMPLETE',json.dumps(summary),flush=True)
