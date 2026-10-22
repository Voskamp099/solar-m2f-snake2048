# Run after the shared setup cell and calibration definitions (without its sweep).
import math, random, copy
from collections import OrderedDict
from detectron2.structures import Instances

TRAIN_SECONDS = 2 * 3600
MAX_STEPS = 3000
EVAL_EVERY = 300
CROP_SIZE = 1024
BASE_LR = 1e-5
OUT = Path('/kaggle/working/finetune_results')
OUT.mkdir(exist_ok=True)
archives=list(Path('/kaggle/input').rglob('training_images.bin'))
assert len(archives)==1, archives
training_root=Path('/kaggle/working/training_images')
training_root.mkdir(exist_ok=True)
with zipfile.ZipFile(archives[0]) as z: z.extractall(training_root)
manifest=json.loads((training_root/'training_manifest.json').read_text())
train_names=manifest['train_images']
assert len(train_names)==566 and not set(train_names)&set(part)
assert set(manifest['validation_images'])==set(part)
train_records=defaultdict(list)
for im in raw['images']:
    if im['file_name'] in set(train_names): train_records[im['file_name']].append(im['id'])
assert set(train_records)==set(train_names)
rng=random.Random(20261016)
model.requires_grad_(True)
model.backbone.requires_grad_(False)
parameters=[p for p in model.parameters() if p.requires_grad]
assert parameters
optimizer=torch.optim.AdamW([
    {'params':[p for p in parameters if p.ndim>1],'weight_decay':.05},
    {'params':[p for p in parameters if p.ndim<=1],'weight_decay':0.}],lr=BASE_LR)
scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
PATIENCE=3
no_improvement=0

def sample_training():
    name=rng.choice(train_names)
    sid=rng.choice(train_records[name])
    image=utils.read_image(str(training_root/'train_images'/name),format=cfg.INPUT.FORMAT)
    y,x=rng.randrange(2048-CROP_SIZE+1),rng.randrange(2048-CROP_SIZE+1)
    image=image[y:y+CROP_SIZE,x:x+CROP_SIZE]
    masks=[]
    for a in gt[sid]:
        mask=mu.decode(mask_rle(a['segmentation'],2048,2048))[y:y+CROP_SIZE,x:x+CROP_SIZE]
        if mask.any(): masks.append(mask)
    masks=np.stack(masks) if masks else np.zeros((0,CROP_SIZE,CROP_SIZE),np.uint8)
    if rng.random()<.5: image=image[:,::-1];masks=masks[:,:,::-1]
    if rng.random()<.5: image=image[::-1];masks=masks[:,::-1,:]
    target=Instances((CROP_SIZE,CROP_SIZE))
    target.gt_classes=torch.zeros(len(masks),dtype=torch.int64)
    target.gt_masks=torch.from_numpy(np.ascontiguousarray(masks)).bool()
    return {'image':torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1))),
            'instances':target,'height':CROP_SIZE,'width':CROP_SIZE},name

def evaluate_checkpoint(split):
    model.eval()
    cache={};rows=[]
    with torch.inference_mode():
        for i,name in enumerate(sorted(n for n in part if part[n]==split),1):
            p=disjoint(candidates(train_dir/name,'native'))
            cache[name]=[x['rle'] for x in p if x['score']>=.5]
    for r in records:
        if part[r['file_name']]!=split:continue
        truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[r['source_image_id']]]
        pred=cache[r['file_name']]
        iou=mu.iou(truth,pred,[0]*len(pred)) if pred else np.zeros((len(truth),0))
        rows.append(pq_counts(iou))
    return summarize_counts(rows)

def load_current_format_weights(weights):
    # Plain dicts from the first run lost PyTorch module version metadata.
    # Keys already match this architecture, so suppress legacy key migration
    # by restoring the current architecture's versions, without changing tensors.
    assert set(weights)==set(model.state_dict())
    restored=OrderedDict(weights)
    restored._metadata=copy.deepcopy(model.state_dict()._metadata)
    return model.load_state_dict(restored,strict=True)

def save_checkpoint(path,step,metric):
    original_state=model.state_dict()
    cpu_state=OrderedDict((k,v.detach().cpu()) for k,v in original_state.items())
    cpu_state._metadata=copy.deepcopy(original_state._metadata)
    state={'model':cpu_state,
           'finetune_step':step,'source_iteration':36999,'calibration_metrics':metric,
           'source_checkpoint_sha256':provenance['checkpoint_sha256'],'crop_size':CROP_SIZE}
    if 'last' in path.stem:
        state.update(optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                     random_state=rng.getstate(),torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all(),
                     elapsed_train_seconds=train_elapsed_before+time.monotonic()-train_started)
    temporary=path.with_suffix('.tmp')
    torch.save(state,temporary)
    temporary.replace(path)

baseline_cal=evaluate_checkpoint('calibration')
assert abs(baseline_cal['PQ']-.3137662575115845)<.003,baseline_cal
best_metric=baseline_cal
best_step=0
best_path=OUT/'model_best_finetune.pth'
if best_path.exists():
    previous=torch.load(best_path,map_location='cpu',weights_only=True)
    assert previous['source_checkpoint_sha256']==provenance['checkpoint_sha256']
    assert previous['crop_size']==CROP_SIZE
    best_metric,best_step=previous['calibration_metrics'],previous['finetune_step']
    del previous
else:
    save_checkpoint(best_path,0,best_metric)
history=[]
train_elapsed_before=0.
starting_step=0
last_path=OUT/'model_last_finetune.pth'
if last_path.exists():
    previous=torch.load(last_path,map_location='cpu',weights_only=True)
    assert previous['source_checkpoint_sha256']==provenance['checkpoint_sha256']
    assert previous['crop_size']==CROP_SIZE
    load_current_format_weights(previous['model'])
    optimizer.load_state_dict(previous['optimizer']);scaler.load_state_dict(previous['scaler'])
    rng.setstate(previous['random_state']);torch.set_rng_state(previous['torch_rng'])
    torch.cuda.set_rng_state_all(previous['cuda_rng'])
    train_elapsed_before=previous['elapsed_train_seconds'];starting_step=previous['finetune_step']
    del previous
    history_path=OUT/'training_history.json'
    if history_path.exists():history=json.loads(history_path.read_text())
    print('TRAIN_RESUME',starting_step,'elapsed_train_seconds',round(train_elapsed_before),flush=True)
train_started=time.monotonic()
training_steps=starting_step
stop_reason='max_steps'
print('TRAIN_START',json.dumps(dict(images=566,held_out_images=141,crop=CROP_SIZE,
       max_steps=MAX_STEPS,max_training_seconds=TRAIN_SECONDS,backbone_frozen=True,
       trainable_parameters=sum(p.numel() for p in parameters),baseline_calibration=baseline_cal)),flush=True)
for step in range(starting_step+1,MAX_STEPS+1):
    if train_elapsed_before+time.monotonic()-train_started>=TRAIN_SECONDS:
        stop_reason='training_time_cap';break
    model.train()
    record,name=sample_training()
    optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad(), torch.autocast('cuda',dtype=torch.float16):
        losses=model([record])
        loss=sum(losses.values())
    assert torch.isfinite(loss), {k:float(v) for k,v in losses.items()}
    with torch.enable_grad():
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(parameters,.1)
    old_scale=scaler.get_scale()
    scaler.step(optimizer);scaler.update()
    updated=scaler.get_scale()>=old_scale
    if not updated:
        print('AMP_SKIPPED_UPDATE',step,'scale',scaler.get_scale(),flush=True)
    training_steps=step
    lr=BASE_LR*(.2+.8*(1+math.cos(math.pi*step/MAX_STEPS))/2)
    for group in optimizer.param_groups:group['lr']=lr
    history.append(dict(step=step,loss=float(loss),lr=lr,optimizer_updated=updated,
                        grad_norm=float(norm) if torch.isfinite(norm) else None))
    if step in (1,5,10) or step%50==0:
        print('TRAIN',step,'loss',round(float(loss),4),'elapsed_s',round(time.monotonic()-train_started),
              'peak_vram_gib',round(torch.cuda.max_memory_allocated()/1024**3,3),flush=True)
    del losses,loss,record
    if step%EVAL_EVERY==0 or step==MAX_STEPS:
        metric=evaluate_checkpoint('calibration')
        print('CHECKPOINT',step,json.dumps(metric),flush=True)
        history[-1]['calibration']=metric
        meaningful_gain=metric['PQ']>best_metric['PQ']+.0005
        no_improvement=0 if meaningful_gain else no_improvement+1
        if metric['PQ']>best_metric['PQ']:
            best_metric,best_step=metric,step
            save_checkpoint(OUT/'model_best_finetune.pth',step,metric)
        save_checkpoint(OUT/'model_last_finetune.pth',step,metric)
        (OUT/'training_history.json').write_text(json.dumps(history))
        if no_improvement>=PATIENCE:
            stop_reason='three_evaluations_without_meaningful_improvement'
            print('EARLY_STOP',step,stop_reason,flush=True)
            break
save_checkpoint(OUT/'model_last_finetune.pth',training_steps,None)
best=torch.load(OUT/'model_best_finetune.pth',map_location='cpu',weights_only=True)
load_current_format_weights(best['model']);del best
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
