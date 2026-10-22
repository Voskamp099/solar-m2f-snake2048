import copy, random, warnings, traceback, datetime
from collections import OrderedDict
from detectron2.structures import Instances
warnings.filterwarnings('ignore', category=FutureWarning)
EXPERIMENT_START=time.monotonic()
# Conservative allowance: <=85min GPU wall, <=170 GPU-hours minutes if both T4s charged.
WALL_CAP_SECONDS=5100
TRAIN_SEED=20261018
INIT_SEED=20261017
SOURCE=Path('/kaggle/working/finetune_results/model_best_finetune.pth')
SOURCE_SHA='6c8b0f2752762d9909071f07e0d0a0c4e5e5bda60d642c9a104d8b3f1e68b0e1'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA
base=torch.load(SOURCE,map_location='cpu',weights_only=True)
base_weights=base['model'];del base
model.load_state_dict(base_weights,strict=True)
training_root=Path('/kaggle/working/training_images')
if not (training_root/'training_manifest.json').exists():
    archives=list(Path('/kaggle/input').rglob('training_images.bin'));assert len(archives)==1
    training_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archives[0]) as z:z.extractall(training_root)
manifest=json.loads((training_root/'training_manifest.json').read_text())
train_names=manifest['train_images'];assert len(train_names)==566 and not set(train_names)&set(part)
train_records=defaultdict(list)
for image in raw['images']:
    if image['file_name'] in set(train_names):train_records[image['file_name']].append(image['id'])
MASK_PREFIX='sem_seg_head.predictor.mask_embed.'
assert any(n.startswith(MASK_PREFIX) for n,_ in model.named_parameters())
def emit(marker,data):
    line=marker+' '+json.dumps(data,ensure_ascii=False)
    print(line,flush=True)
    with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(dict(marker=marker,data=data,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
def write_json(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def pack_outputs():
    dst=OUT.with_suffix('.zip');tmp=dst.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.iterdir()):
            if f.is_file() and f.suffix!='.tmp':z.write(f,f.name)
    tmp.replace(dst)
def cleanup():
    h=model.sem_seg_head
    if hasattr(h,'_snake_feature_handle'):
        h._snake_feature_handle.remove();del h._snake_feature_handle
        del h.snake_mask_adapter,h.snake_scale_adapters,h._snake_adapter_config
    model.zero_grad(set_to_none=True)
    gc.collect();torch.cuda.empty_cache()
def create_arm(mode):
    cleanup();model.load_state_dict(base_weights,strict=True);model.requires_grad_(False)
    torch.manual_seed(INIT_SEED)
    config=dict(mode='none',mask_features=False,scales=[])
    if mode!='head_only':config=attach_feature_adapters(model,mode=mode,mask_features=True,scales=(),bottleneck=32,kernel_size=9)
    model.sem_seg_head.predictor.mask_embed.requires_grad_(True)
    return config
def optimizer_for_arm():
    groups=[]
    for label,lr,predicate in [('mask_embed',1e-6,lambda n:n.startswith(MASK_PREFIX)),('gate',1e-3,lambda n:n.endswith('.alpha')),('adapter',1e-4,lambda n:'.snake_' in n and not n.endswith('.alpha'))]:
        params=[p for n,p in model.named_parameters() if p.requires_grad and predicate(n)]
        if params:groups.append(dict(params=params,lr=lr,group_name=label))
    params=[p for p in model.parameters() if p.requires_grad]
    assert len(params)==sum(len(g['params']) for g in groups)
    return torch.optim.AdamW(groups,weight_decay=.01),params
schedule=[];schedule_rng=random.Random(TRAIN_SEED)
for step in range(600):
    name=schedule_rng.choice(train_names);sid=schedule_rng.choice(train_records[name])
    schedule.append(dict(file_name=name,source_image_id=sid,hflip=schedule_rng.random()<.5,vflip=schedule_rng.random()<.5))
write_json(OUT/'training_schedule.json',schedule)
def training_item(spec):
    image=utils.read_image(str(training_root/'train_images'/spec['file_name']),format=cfg.INPUT.FORMAT)
    assert image.shape[:2]==(2048,2048)
    masks=[mu.decode(mask_rle(a['segmentation'],2048,2048)) for a in gt[spec['source_image_id']]]
    masks=np.stack(masks) if masks else np.zeros((0,2048,2048),np.uint8)
    if spec['hflip']:image=image[:,::-1];masks=masks[:,:,::-1]
    if spec['vflip']:image=image[::-1];masks=masks[:,::-1,:]
    target=Instances((2048,2048));target.gt_classes=torch.zeros(len(masks),dtype=torch.int64)
    target.gt_masks=torch.from_numpy(np.ascontiguousarray(masks)).bool()
    return dict(image=torch.as_tensor(np.ascontiguousarray(image.transpose(2,0,1))),instances=target,height=2048,width=2048)
def save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history,metrics=None):
    state=model.state_dict()
    delta=OrderedDict((k,v.detach().cpu()) for k,v in state.items() if '.snake_' in k or k.startswith(MASK_PREFIX))
    delta._metadata=copy.deepcopy(state._metadata)
    data=dict(format='mask_embed_and_adapter_delta_v1',delta=delta,adapter_config=config,arm=mode,
              source_checkpoint_sha256=SOURCE_SHA,source_checkpoint=str(SOURCE),effective_updates=updates,attempts=attempts,
              optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
              torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),history=history,metrics=metrics,
              schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),input_size=2048,
              trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
              recovery='Load original base, attach adapter_config when mode!=none, load delta strict=False with exact allowed missing keys, restore optimizer/scaler/RNG, resume schedule[effective_updates:].')
    path=OUT/(mode+'_last.pth');tmp=path.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(path)
    write_json(OUT/(mode+'_history.json'),history)
def verify_frozen():
    state=model.state_dict()
    assert all(torch.equal(v.detach().cpu(),base_weights[k]) for k,v in state.items() if '.snake_' not in k and not k.startswith(MASK_PREFIX)), 'Unexpected frozen-base mutation'
    return dict(frozen_parameters_identical=True,mask_embed_changed=any(not torch.equal(v.detach().cpu(),base_weights[k]) for k,v in state.items() if k.startswith(MASK_PREFIX)),
                offset_nonzero={n:int(torch.count_nonzero(p.detach()).item()) for n,p in model.named_parameters() if '.spatial.offset.weight' in n},
                gates={n:float(p.detach()) for n,p in model.named_parameters() if n.endswith('.alpha')})
def evaluate_arm(mode):
    model.eval();cache={};selected=sorted(n for n in part if part[n]=='calibration');assert len(selected)==70
    with torch.inference_mode():
        for idx,name in enumerate(selected,1):
            assert time.monotonic()-EXPERIMENT_START<WALL_CAP_SECONDS-120,'Evaluation wall cap reached'
            cache[name]={v:candidates(train_dir/name,v) for v in VIEWS}
            write_json(OUT/(mode+'_candidates.json'),cache)
            if idx%10==0:emit('FULL2048_EVAL_PROGRESS',dict(arm=mode,images=idx,total=70,elapsed_seconds=round(time.monotonic()-EXPERIMENT_START)))
    pred={name:[p['rle'] for p in disjoint(consensus(views)) if p['score']>=.9 and p['area']>=256] for name,views in cache.items()}
    counts=[]
    for r in records:
        name=r['file_name']
        if name not in pred:continue
        truth=[mask_rle(a['segmentation'],2048,2048) for a in gt[r['source_image_id']]]
        ps=pred[name];mat=mu.iou(truth,ps,[0]*len(ps)) if ps else np.zeros((len(truth),0))
        counts.append(pq_counts(mat))
    return summarize_counts(counts)
summary=dict(experiment='shared_mask_adapter_plus_mask_embed_full2048',source_sha256=SOURCE_SHA,input_size=2048,
             input_policy='whole native2048, no crop/no downsample, matched H/V flips',planned_effective_updates=600,
             arms=[],preflights=[],baseline_calibration_pq=.4563696594893772,baseline_public_score=.36,
             mask_embed_lr=1e-6,adapter_lr=1e-4,gate_lr=1e-3,weight_decay=.01,grad_clip=1.,
             train_seed=TRAIN_SEED,init_seed=INIT_SEED,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
             calibration_images=70,training_images=566,review_evaluated=False,submitted=False,
             evaluation='full2048FP32,4views,2-viewIoU>.5 consensus,mean score>=.90,area>=256,greedy nonoverlap',
             wall_cap_seconds=WALL_CAP_SECONDS,conservative_max_two_device_quota_hours=(WALL_CAP_SECONDS+300)/1800,
             status='preflight_running',device=torch.cuda.get_device_name(0),visible_devices=torch.cuda.device_count())
write_json(OUT/'summary.json',summary);emit('FULL2048_START',summary)
try:
    # Compare actual full-image inference at initialization, before any updates.
    model.eval();torch.set_grad_enabled(False)
    identity_name=sorted(n for n in part if part[n]=='calibration')[0]
    with torch.inference_mode():before=candidates(train_dir/identity_name,'native')
    create_arm('snake');model.eval()
    with torch.inference_mode():after=candidates(train_dir/identity_name,'native')
    assert before==after,'Zero-gate baseline identity failed'
    summary['actual_zero_gate_full2048_identity']=True
    # Two batches per arm, including densest training annotation; Adam state allocated.
    dense_name,dense_sid=max(((name,sid) for name in train_names for sid in train_records[name]),key=lambda item:len(gt[item[1]]))
    probe_specs=[schedule[0],dict(file_name=dense_name,source_image_id=dense_sid,hflip=False,vflip=False)]
    for mode in ('head_only','fixed_snake','snake'):
        config=create_arm(mode);model.train();model.backbone.eval()
        if mode!='head_only':model.sem_seg_head.snake_mask_adapter.alpha.data.fill_(.01)
        optimizer,params=optimizer_for_arm();scaler=torch.amp.GradScaler('cuda',init_scale=64.)
        torch.manual_seed(TRAIN_SEED);torch.cuda.reset_peak_memory_stats();probe_started=time.monotonic();probe_losses=[]
        for spec in probe_specs:
            item=training_item(spec);optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):losses=model([item]);loss=sum(losses.values())
            assert torch.isfinite(loss),'Nonfinite preflight loss'
            with torch.enable_grad():scaler.scale(loss).backward()
            scaler.unscale_(optimizer);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            probe_losses.append(float(loss.detach()));del item,losses,loss
        probe=dict(arm=mode,status='PASS',input_size=2048,max_gt_instances=len(gt[dense_sid]),losses=probe_losses,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                   free_gib=torch.cuda.mem_get_info()[0]/1024**3,probe_optimizer_steps=2,discarded_before_training=True,
                   elapsed_seconds=time.monotonic()-probe_started,trainable_parameters=sum(p.numel() for p in params))
        summary['preflights'].append(probe);write_json(OUT/'summary.json',summary);emit('FULL2048_PREFLIGHT_PASS',probe)
        assert probe['peak_reserved_gib']<.92*torch.cuda.get_device_properties(0).total_memory/1024**3,'Insufficient memory margin'
        del optimizer,scaler,params;cleanup()
    # Same effective update budget for all arms, chosen by preflight throughput only.
    worst=max(p['elapsed_seconds']/2 for p in summary['preflights'])
    available=WALL_CAP_SECONDS-(time.monotonic()-EXPERIMENT_START)-2100
    common_steps=min(600,int(available/(3*worst*1.20))//50*50)
    assert common_steps>=200,'Not enough bounded time for matched three-arm training'
    summary['effective_updates_per_arm']=common_steps;summary['status']='training';emit('FULL2048_MATCHED_BUDGET',dict(effective_updates=common_steps,estimated_worst_seconds_per_update=worst))
    for mode in ('head_only','fixed_snake','snake'):
        config=create_arm(mode);model.train();model.backbone.eval();torch.manual_seed(TRAIN_SEED);np.random.seed(TRAIN_SEED);random.seed(TRAIN_SEED)
        optimizer,params=optimizer_for_arm();scaler=torch.amp.GradScaler('cuda',init_scale=64.)
        updates=0;attempts=0;history=[];arm_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        while updates<common_steps:
            assert time.monotonic()-EXPERIMENT_START<WALL_CAP_SECONDS-900,'Training wall cap reached'
            assert attempts<common_steps+50,'Excessive AMP retries'
            spec=schedule[updates];item=training_item(spec);optimizer.zero_grad(set_to_none=True);attempts+=1
            with torch.enable_grad(),torch.autocast('cuda',dtype=torch.float16):losses=model([item]);loss=sum(losses.values())
            assert torch.isfinite(loss),'Nonfinite training loss'
            old_scale=scaler.get_scale()
            with torch.enable_grad():scaler.scale(loss).backward()
            scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(optimizer);scaler.update()
            accepted=scaler.get_scale()>=old_scale;updates+=int(accepted)
            entry=dict(attempt=attempts,effective_update=updates,optimizer_updated=bool(accepted),loss=float(loss.detach()),gradient_norm=float(norm),scale=scaler.get_scale(),**spec)
            if mode!='head_only':entry['alpha']=float(model.sem_seg_head.snake_mask_adapter.alpha.detach())
            history.append(entry);del item,losses,loss
            if attempts==1 or attempts%25==0:emit('FULL2048_TRAIN',dict(arm=mode,**entry,elapsed_seconds=round(time.monotonic()-arm_started)))
            if accepted and updates%50==0:save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history)
        training_peak=torch.cuda.max_memory_allocated()/1024**3
        checks=verify_frozen();save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history)
        model.zero_grad(set_to_none=True);metrics=evaluate_arm(mode)
        save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history,metrics)
        result=dict(arm=mode,config=config,effective_updates=updates,attempts=attempts,calibration=metrics,
                    training_peak_allocated_gib=training_peak,trainable_parameters=sum(p.numel() for p in params),
                    integrity=checks,elapsed_seconds=round(time.monotonic()-arm_started))
        summary['arms'].append(result);write_json(OUT/'summary.json',summary);emit('FULL2048_ARM_COMPLETE',result)
        del optimizer,scaler,params;cleanup();pack_outputs()
    summary['status']='complete_three_arms'
except Exception as error:
    summary['status']='error';summary['error_type']=type(error).__name__;summary['error']=str(error);traceback.print_exc()
    emit('FULL2048_ERROR',dict(type=type(error).__name__,message=str(error)))
    if 'optimizer' in globals() and 'history' in globals():
        save_checkpoint(mode,config,updates,attempts,optimizer,scaler,history)
finally:
    summary['elapsed_seconds']=round(time.monotonic()-EXPERIMENT_START)
    summary['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(OUT/'summary.json',summary);emit('FULL2048_COMPLETE',summary)
    model.eval();model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();pack_outputs()
    print('FULL2048_ARCHIVE_READY',str(OUT.with_suffix('.zip')),flush=True)
